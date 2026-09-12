/**
 * @file test_c3_pgo_deopt.cpp
 * @brief 验证 PGOCompiledKernel deoptimization 机制（ADR-006）
 * @details 覆盖场景：
 *   1. 基线：正常 kernel 不会 deopt，deoptCount==0
 *   2. Ofast 运行时抛异常 → 自动 disable + 降级到 O2
 *   3. O2 运行时抛异常 → 自动 disable + 降级到 Eager
 *   4. Disable 后不重试崩溃 kernel（连续多次 execute，deoptCount 不变）
 *   5. lastDeoptReason 包含 tier 标签 + 原因
 *   6. isOfastDisabled / isO2Disabled 状态可观察
 *
 * 实现策略：
 *   - 通过 PGOCompiledKernelTestAccess friend 注入 mock kernel
 *   - Mock kernel 模拟"运行 N 次后抛 std::runtime_error"
 *
 * @date 2026/8/3
 */

#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
#include <cstring>

#include "Tensor.h"
#include "C3/Graph.h"
#include "C3/C3Engine.h"
#include "C3/PGOManager.h"

using namespace ct;
using namespace ct::c3;

// ====================== Mock Kernel ======================
//
// CrashingMockKernel：每次 execute 都抛 std::runtime_error
// 用于模拟 O2/Ofast 运行时崩溃
class CrashingMockKernel : public CompiledKernel {
public:
    explicit CrashingMockKernel(const std::string& name) : name_(name) {}
    std::vector<Tensor> execute(const std::vector<Tensor>& inputs) override {
        (void)inputs;
        throw std::runtime_error("Mock kernel '" + name_ + "' simulated crash");
    }
    [[nodiscard]] const std::string& cacheKey() const override { return name_; }
    [[nodiscard]] DeviceType targetDevice() const override { return DeviceType::kCPU; }
    [[nodiscard]] size_t workspaceBytes() const override { return 0; }
    bool installIntoRegistry(op, const KernelShapeInfo&,
                             std::shared_ptr<CompiledKernel>) override { return false; }
private:
    std::string name_;
};

// CrashingAfterNKernel：前 N 次成功，之后抛异常
// 用于测试"前几次成功、之后崩溃"场景
class CrashingAfterNKernel : public CompiledKernel {
public:
    CrashingAfterNKernel(const std::string& name, int crash_after)
        : name_(name), crash_after_(crash_after), call_count_(0) {}
    std::vector<Tensor> execute(const std::vector<Tensor>& inputs) override {
        ++call_count_;
        if (call_count_ > crash_after_) {
            throw std::runtime_error("Mock kernel '" + name_ +
                                     "' crashed after " + std::to_string(crash_after_) + " calls");
        }
        // 简单返回 input[0]（确保 shape 一致）
        return inputs;
    }
    [[nodiscard]] const std::string& cacheKey() const override { return name_; }
    [[nodiscard]] DeviceType targetDevice() const override { return DeviceType::kCPU; }
    [[nodiscard]] size_t workspaceBytes() const override { return 0; }
    bool installIntoRegistry(op, const KernelShapeInfo&,
                             std::shared_ptr<CompiledKernel>) override { return false; }
    int callCount() const { return call_count_; }
private:
    std::string name_;
    int crash_after_;
    int call_count_;
};

// IdentityMockKernel：透传输入，不抛异常（用于基线测试）
class IdentityMockKernel : public CompiledKernel {
public:
    explicit IdentityMockKernel(const std::string& name) : name_(name) {}
    std::vector<Tensor> execute(const std::vector<Tensor>& inputs) override {
        return inputs;
    }
    [[nodiscard]] const std::string& cacheKey() const override { return name_; }
    [[nodiscard]] DeviceType targetDevice() const override { return DeviceType::kCPU; }
    [[nodiscard]] size_t workspaceBytes() const override { return 0; }
    bool installIntoRegistry(op, const KernelShapeInfo&,
                             std::shared_ptr<CompiledKernel>) override { return false; }
private:
    std::string name_;
};

// ====================== Test Accessor ======================
//
// 通过 friend PGOCompiledKernelTestAccess 注入 mock kernel 到 private 字段
//
// 注意：必须放在 ct::c3 命名空间内，否则 PGOManager.h 里的 friend 声明
//       `friend class PGOCompiledKernelTestAccess;` 不会匹配全局命名空间。
namespace ct {
namespace c3 {

class PGOCompiledKernelTestAccess {
public:
    static void setO2Kernel(PGOCompiledKernel* p, std::shared_ptr<CompiledKernel> k) {
        if (!p) return;
        std::lock_guard<std::mutex> lock(p->compile_mutex_);
        p->o2_kernel_ = std::move(k);
    }
    static void setOfastKernel(PGOCompiledKernel* p, std::shared_ptr<CompiledKernel> k) {
        if (!p) return;
        std::lock_guard<std::mutex> lock(p->compile_mutex_);
        p->ofast_kernel_ = std::move(k);
    }
};

}  // namespace c3
}  // namespace ct

// ====================== 简单图构造 ======================
static Graph buildSimpleAddGraph() {
    Graph g;
    auto desc = TensorDesc::fromShape({2, 2});
    size_t a = g.addInput(desc);
    size_t b = g.addInput(desc);
    size_t c = g.addNode(AddNode{desc, desc}, {a, b}, desc);
    g.markOutput(c);
    return g;
}

static void makeInputs(std::vector<Tensor>& inputs) {
    inputs.clear();
    inputs.emplace_back(ShapeTag{}, std::vector<size_t>{2, 2});
    inputs.emplace_back(ShapeTag{}, std::vector<size_t>{2, 2});
    for (auto& t : inputs) {
        std::memset(t.data_write<float>(), 0, t.numel() * sizeof(float));
    }
}

// ====================== 主测试 ======================
int main() {
    CtorchError::setPrintLevel(PrintLevel::MINIUM);
    C3Engine::getInstance().clearCache();

    std::cout << "=== PGOCompiledKernel deoptimization PoC 测试 ===" << std::endl;
    int passed = 0, failed = 0;

    // ============== 测试 1: 基线（无 mock，无崩溃）==============
    {
        Graph g = buildSimpleAddGraph();
        CompileOptions opts;
        opts.opt_level = 0;  // 不触发 PGO 异步编译
        opts.pgo_mode = true;

        auto profile = std::make_shared<ProfileData>();
        PGOCompiledKernel kernel(g, opts, "test-baseline", profile, C3Engine::getInstance());

        std::vector<Tensor> inputs;
        makeInputs(inputs);

        // 注入 IdentityMockKernel 到 O2 + Ofast（模拟"已完成编译"）
        PGOCompiledKernelTestAccess::setO2Kernel(&kernel, std::make_shared<IdentityMockKernel>("O2"));
        PGOCompiledKernelTestAccess::setOfastKernel(&kernel, std::make_shared<IdentityMockKernel>("Ofast"));

        // 执行 3 次
        for (int i = 0; i < 3; ++i) {
            auto result = kernel.execute(inputs);
            if (result.size() != inputs.size()) {
                std::cout << "  FAIL [1]: execute returned wrong size\n";
                ++failed;
                goto test1_done;
            }
        }

        if (kernel.deoptCount() != 0) {
            std::cout << "  FAIL [1]: baseline deoptCount=" << kernel.deoptCount() << " (should be 0)\n";
            ++failed;
        } else {
            std::cout << "  PASS [1]: 基线无 deopt (deoptCount=0)\n";
            ++passed;
        }
        if (kernel.isOfastDisabled() || kernel.isO2Disabled()) {
            std::cout << "  FAIL [1]: baseline should not disable any kernel\n";
            ++failed;
        } else {
            std::cout << "  PASS [1]: 基线无 disable 状态\n";
            ++passed;
        }
    }
    test1_done:;

    // ============== 测试 2: Ofast 崩溃 → 降级到 O2 ==============
    {
        Graph g = buildSimpleAddGraph();
        CompileOptions opts;
        opts.opt_level = 0;
        opts.pgo_mode = true;

        auto profile = std::make_shared<ProfileData>();
        PGOCompiledKernel kernel(g, opts, "test-ofast-crash", profile, C3Engine::getInstance());

        std::vector<Tensor> inputs;
        makeInputs(inputs);

        // O2 = Identity（成功），Ofast = Crashing（必崩）
        PGOCompiledKernelTestAccess::setO2Kernel(&kernel, std::make_shared<IdentityMockKernel>("O2"));
        PGOCompiledKernelTestAccess::setOfastKernel(&kernel, std::make_shared<CrashingMockKernel>("Ofast"));

        // 第一次：Ofast 崩 → deopt → 降级到 O2 → 返回结果
        auto result = kernel.execute(inputs);
        if (result.size() != inputs.size()) {
            std::cout << "  FAIL [2]: 降级后 execute 失败\n";
            ++failed;
        } else if (kernel.deoptCount() != 1) {
            std::cout << "  FAIL [2]: deoptCount=" << kernel.deoptCount() << " (expected 1)\n";
            ++failed;
        } else if (!kernel.isOfastDisabled()) {
            std::cout << "  FAIL [2]: isOfastDisabled() 应为 true\n";
            ++failed;
        } else if (kernel.isO2Disabled()) {
            std::cout << "  FAIL [2]: isO2Disabled() 不应为 true (O2 正常)\n";
            ++failed;
        } else {
            std::cout << "  PASS [2]: Ofast 崩溃 → 自动降级到 O2（deoptCount=1, ofast disabled）\n";
            ++passed;
        }

        // last_deopt_reason 应包含 "ofast:"
        auto reason = kernel.lastDeoptReason();  // [§4.95 P1-05] 现为按值返回
        if (reason.find("ofast") == std::string::npos) {
            std::cout << "  FAIL [2]: lastDeoptReason 应包含 'ofast': " << reason << "\n";
            ++failed;
        } else {
            std::cout << "  PASS [2]: lastDeoptReason 包含 'ofast': " << reason << "\n";
            ++passed;
        }
    }

    // ============== 测试 3: O2 崩溃 → 降级到 Eager ==============
    {
        Graph g = buildSimpleAddGraph();
        CompileOptions opts;
        opts.opt_level = 0;
        opts.pgo_mode = true;

        auto profile = std::make_shared<ProfileData>();
        PGOCompiledKernel kernel(g, opts, "test-o2-crash", profile, C3Engine::getInstance());

        std::vector<Tensor> inputs;
        makeInputs(inputs);

        // O2 = Crashing，Ofast = Crashing（两个都崩）
        PGOCompiledKernelTestAccess::setO2Kernel(&kernel, std::make_shared<CrashingMockKernel>("O2"));
        PGOCompiledKernelTestAccess::setOfastKernel(&kernel, std::make_shared<CrashingMockKernel>("Ofast"));

        // 第一次：Ofast 崩 → deopt(1) → O2 崩 → deopt(2) → 走 Eager
        std::vector<Tensor> result;
        std::string err;
        try {
            result = kernel.execute(inputs);
        } catch (const std::exception& e) {
            err = e.what();
        }
        if (!err.empty()) {
            std::cout << "  FAIL [3]: execute 抛异常: " << err << "\n";
            ++failed;
        } else if (result.size() != 1) {
            std::cout << "  FAIL [3]: result.size()=" << result.size()
                      << " (expected 1: add graph has 1 output)\n";
            ++failed;
        } else if (kernel.deoptCount() != 2) {
            std::cout << "  FAIL [3]: deoptCount=" << kernel.deoptCount() << " (expected 2)\n";
            ++failed;
        } else if (!kernel.isO2Disabled() || !kernel.isOfastDisabled()) {
            std::cout << "  FAIL [3]: O2 + Ofast 都应被 disabled\n";
            ++failed;
        } else {
            std::cout << "  PASS [3]: O2 + Ofast 都崩 → 自动降级到 Eager（deoptCount=2）\n";
            ++passed;
        }
    }

    // ============== 测试 4: Disable 后不重试 ==============
    {
        Graph g = buildSimpleAddGraph();
        CompileOptions opts;
        opts.opt_level = 0;
        opts.pgo_mode = true;

        auto profile = std::make_shared<ProfileData>();
        PGOCompiledKernel kernel(g, opts, "test-no-retry", profile, C3Engine::getInstance());

        std::vector<Tensor> inputs;
        makeInputs(inputs);

        // O2 = CrashingAfterN（前 2 次成功，第 3 次崩），Ofast = Crashing（必崩）
        PGOCompiledKernelTestAccess::setO2Kernel(&kernel, std::make_shared<CrashingAfterNKernel>("O2", 2));
        PGOCompiledKernelTestAccess::setOfastKernel(&kernel, std::make_shared<CrashingMockKernel>("Ofast"));

        // 第 1 次：Ofast 崩 → deopt(1) → O2 成功 → 返回
        // 第 2 次：Ofast 已 disable → O2 成功 → 返回
        // 第 3 次：Ofast 已 disable → O2 崩（call_count > 2）→ deopt(2) → Eager
        // 第 4 次：Ofast disabled + O2 disabled → 直接 Eager（deoptCount 不变）
        for (int i = 0; i < 4; ++i) {
            try {
                auto result = kernel.execute(inputs);
                (void)result;
            } catch (const std::exception& e) {
                std::cout << "  FAIL [4]: execute 第 " << (i+1) << " 次未捕获异常: " << e.what() << "\n";
                ++failed;
                goto test4_done;
            }
        }

        // 第 3 次后 O2 应被 disabled，deoptCount 应为 2
        // 第 4 次不会触发新的 deopt（O2 + Ofast 都已 disabled）
        if (kernel.deoptCount() != 2) {
            std::cout << "  FAIL [4]: deoptCount=" << kernel.deoptCount() << " (expected 2)\n";
            ++failed;
        } else {
            std::cout << "  PASS [4]: 连续 4 次 execute，deoptCount=2（disable 后不重试）\n";
            ++passed;
        }
    }
    test4_done:;

    // ============== 测试 5: lastDeoptReason 验证 ==============
    {
        Graph g = buildSimpleAddGraph();
        CompileOptions opts;
        opts.opt_level = 0;
        opts.pgo_mode = true;

        auto profile = std::make_shared<ProfileData>();
        PGOCompiledKernel kernel(g, opts, "test-reason", profile, C3Engine::getInstance());

        std::vector<Tensor> inputs;
        makeInputs(inputs);

        PGOCompiledKernelTestAccess::setO2Kernel(&kernel, std::make_shared<IdentityMockKernel>("O2"));
        PGOCompiledKernelTestAccess::setOfastKernel(&kernel, std::make_shared<CrashingMockKernel>("Ofast"));

        kernel.execute(inputs);

        auto reason = kernel.lastDeoptReason();  // [§4.95 P1-05] 现为按值返回
        bool has_tier = reason.find("ofast") != std::string::npos;
        bool has_msg = reason.find("simulated crash") != std::string::npos;

        if (has_tier && has_msg) {
            std::cout << "  PASS [5]: lastDeoptReason 完整: " << reason << "\n";
            ++passed;
        } else {
            std::cout << "  FAIL [5]: lastDeoptReason 缺字段: " << reason
                      << " (has_tier=" << has_tier << ", has_msg=" << has_msg << ")\n";
            ++failed;
        }
    }

    std::cout << "\n=== 总计: " << passed << " passed, " << failed << " failed ===" << std::endl;
    return failed == 0 ? 0 : 1;
}
