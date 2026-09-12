/**
 * @file test_autograd_v2.cpp
 * @brief v2 AD 引擎（AutoGrad 命名空间）的单元测试
 * @details 覆盖加/减/乘/除/矩阵乘/激活函数/损失函数的反向传播正确性
 */

#include "AutoGrad.h"
#include "Tensor.h"
#include "Arena.h"
#include "AutoGrad/Nodes/AddNode.h"
#include "../../../kernels/kernels.h"
#include "C3/C3Cleanup.h"
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

namespace {

constexpr float kEps = 1e-5f;

int g_passed = 0;
int g_failed = 0;
DeviceType g_device = DeviceType::kCPU;

const char* deviceName(DeviceType dev) {
    switch (dev) {
        case DeviceType::kCPU: return "CPU";
        case DeviceType::kCUDA: return "CUDA";
        case DeviceType::kMPS: return "MPS";
        case DeviceType::kAMX: return "AMX";
        case DeviceType::kSIMD: return "SIMD";
        case DeviceType::kUNKNOWN: return "Unknown";
        case DeviceType::kGENERAL: return "General";
        case DeviceType::kCount: return "Count";
        default: return "Unknown";
    }
}

// 在读取 MPS buffer 前显式同步，确保 GPU kernel 已完成写入
void syncDevice(DeviceType dev) {
#ifdef __APPLE__
    if (dev == DeviceType::kMPS) {
        MPS_flush_wait(true);
    }
#endif
}

#define EXPECT(cond, msg) do { \
    if (cond) { ++g_passed; } else { ++g_failed; std::cerr << "[FAIL][" << deviceName(g_device) << "] " << msg << std::endl; } \
} while (0)

#define EXPECT_NEAR_F(a, b, eps) do { \
    float av = (a), bv = (b); \
    if (std::fabs(av - bv) <= (eps)) { ++g_passed; } \
    else { ++g_failed; std::cerr << "[FAIL][" << deviceName(g_device) << "] expected " << av << " ≈ " << bv << " (|diff| > " << (eps) << ")" << std::endl; } \
} while (0)

Tensor makeTensor(std::initializer_list<float> values) {
    Tensor t(values, g_device);
    return t;
}

Tensor makeTensor2D(const std::vector<float>& values, size_t rows, size_t cols) {
    Tensor t(ShapeTag{}, {rows, cols}, DType::kFloat, g_device);
    std::copy(values.begin(), values.end(), t.data_write<float>());
    return t;
}

void test_add_grad() {
    AutoGrad::EnableGrad = true;
    Tensor a = makeTensor({2.0f, 3.0f, 4.0f});
    Tensor b = makeTensor({10.0f, 20.0f, 30.0f});
    a.requires_grad(true);
    b.requires_grad(true);
    Tensor c = a + b;
    AutoGrad::backward(c.getRelatedNode(), false);
    syncDevice(g_device);
    Tensor ga = a.grad();
    Tensor gb = b.grad();
    const float* ga_p = ga.data_read<float>();
    const float* gb_p = gb.data_read<float>();
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR_F(ga_p[i], 1.0f, kEps);
        EXPECT_NEAR_F(gb_p[i], 1.0f, kEps);
    }
}

void test_mul_grad() {
    AutoGrad::EnableGrad = true;
    Tensor a = makeTensor({2.0f, 3.0f});
    Tensor b = makeTensor({4.0f, 5.0f});
    a.requires_grad(true);
    b.requires_grad(true);
    Tensor c = a * b;
    AutoGrad::backward(c.getRelatedNode(), false);
    syncDevice(g_device);
    Tensor ga = a.grad();
    Tensor gb = b.grad();
    const float* ga_p = ga.data_read<float>();
    const float* gb_p = gb.data_read<float>();
    EXPECT_NEAR_F(ga_p[0], 4.0f, kEps); EXPECT_NEAR_F(ga_p[1], 5.0f, kEps);
    EXPECT_NEAR_F(gb_p[0], 2.0f, kEps); EXPECT_NEAR_F(gb_p[1], 3.0f, kEps);
}

void test_sub_grad() {
    AutoGrad::EnableGrad = true;
    Tensor a = makeTensor({5.0f, 7.0f});
    Tensor b = makeTensor({2.0f, 3.0f});
    a.requires_grad(true);
    b.requires_grad(true);
    Tensor c = a - b;
    AutoGrad::backward(c.getRelatedNode(), false);
    syncDevice(g_device);
    Tensor ga = a.grad();
    Tensor gb = b.grad();
    const float* ga_p = ga.data_read<float>();
    const float* gb_p = gb.data_read<float>();
    EXPECT_NEAR_F(ga_p[0], 1.0f, kEps); EXPECT_NEAR_F(ga_p[1], 1.0f, kEps);
    EXPECT_NEAR_F(gb_p[0], -1.0f, kEps); EXPECT_NEAR_F(gb_p[1], -1.0f, kEps);
}

void test_div_grad() {
    AutoGrad::EnableGrad = true;
    Tensor a = makeTensor({6.0f, 8.0f});
    Tensor b = makeTensor({2.0f, 4.0f});
    a.requires_grad(true);
    b.requires_grad(true);
    Tensor c = a / b;
    AutoGrad::backward(c.getRelatedNode(), false);
    syncDevice(g_device);
    Tensor ga = a.grad();
    Tensor gb = b.grad();
    const float* ga_p = ga.data_read<float>();
    const float* gb_p = gb.data_read<float>();
    EXPECT_NEAR_F(ga_p[0], 0.5f, kEps);
    EXPECT_NEAR_F(ga_p[1], 0.25f, kEps);
    EXPECT_NEAR_F(gb_p[0], -1.5f, kEps);
    EXPECT_NEAR_F(gb_p[1], -0.5f, kEps);
}

void test_matmul_grad() {
    AutoGrad::EnableGrad = true;
    Tensor A = makeTensor2D({1, 2, 3, 4, 5, 6}, 2, 3);
    Tensor B = makeTensor2D({7, 8, 9, 10, 11, 12}, 3, 2);
    A.requires_grad(true);
    B.requires_grad(true);
    Tensor C = A.matmul(B);
    AutoGrad::backward(C.getRelatedNode(), false);
    syncDevice(g_device);

    // C = A(2x3) * B(3x2) = 2x2
    // grad_C 全 1 (2x2)
    // grad_A = grad_C * B^T = (2x2) * (2x3) = 2x3
    //   = [[15, 19, 23], [15, 19, 23]]
    // grad_B = A^T * grad_C = (3x2) * (2x2) = 3x2
    //   = [[5, 5], [7, 7], [9, 9]]
    const float expected_grad_A[] = {15, 19, 23, 15, 19, 23};
    const float expected_grad_B[] = {5, 5, 7, 7, 9, 9};
    Tensor ga = A.grad();
    Tensor gb = B.grad();
    const float* ga_p = ga.data_read<float>();
    const float* gb_p = gb.data_read<float>();
    for (int i = 0; i < 6; ++i) {
        EXPECT_NEAR_F(ga_p[i], expected_grad_A[i], kEps);
        EXPECT_NEAR_F(gb_p[i], expected_grad_B[i], kEps);
    }
}

// 关键回归测试：两个输入都不需要梯度时，dispatch 不应再无条件 requires_grad(true)
void test_scheduler_no_grad_propagation() {
    AutoGrad::EnableGrad = true;
    Tensor a = makeTensor({1.0f, 2.0f});
    Tensor b = makeTensor({3.0f, 4.0f});
    Tensor c = a + b;
    EXPECT(!c.requires_grad(), "no-grad inputs should not produce grad output");
}

// 至少一个输入需要梯度时，应该记录计算图
void test_scheduler_grad_propagation() {
    AutoGrad::EnableGrad = true;
    Tensor a = makeTensor({1.0f, 2.0f});
    Tensor b = makeTensor({3.0f, 4.0f});
    a.requires_grad(true);
    Tensor c = a + b;
    EXPECT(c.requires_grad(), "any grad input should propagate to output");
    EXPECT(c.getRelatedNode().get() != nullptr, "graph node should be attached");
}

void test_relu_grad() {
    AutoGrad::EnableGrad = true;
    Tensor a = makeTensor({-2.0f, 0.0f, 3.0f, -1.0f, 5.0f});
    a.requires_grad(true);
    Tensor b = a.relu();
    AutoGrad::backward(b.getRelatedNode(), false);
    syncDevice(g_device);
    Tensor ga = a.grad();
    const float* ga_p = ga.data_read<float>();
    EXPECT_NEAR_F(ga_p[0], 0.0f, kEps);
    EXPECT_NEAR_F(ga_p[1], 0.0f, kEps);
    EXPECT_NEAR_F(ga_p[2], 1.0f, kEps);
    EXPECT_NEAR_F(ga_p[3], 0.0f, kEps);
    EXPECT_NEAR_F(ga_p[4], 1.0f, kEps);
}

void test_lrelu_grad() {
    AutoGrad::EnableGrad = true;
    Tensor a = makeTensor({-2.0f, 0.0f, 3.0f, -1.0f, 5.0f});
    a.requires_grad(true);
    Tensor b = a.leaky_relu(0.01f);
    AutoGrad::backward(b.getRelatedNode(), false);
    syncDevice(g_device);
    Tensor ga = a.grad();
    const float* ga_p = ga.data_read<float>();
    EXPECT_NEAR_F(ga_p[0], 0.01f, kEps);
    EXPECT_NEAR_F(ga_p[1], 0.01f, kEps);
    EXPECT_NEAR_F(ga_p[2], 1.0f, kEps);
    EXPECT_NEAR_F(ga_p[3], 0.01f, kEps);
    EXPECT_NEAR_F(ga_p[4], 1.0f, kEps);
}

void test_neg_grad() {
    AutoGrad::EnableGrad = true;
    Tensor a = makeTensor({2.0f, -3.0f, 5.0f});
    a.requires_grad(true);
    Tensor b = -a;
    AutoGrad::backward(b.getRelatedNode(), false);
    syncDevice(g_device);
    Tensor ga = a.grad();
    const float* ga_p = ga.data_read<float>();
    EXPECT_NEAR_F(ga_p[0], -1.0f, kEps);
    EXPECT_NEAR_F(ga_p[1], -1.0f, kEps);
    EXPECT_NEAR_F(ga_p[2], -1.0f, kEps);
}

void test_sin_grad() {
    AutoGrad::EnableGrad = true;
    Tensor a = makeTensor({0.0f, 1.5707963f, 3.1415926f});
    a.requires_grad(true);
    Tensor b = a.sin();
    AutoGrad::backward(b.getRelatedNode(), false);
    syncDevice(g_device);
    Tensor ga = a.grad();
    const float* ga_p = ga.data_read<float>();
    EXPECT_NEAR_F(ga_p[0], 1.0f, kEps);
    EXPECT_NEAR_F(ga_p[1], 0.0f, kEps);
    EXPECT_NEAR_F(ga_p[2], -1.0f, kEps);
}

void test_cos_grad() {
    AutoGrad::EnableGrad = true;
    Tensor a = makeTensor({0.0f, 1.5707963f, 3.1415926f});
    a.requires_grad(true);
    Tensor b = a.cos();
    AutoGrad::backward(b.getRelatedNode(), false);
    syncDevice(g_device);
    Tensor ga = a.grad();
    const float* ga_p = ga.data_read<float>();
    EXPECT_NEAR_F(ga_p[0], 0.0f, kEps);
    EXPECT_NEAR_F(ga_p[1], -1.0f, kEps);
    EXPECT_NEAR_F(ga_p[2], 0.0f, kEps);
}

void test_tanh_grad() {
    AutoGrad::EnableGrad = true;
    Tensor a = makeTensor({0.0f, 1.0f, -1.0f});
    a.requires_grad(true);
    Tensor b = a.tanh();
    AutoGrad::backward(b.getRelatedNode(), false);
    syncDevice(g_device);
    Tensor ga = a.grad();
    const float* ga_p = ga.data_read<float>();
    float t0 = std::tanh(0.0f);
    float t1 = std::tanh(1.0f);
    float t2 = std::tanh(-1.0f);
    
    EXPECT_NEAR_F(ga_p[0], 1.0f - t0 * t0, kEps);
    EXPECT_NEAR_F(ga_p[1], 1.0f - t1 * t1, kEps);
    EXPECT_NEAR_F(ga_p[2], 1.0f - t2 * t2, kEps);
}

void test_sigmoid_grad() {
    AutoGrad::EnableGrad = true;
    Tensor a = makeTensor({0.0f, 1.0f, -1.0f});
    a.requires_grad(true);
    Tensor b = a.sigmoid();
    AutoGrad::backward(b.getRelatedNode(), false);
    syncDevice(g_device);
    Tensor ga = a.grad();
    const float* ga_p = ga.data_read<float>();
    float s0 = 1.0f / (1.0f + std::exp(-0.0f));
    float s1 = 1.0f / (1.0f + std::exp(-1.0f));
    float s2 = 1.0f / (1.0f + std::exp(1.0f));
    EXPECT_NEAR_F(ga_p[0], s0 * (1.0f - s0), kEps);
    EXPECT_NEAR_F(ga_p[1], s1 * (1.0f - s1), kEps);
    EXPECT_NEAR_F(ga_p[2], s2 * (1.0f - s2), kEps);
}

void test_memory_grad_accumulator_safety() {
    AutoGrad::EnableGrad = true;
    std::shared_ptr<Tensor> a_ptr = std::make_shared<Tensor>(makeTensor({2.0f, 3.0f}));
    a_ptr->requires_grad(true);
    Tensor b = makeTensor({10.0f, 20.0f});
    b.requires_grad(true);
    Tensor c = (*a_ptr) + b;
    a_ptr.reset();
    AutoGrad::backward(c.getRelatedNode(), false);
    EXPECT(true, "GradAccumulator with weak_ptr should not crash");
}

void test_memory_tensor_copy_grad_independence() {
    AutoGrad::EnableGrad = true;
    Tensor a = makeTensor({1.0f, 2.0f});
    a.requires_grad(true);
    Tensor b = makeTensor({10.0f, 20.0f});
    b.requires_grad(true);
    Tensor c = a + b;
    AutoGrad::backward(c.getRelatedNode(), false);
    syncDevice(g_device);
    Tensor a_copy = a;
    auto grad_before = a.grad().data_read<float>();
    auto grad_copy_before = a_copy.grad().data_read<float>();
    EXPECT(grad_before[0] == grad_copy_before[0], "Initial grads should be equal");
    Tensor d = a_copy * makeTensor({2.0f, 3.0f});
    AutoGrad::backward(d.getRelatedNode(), false);
    syncDevice(g_device);
    auto grad_after = a.grad().data_read<float>();
    auto grad_copy_after = a_copy.grad().data_read<float>();
    EXPECT(grad_after[0] != grad_copy_after[0], "Grads should be independent after copy");
}

void test_memory_arena_clear() {
    Arena& arena = Arena::getInstance();
    auto node = arena.invoke<AddNode>(std::vector<std::shared_ptr<Node>>(), std::vector<Tensor>());
    EXPECT(node != nullptr, "Arena should allocate node");
    arena.clear();
    EXPECT(arena.invoke<AddNode>(std::vector<std::shared_ptr<Node>>(), std::vector<Tensor>()) != nullptr, 
           "Arena should work after clear");
}

void test_memory_tensor_move_grad() {
    AutoGrad::EnableGrad = true;
    Tensor a = makeTensor({1.0f, 2.0f});
    a.requires_grad(true);
    Tensor b = makeTensor({10.0f, 20.0f});
    b.requires_grad(true);
    Tensor c = a + b;
    AutoGrad::backward(c.getRelatedNode(), false);
    syncDevice(g_device);
    Tensor d = std::move(a);
    EXPECT(d.grad().numel() == 2, "Moved tensor should retain grad");
}

// [Fix §4.95 P1-03] move 后 GradAccumulator 的 weak_ptr 必须指向新控制块:
// move 之后再 backward, 梯度应正常写回 moved 张量。
// (修复前 weak_ptr 绑旧控制块 → lock() 失败 → 梯度静默丢失)
void test_tensor_move_then_backward() {
    AutoGrad::EnableGrad = true;
    Tensor a = makeTensor({1.0f, 2.0f});
    a.requires_grad(true);
    Tensor b = makeTensor({10.0f, 20.0f});
    b.requires_grad(true);

    // 模拟容器扩容触发的 move 链
    std::vector<Tensor> sink;
    sink.push_back(std::move(a));
    Tensor moved_a = std::move(sink[0]);

    Tensor c = moved_a + b;
    AutoGrad::backward(c.getRelatedNode(), false);
    syncDevice(g_device);

    EXPECT(moved_a.grad().numel() == 2, "moved tensor should receive grad after backward");
    if (moved_a.grad().numel() == 2) {
        EXPECT_NEAR_F(moved_a.grad().data_read<float>()[0], 1.0f, kEps);
        EXPECT_NEAR_F(moved_a.grad().data_read<float>()[1], 1.0f, kEps);
    }
}

void test_tensor_to_same_device() {
    Tensor a = makeTensor({1.0f, 2.0f, 3.0f});
    a.requires_grad(true);
    Tensor b = a.to(g_device);

    EXPECT(b.shape() == a.shape(), "to(same_device) should preserve shape");
    EXPECT(b.dtype() == a.dtype(), "to(same_device) should preserve dtype");
    EXPECT(b.device() == g_device, "to(same_device) should keep device");
    EXPECT(b.requires_grad() == a.requires_grad(),
           "to(same_device) should preserve requires_grad");

    syncDevice(g_device);
    const float* a_p = a.data_read<float>();
    const float* b_p = b.data_read<float>();
    EXPECT_NEAR_F(a_p[0], b_p[0], kEps);
    EXPECT_NEAR_F(a_p[1], b_p[1], kEps);
    EXPECT_NEAR_F(a_p[2], b_p[2], kEps);
}

void test_tensor_to_dtype() {
    Tensor a = makeTensor({1.0f, 2.0f, 3.0f});
    Tensor b = a.to(DType::kDouble);
    Tensor c = b.to(DType::kFloat);

    EXPECT(b.dtype() == DType::kDouble, "to(kDouble) should set dtype");
    EXPECT(c.dtype() == DType::kFloat, "to(kFloat) should restore dtype");
    EXPECT(c.shape() == a.shape(), "dtype conversion should preserve shape");

    syncDevice(g_device);
    const float* a_p = a.data_read<float>();
    const float* c_p = c.data_read<float>();
    EXPECT_NEAR_F(a_p[0], c_p[0], kEps);
    EXPECT_NEAR_F(a_p[1], c_p[1], kEps);
    EXPECT_NEAR_F(a_p[2], c_p[2], kEps);
}

void test_tensor_to_cross_device_and_back() {
    if (g_device == DeviceType::kCPU) {
        // CPU<->CPU 已在 same_device 中覆盖
        return;
    }

    Tensor a = makeTensor({1.0f, 2.0f, 3.0f});
    a.requires_grad(true);

    Tensor on_cpu = a.to(DeviceType::kCPU);
    EXPECT(on_cpu.device() == DeviceType::kCPU, "to(kCPU) should move to CPU");
    EXPECT(on_cpu.shape() == a.shape(), "cross-device to should preserve shape");
    EXPECT(on_cpu.requires_grad() == a.requires_grad(),
           "cross-device to should preserve requires_grad");

    // CPU 上可直接读取；对 MPS 源张量，to(kCPU) 内部若走 memcpy 仍需同步
    syncDevice(DeviceType::kCPU);
    const float* cpu_p = on_cpu.data_read<float>();
    EXPECT_NEAR_F(cpu_p[0], 1.0f, kEps);
    EXPECT_NEAR_F(cpu_p[1], 2.0f, kEps);
    EXPECT_NEAR_F(cpu_p[2], 3.0f, kEps);

    Tensor back = on_cpu.to(g_device);
    EXPECT(back.device() == g_device, "to(original_device) should move back");
    syncDevice(g_device);
    const float* back_p = back.data_read<float>();
    EXPECT_NEAR_F(back_p[0], 1.0f, kEps);
    EXPECT_NEAR_F(back_p[1], 2.0f, kEps);
    EXPECT_NEAR_F(back_p[2], 3.0f, kEps);
}

} // namespace

void run_all_tests() {
    test_add_grad();
    test_mul_grad();
    test_sub_grad();
    test_div_grad();
    test_matmul_grad();
    test_scheduler_no_grad_propagation();
    test_scheduler_grad_propagation();
    test_relu_grad();
    test_lrelu_grad();
    test_neg_grad();
    test_sin_grad();
    test_cos_grad();
    test_tanh_grad();
    test_sigmoid_grad();
    test_memory_grad_accumulator_safety();
    test_memory_tensor_copy_grad_independence();
    test_memory_arena_clear();
    test_memory_tensor_move_grad();
    test_tensor_move_then_backward();
    test_tensor_to_same_device();
    test_tensor_to_dtype();
    test_tensor_to_cross_device_and_back();
}

int main() {
    std::cout << "=== v2 AD 引擎单元测试 ===" << std::endl;

    // 提前初始化 Scheduler / Allocator，确保 MPS 等后端 allocator 在创建张量前已注册
    CtorchScheduler::getInstance();

    // 在可用的独立设备上运行测试（CPU/MPS）；SIMD/AMX作为优化路径自动处理
    // [Fix 2026-08-13] 只测试独立设备，SIMD/AMX是优化路径不是独立设备
    const DeviceType devices[] = {
        DeviceType::kCPU,
        DeviceType::kMPS
    };

    for (DeviceType dev : devices) {
        if (!CtorchScheduler::isDeviceAvailable(dev)) {
            std::cout << "[SKIP] 设备不可用: " << deviceName(dev) << std::endl;
            continue;
        }
        g_device = dev;
        std::cout << "\n--- 设备: " << deviceName(dev) << " ---" << std::endl;
        run_all_tests();
    }

    std::cout << "\n通过: " << g_passed << " / 失败: " << g_failed << std::endl;
    
    // 优雅清理 C3，避免静态析构期的 recursive_mutex lock failed
    ct::c3::shutdownAll();
    
    return g_failed == 0 ? 0 : 1;
}
