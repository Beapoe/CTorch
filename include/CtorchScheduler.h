/**
 * @file CtorchScheduler.h
 * @brief Ctorch 框架的核心调度器类
 * @details 采用单例模式实现，负责管理所有 kernel 映射关系，根据算子类型和设备类型，
 *          自动查找并调用对应的 kernel，实现 kernel 的统一调度。
 *          职责单一：只做 kernel 查找和调用，不涉及自动微分。
 *          支持热替换：通过原子操作实现 C3 JIT 编译后的 kernel 在线替换。
 * @author GhostFace
 * @date 2025/12/20
 */
#ifndef CTORCH_SCHEDULER_H
#define CTORCH_SCHEDULER_H
#include <atomic>
#include <array>
#include <deque>
#include <mutex>
#include <optional>
#include <vector>
#include "CtorchError.h"
#include "Tensor.h"
#include "./../src/kernels/kernels.h"
#ifndef CT_DISABLE_C3
#include "C3/C3Config.h"
#include "C3/C3KernelRegistry.h"
#include "C3/C3HotPathManager.h"
#include "C3/RegionFusion.h"
#endif

namespace ct { namespace detail {
// [dispatch 裁剪] 固定 region 候选 op 集合（与 Region pattern 同步）：
//   MatMul/Add/ReLU/Sigmoid/SiLU 才可能构成当前已注册 region，其余 op 直接跳过
//   tryRegionDispatch 调用，避免每次无谓的调度尝试。
//   PEL25 #7: 新增 SiLU (LLaMA/PaLM FFN SwiGLU 第一个 op, 跟 MatMul 构成 region)
//   注意：若未来新增含其他 op 的 region pattern，必须在此补对应 op。
constexpr bool isRegionCandidateOp(op t) {
    return t == op::MatMul || t == op::Add || t == op::ReLU || t == op::Sigmoid || t == op::SiLU;
}
}} // namespace ct::detail

#ifdef C3_DISPATCH_TIMING
#include <chrono>
#endif

// DEBT-NEW-7 性能采样(v0.5.1+ 代码审查用):
// c3 off build (CT_DISABLE_C3=ON) 时,ct::c3::C3KernelRegistry 不可用,
// 改用文件级 inline static 计数器(每个 TU 看到同一份实例)。
#ifdef CT_PROFILE_PERF
namespace ct { namespace detail {
inline std::atomic<uint64_t>& perfEagerNs() {
    static std::atomic<uint64_t> v{0}; return v;
}
inline std::atomic<uint64_t>& perfEagerCount() {
    static std::atomic<uint64_t> v{0}; return v;
}
inline void perfEagerAdd(uint64_t ns) {
    perfEagerNs().fetch_add(ns, std::memory_order_relaxed);
    perfEagerCount().fetch_add(1, std::memory_order_relaxed);
}
inline void perfEagerReset() {
    perfEagerNs().store(0, std::memory_order_relaxed);
    perfEagerCount().store(0, std::memory_order_relaxed);
}
inline std::pair<uint64_t, uint64_t> perfEagerRead() {
    return { perfEagerNs().load(std::memory_order_relaxed),
             perfEagerCount().load(std::memory_order_relaxed) };
}
}} // namespace ct::detail
#endif

#ifndef CT_DISABLE_C3
namespace ct { namespace detail {
/// 实验开关：单 kernel hotpath 注入是否禁用（编译期 CT_C3_DISABLE_SINGLE_KERNEL 宏
/// 或运行时 C3_DISABLE_SINGLE_KERNEL=1 均视为禁用）。禁用后仅保留区域融合。
/// 用于归因 H2 缺陷（单 kernel 渐进注入形成 Eager→JIT 混合轨迹破坏训练一致性）。
inline bool c3SingleKernelDisabled() {
    return !ct::c3::singleKernelInjectionEnabled();
}

/// 反向传播标志：ComputeCore::backward 入口置 true、出口置 false。
/// 用于在 dispatch 中识别反向传播路径（其梯度 MatMul 输入不 requires_grad，
/// 但形状是转置的，走单 kernel 会命中错误形状）。
inline bool& g_in_backward_flag() {
    static thread_local bool in_backward = false;
    return in_backward;
}
inline bool g_in_backward() { return g_in_backward_flag(); }
inline void set_in_backward(bool v) { g_in_backward_flag() = v; }

/// 判断是否处于 autograd 追踪区域（反向传播中，或任一输入 requires_grad）。
/// 方案 A：训练/反向传播期间禁用单 kernel 渐进注入（Eager→JIT 混合轨迹
/// 破坏训练一致性），仅保留区域融合（整体接管序列，不产生逐 op 交替）。
/// 纯推理（EnableGrad=true 且输入不 requires_grad）时保留单 kernel 加速。
inline bool inAutogradScope(bool a_grad, bool b_grad) {
    return g_in_backward() || a_grad || b_grad;
}

/// 诊断开关：C3_DISABLE_OP=<逗号分隔的 op 整数值> 时，跳过这些 op 的单 kernel 注入。
/// 仅用于二分定位"哪个 op 破坏训练轨迹"，排查完移除。
inline bool c3OpDisabled(int op_id) {
    static const std::vector<int> disabled = []() {
        std::vector<int> v;
        const char* env = std::getenv("C3_DISABLE_OP");
        if (env != nullptr) {
            std::istringstream ss(env);
            std::string tok;
            while (std::getline(ss, tok, ',')) {
                if (!tok.empty()) v.push_back(atoi(tok.c_str()));
            }
        }
        return v;
    }();
    for (int d : disabled) if (d == op_id) return true;
    return false;
}
}} // namespace ct::detail
#endif

extern thread_local bool g_in_recomputation;

class CtorchScheduler{

    // ========== [DEPRECATED] Trace-based fusion ==========
    // tryFusedDispatch 已被区域融合 (Region Fusion) 替代。
    // 该实现存在形状匹配过于宽松的 bug（只检查 lhs_shape[0]），
    // 导致融合核输出形状错误，MNIST 准确率从 97.24% 跌至 11.23%。
    // 保留代码作为历史参考，详见 STATUS_CONTEXT.md Bug 5。
    //
    // struct FusedTraceEntry {
    //     op op_type;
    //     std::vector<size_t> input_shapes;
    //     std::vector<size_t> output_shape;
    //     std::vector<Tensor> input_tensors;
    //     DeviceType dev;
    // };
    //
    // static constexpr size_t kFusedTraceMaxLen = 8;
    // std::deque<FusedTraceEntry> fused_trace_;
    // mutable std::mutex fused_trace_mutex_;
    //
    // void recordToFusedTrace(op op_type, const std::vector<size_t>& input_shapes,
    //                        const std::vector<size_t>& output_shape,
    //                        std::vector<Tensor> input_tensors, DeviceType dev);
    //
    // std::optional<Tensor> tryFusedDispatch(op op_type, const std::vector<Tensor>& inputs,
    //                                        const std::vector<size_t>& output_shape,
    //                                        DeviceType dev);
    // =====================================================

    #ifndef CT_DISABLE_C3
    // ======================= 区域融合 trace =======================

    /// 区域融合 trace：记录 dispatch 的 op 类型序列，用于 rolling hash 匹配
    std::vector<op> region_trace_;
    mutable std::mutex region_trace_mutex_;

    // ======================= 区域融合（Region Fusion） =======================

    enum class PrewalkState {
        kIdle,          // 不在预走模式
        kPrewalking,    // 预走进行中
        kFallback       // 预走失败，回退 eager
    };

    /// 预走缓存项
    struct PrewalkEntry {
        op op_type;
        std::vector<Tensor> original_inputs;  // 原始输入（非占位符）
    };

    PrewalkState prewalk_state_ = PrewalkState::kIdle;
    // [Dev 2026-08-11] prewalk_cache_ 固定容量环形缓冲：
    //   - 消除 std::vector 每次 push_back 的 realloc + erase(begin()) 的 O(n) 前移
    //   - prewalk_cache_head_ 指向最旧元素物理下标，prewalk_cache_count_ 为有效元素数
    //   - 逻辑下标 0 = 最旧，prewalkAt(logical) = 物理下标
    // [HPC 2026-08-11] kPrewalkCacheCapacity=8 是 2 的幂，取模改写为位与：
    //   %8 → &7，消除整数除法（慢约 20-50 cycles），纯位运算无分支。
    static constexpr size_t kPrewalkCacheCapacity = 8;
    static constexpr size_t kPrewalkCacheMask   = kPrewalkCacheCapacity - 1;
    static_assert((kPrewalkCacheCapacity & kPrewalkCacheMask) == 0,
                  "kPrewalkCacheCapacity must be a power of two for & mask");
    std::array<PrewalkEntry, kPrewalkCacheCapacity> prewalk_cache_;
    size_t prewalk_cache_head_ = 0;
    size_t prewalk_cache_count_ = 0;
    PrewalkEntry& prewalkAt(size_t logical_idx) {
        return prewalk_cache_[(prewalk_cache_head_ + logical_idx) & kPrewalkCacheMask];
    }
    const PrewalkEntry& prewalkAt(size_t logical_idx) const {
        return prewalk_cache_[(prewalk_cache_head_ + logical_idx) & kPrewalkCacheMask];
    }
    // [Fix 2026-09-10 §4.95 P1-11] 按值持有: registry 返回裸指针在后台 install
    // rehash 后失效(悬垂), 改为锁内拷贝出的 optional<RegionEntry>
    std::optional<ct::c3::RegionEntry> matched_region_;
    size_t prewalk_pos_ = 0;  // 当前预走到的位置（在 region 的 op_seq 中）
    // [Prewalk] 缓存 region 执行所需的 external inputs（按 dispatch 顺序追加）
    std::vector<Tensor> prewalk_external_inputs_;

    /// 缓存上次匹配的 region，避免空闲模式重复计算 hash
    ct::c3::RegionEntry* cached_region_ = nullptr;
    uint64_t cached_hash_ = 0;

    /// 执行区域融合预走检查
    /// @return 非空：区域融合结果（占位符/region 执行结果/回退结果）
    ///         nullopt：继续正常 eager dispatch
    std::optional<Tensor> tryRegionDispatch(op op_type,
                                            const Tensor* inputs, size_t num_inputs,
                                            DeviceType dev);

    /// 计算 op 的输出形状（用于预走占位符）
    std::vector<size_t> computeOutputShape(op op_type,
                                           const Tensor* inputs, size_t num_inputs) const;

    /// [Prewalk] 物化某个中间 op 的占位符：用 eager kernel 重算该 op 的真实值。
    /// @details 供 LazyMaterializer 闭包调用。通过 data_read() 读取输入时，
    ///          会自动链式物化上游占位符（幂等）。返回重算结果。
    Tensor eagerMaterializeOp(op op_type,
                              const std::vector<Tensor>& inputs,
                              DeviceType dev);
#endif

private:
    CtorchScheduler();
    CtorchScheduler(const CtorchScheduler&);
    CtorchScheduler& operator=(const CtorchScheduler&) = delete;

    static constexpr size_t OP_COUNT = static_cast<size_t>(op::kCount);
    static constexpr size_t DEVICE_COUNT = static_cast<size_t>(DeviceType::kCount);

    // ABI 门控：op::kCount / DeviceType::kCount 改变会改变 kernel 查找表维度。
    // 以下硬编码数字是 ABI 变更的强制检查点。新增枚举值时必须同步更新此处，
    // 否则编译失败，防止调度表维度与注册逻辑错位。
    // 详见 ABI_POLICY.md 第 3.2 节“新增算子 ABI 检查清单”。
    static_assert(static_cast<size_t>(op::kCount) == 30,
                  "op::kCount changed. Update this assert and all backend kernel registrations (see ABI_POLICY.md) — PEL25 #6: 28→30 (SiLU + SwiGLU)");
    // [Dev] v0.5.2 DCU 接入: kDCU = 7 加到 DeviceType 后, kCount 改 8
    // 注: DeviceType 跟 ABI_POLICY 一起看, DCU 节点 activation 后重新 review
    static_assert(static_cast<size_t>(DeviceType::kCount) == 8,
                  "DeviceType::kCount changed. Update this assert and all backend kernel registrations (see ABI_POLICY.md)");

    std::array<std::array<std::atomic<BinaryKernelFunc>, DEVICE_COUNT>, OP_COUNT> binary_kernels_{};
    std::array<std::array<std::atomic<UnaryKernelFunc>, DEVICE_COUNT>, OP_COUNT> unary_kernels_{};
    std::array<std::array<std::atomic<UnaryInplaceKernelFunc>, DEVICE_COUNT>, OP_COUNT> unary_inplace_kernels_{};
    std::array<std::atomic<Tensor (*)(const Tensor&, int)>, DEVICE_COUNT> softmax_kernels_{};

    // ========== 线性代数专用 kernel 表（不走 op enum）==========
    // 2026-08-10：Rot (Givens 旋转) 和 ApplyHk (Householder apply)
    // 这些 op 有特殊参数签名（Rot 需要 c,s；ApplyHk 需要 v,tau,k,p），
    // 不适合塞进 op enum，所以单独注册表。
    std::array<std::atomic<RotKernelFunc>, DEVICE_COUNT>     rot_kernels_{};
    std::array<std::atomic<ApplyHkKernelFunc>, DEVICE_COUNT> applyhk_kernels_{};

    void initKernels();

    static BinaryKernelFunc selectBestBinary(op op_type, DeviceType dev,
        const std::array<std::array<std::atomic<BinaryKernelFunc>, DEVICE_COUNT>, OP_COUNT>& table);
    static UnaryKernelFunc selectBestUnary(op op_type, DeviceType dev,
        const std::array<std::array<std::atomic<UnaryKernelFunc>, DEVICE_COUNT>, OP_COUNT>& table);
    static UnaryInplaceKernelFunc selectBestUnaryInplace(op op_type, DeviceType dev,
        const std::array<std::array<std::atomic<UnaryInplaceKernelFunc>, DEVICE_COUNT>, OP_COUNT>& table);
public:
    static CtorchScheduler& getInstance() {
        static CtorchScheduler instance_;
        return instance_;
    }

    static bool isDeviceAvailable(DeviceType dev_type);

    static DeviceType getTargetDevice(const Tensor& a, const Tensor& b) {
        if (a.device() != b.device()) {
            CtorchError::log(ErrorLevel::ERROR,ErrorPlatform::kGENERAL,ErrorType::DEVICE_COMPAT,"Ctorch_Scheduler: Tensor不在同一平台");
        }
        return a.device();
    }

    // P1-3: MPS in-place unary kernel memory overlap 支持。
    // 仅对逐元素 unary 算子返回 true；Softmax/Sum/Min/Max 等含规约或二元语义的算子返回 false。
    static bool supports_unary_memory_overlap(DeviceType dev, op op_type) {
        (void)dev;
        switch (op_type) {
            case op::Neg:
            case op::Cos:
            case op::Sin:
            case op::ReLU:
            case op::Tanh:
            case op::Sigmoid:
            case op::GELU:
            case op::LReLU:
            case op::Log:
            case op::Exp:
            case op::Abs:
                return true;
            default:
                return false;
        }
    }

    BinaryKernelFunc get_binary_kernel(op op_type, DeviceType dev) const {
        return binary_kernels_[static_cast<size_t>(op_type)][static_cast<size_t>(dev)]
            .load(std::memory_order_acquire);
    }

    UnaryKernelFunc get_unary_kernel(op op_type, DeviceType dev) const {
        return unary_kernels_[static_cast<size_t>(op_type)][static_cast<size_t>(dev)]
            .load(std::memory_order_acquire);
    }

    UnaryInplaceKernelFunc get_unary_inplace_kernel(op op_type, DeviceType dev) const {
        return unary_inplace_kernels_[static_cast<size_t>(op_type)][static_cast<size_t>(dev)]
            .load(std::memory_order_acquire);
    }

    Tensor (*get_softmax_kernel(DeviceType dev) const)(const Tensor&, int) {
        return softmax_kernels_[static_cast<size_t>(dev)].load(std::memory_order_acquire);
    }

    void replace_binary_kernel(op op_type, DeviceType dev, BinaryKernelFunc new_kernel) {
        binary_kernels_[static_cast<size_t>(op_type)][static_cast<size_t>(dev)]
            .store(new_kernel, std::memory_order_release);
    }

    void replace_unary_kernel(op op_type, DeviceType dev, UnaryKernelFunc new_kernel) {
        unary_kernels_[static_cast<size_t>(op_type)][static_cast<size_t>(dev)]
            .store(new_kernel, std::memory_order_release);
    }

    void replace_unary_inplace_kernel(op op_type, DeviceType dev, UnaryInplaceKernelFunc new_kernel) {
        unary_inplace_kernels_[static_cast<size_t>(op_type)][static_cast<size_t>(dev)]
            .store(new_kernel, std::memory_order_release);
    }

    void replace_softmax_kernel(DeviceType dev, Tensor (*new_kernel)(const Tensor&, int)) {
        softmax_kernels_[static_cast<size_t>(dev)].store(new_kernel, std::memory_order_release);
    }

    // ========== 线性代数专用 kernel 注册 / dispatch（2026-08-10）==========
    // 不走 op enum，因为 Rot/ApplyHk 的参数签名与现有 Unary/Binary 不同
    void register_rot_kernel(DeviceType dev, RotKernelFunc f) {
        rot_kernels_[static_cast<size_t>(dev)].store(f, std::memory_order_release);
    }
    void register_applyhk_kernel(DeviceType dev, ApplyHkKernelFunc f) {
        applyhk_kernels_[static_cast<size_t>(dev)].store(f, std::memory_order_release);
    }
    RotKernelFunc get_rot_kernel(DeviceType dev) const {
        return rot_kernels_[static_cast<size_t>(dev)].load(std::memory_order_acquire);
    }
    ApplyHkKernelFunc get_applyhk_kernel(DeviceType dev) const {
        return applyhk_kernels_[static_cast<size_t>(dev)].load(std::memory_order_acquire);
    }

    /// Givens 旋转 dispatch：原地旋转两向量，自动按 device 选 kernel
    /// 选 kernel 优先级：MPS > AMX > SIMD > CPU
    void dispatch_rot(Tensor& x, Tensor& y, float c, float s);

    /// Householder apply dispatch：原地更新 M 子块
    void dispatch_applyhk(Tensor& M, const Tensor& v, float tau,
                          std::size_t k_offset, std::size_t p_cols);

    Tensor dispatch(const Tensor& a, const Tensor& b, op op_type);
    Tensor dispatch(const Tensor& a, op op_type);
    void dispatch_inplace(Tensor& a, op op_type);
    Tensor dispatch_softmax(const Tensor& a, int dim = -1);

    #ifndef CT_DISABLE_C3
    /// 重置区域融合状态（包含 trace 和预走状态，用于测试场景）
    void resetRegionFusion();
#endif

    template <op OpType>
    inline Tensor dispatch(const Tensor& a, const Tensor& b) {
        if (g_in_recomputation) {
            DeviceType target_dev = getTargetDevice(a, b);
            BinaryKernelFunc func = selectBestBinary(OpType, target_dev, binary_kernels_);
            if (func == nullptr) {
                CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::PLATFORM_API, "Ctorch_Scheduler: 没有可用的Kernel");
            }
            return func(a, b);
        }
#ifndef CT_DISABLE_C3
        // [区域融合] 快速路径：预走模式中跳过 dtype/shape 检查，直接调用 tryRegionDispatch
        if (prewalk_state_ == PrewalkState::kPrewalking && ct::c3::regionFusionEnabled()
                && ct::detail::isRegionCandidateOp(OpType)) {
            std::array<Tensor, 2> region_in = {a, b};
            auto region_result = tryRegionDispatch(OpType, region_in.data(), region_in.size(), getTargetDevice(a, b));
            if (region_result.has_value()) {
                return std::move(region_result.value());
            }
        }
#endif
#ifdef C3_DISPATCH_TIMING
        static int64_t g_dtype_ns = 0, g_shape_ns = 0, g_dev_ns = 0, g_vec_ns = 0, g_region_ns = 0;
        static int64_t g_c3_ns = 0, g_fused_ns = 0, g_hotpath_ns = 0, g_select_ns = 0, g_kernel_ns = 0;
        static int64_t g_trace_ns = 0, g_region_trace_ns = 0;
        static int g_count = 0;
        auto t_total = std::chrono::high_resolution_clock::now();
        auto t0 = t_total;
#endif
        if (a.dtype() != b.dtype()) {
            CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DATATYPE, "Ctorch_Scheduler: Tensor类型不一致");
        }
#ifdef C3_DISPATCH_TIMING
        auto t1 = std::chrono::high_resolution_clock::now();
        g_dtype_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        t0 = t1;
#endif
        if (OpType != op::Add && OpType != op::Mul && OpType != op::Sub && OpType != op::Div && OpType != op::CE && OpType != op::MatMul && a.sizes() != b.sizes()) {
            CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DIMENSION, "Ctorch_Scheduler: Tensor形状不一致");
        }
#ifdef CT_DEBUG
        // 无计数限制的 MatMul 追踪：打印每个 MatMul 的输入形状和 tensor id
        if constexpr (OpType == op::MatMul) {
            fprintf(stderr, "[TRACE-MatMul] a=[");
            for (size_t i = 0; i < a.shape().size(); ++i) { if (i > 0) fprintf(stderr, ","); fprintf(stderr, "%zu", a.shape()[i]); }
            fprintf(stderr, "] b=[");
            for (size_t i = 0; i < b.shape().size(); ++i) { if (i > 0) fprintf(stderr, ","); fprintf(stderr, "%zu", b.shape()[i]); }
            fprintf(stderr, "] tid_a=%zu tid_b=%zu\n", a.id(), b.id());
        }
#endif
#ifdef C3_DISPATCH_TIMING
        t1 = std::chrono::high_resolution_clock::now();
        g_shape_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        t0 = t1;
#endif

        DeviceType target_dev = getTargetDevice(a, b);
#ifdef C3_DISPATCH_TIMING
        t1 = std::chrono::high_resolution_clock::now();
        g_dev_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        t0 = t1;
#endif

        #ifndef CT_DISABLE_C3
        // [区域融合] 预走/匹配检查
        if (ct::c3::regionFusionEnabled() && ct::detail::isRegionCandidateOp(OpType)) {
            std::array<Tensor, 2> region_inputs = {a, b};
#ifdef C3_DISPATCH_TIMING
            t1 = std::chrono::high_resolution_clock::now();
            g_vec_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            t0 = t1;
#endif
            auto region_result = tryRegionDispatch(OpType, region_inputs.data(), region_inputs.size(), target_dev);
#ifdef C3_DISPATCH_TIMING
            t1 = std::chrono::high_resolution_clock::now();
            g_region_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            t0 = t1;
#endif
            if (region_result.has_value()) {
#ifdef C3_DISPATCH_TIMING
                g_count++;
                if (g_count <= 20) {
                    fprintf(stderr, "[DISPATCH TIMING] op=%d dtype=%.0f shape=%.0f dev=%.0f vec=%.0f region=%.0f ns\n",
                            static_cast<int>(OpType),
                            (double)g_dtype_ns / g_count, (double)g_shape_ns / g_count,
                            (double)g_dev_ns / g_count, (double)g_vec_ns / g_count,
                            (double)g_region_ns / g_count);
                }
#endif
                return std::move(region_result.value());
            }
        }
#endif

#ifndef CT_DISABLE_C3
        // C3 JIT 热替换优先查询：若已安装 C3 kernel，优先使用
        // 实验开关 C3_DISABLE_SINGLE_KERNEL=1 时跳过，仅保留区域融合
        // 方案 A：autograd 追踪区域（输入 requires_grad）也跳过单 kernel 注入，
        // 避免 Eager→JIT 混合轨迹破坏训练一致性。
        // DEBT-NEW-7 H2 fix 计数器:c3_attemptable + in_autograd 时记录 bypass
        const bool c3_attemptable = !ct::detail::c3SingleKernelDisabled() &&
                                    !ct::detail::c3OpDisabled(static_cast<int>(OpType));
        const bool in_autograd = ct::detail::inAutogradScope(a.requires_grad(), b.requires_grad());
        // [实验] 直接允许 autograd scope 内执行单 kernel，不再 bypass
        // 之前的"Eager→JIT 混合轨迹破坏训练一致性"假设需实测验证
        if (c3_attemptable) {
            auto c3_result = ct::c3::C3KernelRegistry::getInstance().tryExecute(OpType, a, b);
            if (c3_result.has_value()) {
#ifdef CT_DEBUG
                {
                    std::ostringstream oss;
                    oss << "[C3-Dispatch] binary_c3_hit op=" << static_cast<int>(OpType)
                        << " a=[";
                    for (size_t i = 0; i < a.shape().size(); ++i) {
                        if (i > 0) oss << ",";
                        oss << a.shape()[i];
                    }
                    oss << "] b=[";
                    for (size_t i = 0; i < b.shape().size(); ++i) {
                        if (i > 0) oss << ",";
                        oss << b.shape()[i];
                    }
                    oss << "] out=[";
                    auto& out_s = c3_result.value().shape();
                    for (size_t i = 0; i < out_s.size(); ++i) {
                        if (i > 0) oss << ",";
                        oss << out_s[i];
                    }
                    oss << "]";
                    CtorchError::log(ErrorLevel::DEBUG, ErrorPlatform::kGENERAL,
                                     ErrorType::UNKNOWN, oss.str());
                }
#endif
                return c3_result.value();
            }
        }
#endif

        // [DEPRECATED] trace-based 融合已被区域融合替代，此处跳过

        #ifndef CT_DISABLE_C3
        // 记录热路径（触发 C3 自动编译）
        // [Fix 2026-08-09 用户审查 P0-1]: 之前把 a.shape() + b.shape() 拼接当 lhs_shape
        // 传给 recordCall,导致 add 走 submitCompileAsync else 分支时
        //   info.lhs_shape = 拼接 (e.g. [128,256,256])
        //   info.rhs_shape = 空
        // 但 tryExecute 用 makeKeyFromShapes(op, dev, a, b) 算 hashShapes([a], [b]),
        // 跟 install 时的 key 不一致 → 永远 miss 静默回退 eager (单 kernel 永远不命中)
        // 修法: 直接传 a.shape(), b.shape() 分开, 让 C3HotPathManager 内部
        // 用跟 tryExecute 一致的 hash 算法 (a, b 分开 hash 组合)
        // [修复 2026-08-15] 允许在 autograd 期间记录调用，以便触发前向区域融合编译！
        if (target_dev != DeviceType::kMPS) {
            ct::c3::C3HotPathManager::instance().recordCall(
                OpType, target_dev, a.shape(), b.shape(),
                in_autograd);
        }
#endif

        BinaryKernelFunc func = selectBestBinary(OpType, target_dev, binary_kernels_);
        if (func == nullptr) {
            CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::PLATFORM_API, "Ctorch_Scheduler: 没有可用的Kernel");
        }
#ifdef CT_DEBUG
        {
            static int dbg_cnt = 0;
            if (dbg_cnt < 500) {
                std::ostringstream oss;
                oss << "[C3-Dispatch] binary_eager op=" << static_cast<int>(OpType)
                    << " a=[";
                for (size_t i = 0; i < a.shape().size(); ++i) {
                    if (i > 0) oss << ",";
                    oss << a.shape()[i];
                }
                oss << "] b=[";
                for (size_t i = 0; i < b.shape().size(); ++i) {
                    if (i > 0) oss << ",";
                    oss << b.shape()[i];
                }
                oss << "] dev=" << static_cast<int>(target_dev);
                CtorchError::log(ErrorLevel::DEBUG, ErrorPlatform::kGENERAL, ErrorType::UNKNOWN, oss.str());
            }
            dbg_cnt++;
        }
#endif
        Tensor result;
#ifdef CT_PROFILE_PERF
        {
            auto _t0 = std::chrono::steady_clock::now();
            result = func(a, b);
            auto _t1 = std::chrono::steady_clock::now();
            uint64_t _ns = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(_t1 - _t0).count();
#ifndef CT_DISABLE_C3
            ct::c3::C3KernelRegistry::getInstance().recordPerfEagerInvoke(_ns);
#else
            // CT_DISABLE_C3 build:用文件级 inline static 累加(c3 off 也能量化 eager)
            ct::detail::perfEagerAdd(_ns);
#endif
        }
#else
        result = func(a, b);
#endif

        // [DEPRECATED] 融合 trace 记录已被移除，由区域融合替代

        #ifndef CT_DISABLE_C3
        // 记录到区域融合 trace + 同步更新 prewalk_cache_
        {
            std::lock_guard<std::mutex> lk(region_trace_mutex_);
            region_trace_.push_back(OpType);
            // 限制 trace 最大长度，防止 O(n) 哈希计算和内存膨胀
            if (region_trace_.size() > 64) {
                region_trace_.erase(region_trace_.begin());
            }

            // DEBT-NEW-7:同步更新 prewalk_cache_（region fusion match 用）,
            // 按 buildFusedGraph 约定只缓存每个 dispatch 的 external inputs:
            //   - MatMul: 2 external（区域入口）
            //   - 二元非 MatMul: 1 external（inputs[0] 是 chain）
            //   - 一元: 0 external（input 是 chain）
            // [Dev 2026-08-11] 环形缓冲 push: 固定容量, 无 realloc / erase(begin()) 前移
            PrewalkEntry entry;
            entry.op_type = OpType;
            if constexpr (OpType == op::MatMul) {
                entry.original_inputs = {a, b};
            } else if constexpr (OpType == op::Add || OpType == op::Sub ||
                                 OpType == op::Mul || OpType == op::Div ||
                                 OpType == op::CE) {
                entry.original_inputs = {b};
            } else {
                entry.original_inputs = {};
            }
            prewalkAt(prewalk_cache_count_) = std::move(entry);
            if (prewalk_cache_count_ < kPrewalkCacheCapacity) {
                ++prewalk_cache_count_;
            } else {
                prewalk_cache_head_ = (prewalk_cache_head_ + 1) & kPrewalkCacheMask;
            }
        }
#endif

        return result;
    }

    template <op OpType>
    inline Tensor dispatch(const Tensor& a) {
        if (g_in_recomputation) {
            DeviceType target_dev = a.device();
            UnaryKernelFunc func = selectBestUnary(OpType, target_dev, unary_kernels_);
            if (func == nullptr) {
                CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::PLATFORM_API, "Ctorch_Scheduler: 没有可用的Kernel");
            }
            return func(a);
        }
#ifdef CT_DEBUG
        // 无计数限制的 unary 追踪
        if constexpr (OpType == op::ReLU) {
            fprintf(stderr, "[TRACE-ReLU] in=[");
            for (size_t i = 0; i < a.shape().size(); ++i) { if (i > 0) fprintf(stderr, ","); fprintf(stderr, "%zu", a.shape()[i]); }
            fprintf(stderr, "] tid=%zu\n", a.id());
        }
#endif
#ifndef CT_DISABLE_C3
        // [区域融合] 快速路径：预走模式中跳过检查，直接调用 tryRegionDispatch
        if (prewalk_state_ == PrewalkState::kPrewalking && ct::c3::regionFusionEnabled()
                && ct::detail::isRegionCandidateOp(OpType)) {
            auto region_result = tryRegionDispatch(OpType, &a, 1, a.device());
            if (region_result.has_value()) {
                return std::move(region_result.value());
            }
        }
#endif

        DeviceType target_dev = a.device();

#ifndef CT_DISABLE_C3
        // [区域融合] 预走/匹配检查
        if (ct::c3::regionFusionEnabled() && ct::detail::isRegionCandidateOp(OpType)) {
            std::array<Tensor, 1> region_inputs = {a};
            auto region_result = tryRegionDispatch(OpType, region_inputs.data(), region_inputs.size(), target_dev);
            if (region_result.has_value()) {
                return std::move(region_result.value());
            }
        }
#endif

#ifndef CT_DISABLE_C3
        // C3 JIT 热替换优先查询：若已安装 C3 kernel，优先使用
        // 实验开关 C3_DISABLE_SINGLE_KERNEL=1 时跳过，仅保留区域融合
        // 方案 A：autograd 追踪区域（输入 requires_grad）也跳过单 kernel 注入，
        // 避免 Eager→JIT 混合轨迹破坏训练一致性。
        // DEBT-NEW-7 H2 fix 计数器:unary 版（b=false 因 unary 无第二输入）
        const bool c3_attemptable_u = !ct::detail::c3SingleKernelDisabled() &&
                                      !ct::detail::c3OpDisabled(static_cast<int>(OpType));
        const bool in_autograd_u = ct::detail::inAutogradScope(a.requires_grad(), false);
        // [实验] 直接允许 autograd scope 内执行单 kernel，不再 bypass
        if (c3_attemptable_u) {
            auto c3_result = ct::c3::C3KernelRegistry::getInstance().tryExecuteUnary(OpType, a);
            if (c3_result.has_value()) {
#ifdef CT_DEBUG
                {
                    std::ostringstream oss;
                    oss << "[C3-Dispatch] unary_c3_hit op=" << static_cast<int>(OpType)
                        << " in=[";
                    for (size_t i = 0; i < a.shape().size(); ++i) {
                        if (i > 0) oss << ",";
                        oss << a.shape()[i];
                    }
                    oss << "] out=[";
                    auto& out_s = c3_result.value().shape();
                    for (size_t i = 0; i < out_s.size(); ++i) {
                        if (i > 0) oss << ",";
                        oss << out_s[i];
                    }
                    oss << "]";
                    CtorchError::log(ErrorLevel::DEBUG, ErrorPlatform::kGENERAL,
                                     ErrorType::UNKNOWN, oss.str());
                }
#endif
                return c3_result.value();
            }
        }
#endif

        // [DEPRECATED] trace-based 融合已被区域融合替代，此处跳过

#ifndef CT_DISABLE_C3
        // 记录热路径：修复允许在 autograd 期间记录，以便进行前向区域融合
        if (target_dev != DeviceType::kMPS) {
            ct::c3::C3HotPathManager::instance().recordCall(OpType, target_dev, a.shape(), {},
                in_autograd_u);
        }
#endif

        UnaryKernelFunc func = selectBestUnary(OpType, target_dev, unary_kernels_);
        if (func == nullptr) {
            CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::PLATFORM_API, "Ctorch_Scheduler: 没有可用的Kernel");
        }
#ifdef CT_DEBUG
        {
            static int dbg_unary_cnt = 0;
            if (dbg_unary_cnt < 500) {
                std::ostringstream oss;
                oss << "[C3-Dispatch] unary_eager op=" << static_cast<int>(OpType)
                    << " in=[";
                for (size_t i = 0; i < a.shape().size(); ++i) {
                    if (i > 0) oss << ",";
                    oss << a.shape()[i];
                }
                oss << "] dev=" << static_cast<int>(target_dev);
                CtorchError::log(ErrorLevel::DEBUG, ErrorPlatform::kGENERAL,
                                 ErrorType::UNKNOWN, oss.str());
            }
            dbg_unary_cnt++;
        }
#endif
        Tensor result = func(a);

        // [DEPRECATED] 融合 trace 记录已被移除，由区域融合替代

        #ifndef CT_DISABLE_C3
        // 记录到区域融合 trace + 同步更新 prewalk_cache_
        {
            std::lock_guard<std::mutex> lk(region_trace_mutex_);
            region_trace_.push_back(OpType);
            if (region_trace_.size() > 64) {
                region_trace_.erase(region_trace_.begin());
            }

            // DEBT-NEW-7:同步更新 prewalk_cache_。一元 op 的 input 是 chain,无 external
            // [Dev 2026-08-11] 环形缓冲 push
            PrewalkEntry entry;
            entry.op_type = OpType;
            entry.original_inputs = {};  // 一元 op:无 external input
            prewalkAt(prewalk_cache_count_) = std::move(entry);
            if (prewalk_cache_count_ < kPrewalkCacheCapacity) {
                ++prewalk_cache_count_;
            } else {
                prewalk_cache_head_ = (prewalk_cache_head_ + 1) & kPrewalkCacheMask;
            }
        }
#endif

        return result;
    }
};

#endif //CTORCH_SCHEDULER_H
