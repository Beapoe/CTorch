/**
 * @file CtorchScheduler.cpp
 * @brief Ctorch 调度器实现
 * @author GhostFace
 * @date 2026/04/04
 */

#include "./../include/CtorchScheduler.h"
#include "./../include/AutoGrad.h"
#include "./../include/DeviceAllocator.h"

#ifdef __APPLE__
// 前向声明 Metal 设备创建函数，避免在 .cpp 中直接 #import <Metal/Metal.h>
// 该函数返回 id<MTLDevice>，在 C++ 侧以 void* 比较是否非空即可。
extern "C" void* MTLCreateSystemDefaultDevice(void);
#else
#include <cpuid.h>
#endif

thread_local bool g_in_recomputation = false;

bool CtorchScheduler::isDeviceAvailable(DeviceType dev_type) {
    // [Dev] v0.5.2+ (2026-08-09): static cache 让 system call (cpuid / MTLCreateSystemDefaultDevice)
    // 只在第一次调用时执行,后续 O(1) 查表。原版每次 dispatch 都调系统调用 0.5-1us,
    // 53848 dispatch/epoch × 1us = 54ms/epoch 净亏。
    // 设备可用性是 startup-time 不变量 (MPS 设备不会运行时插拔,AMX_TILE CPUID 不会运行时变),
    // 缓存安全。
    static const bool kCPU_Available = true;
    static const bool kCUDA_Available = false;
#ifdef __APPLE__
    static const bool kMPS_Available = MTLCreateSystemDefaultDevice() != nullptr;
    static const bool kAMX_Available = true;  // Apple Silicon: AMX 走 Accelerate 框架
#else
    static const bool kMPS_Available = false;
    static const bool kAMX_Available = []() {
        unsigned int eax, ebx, ecx, edx;
        if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
            return (edx & (1u << 24)) != 0;
        }
        return false;
    }();
#endif
    switch (dev_type) {
        case DeviceType::kCPU:  return kCPU_Available;
        case DeviceType::kCUDA: return kCUDA_Available;
        case DeviceType::kMPS:  return kMPS_Available;
        case DeviceType::kAMX:  return kAMX_Available;
        case DeviceType::kSIMD:
            // SIMD 路径依赖编译器 auto-vectorization，默认在 CPU 上可用。
            return true;
        default:
            return false;
    }
}

CtorchScheduler::CtorchScheduler() {
    printf(ESC_START COLOR_INFO"[INFO]  " ESC_END "[%s %" PRIu64 "] Ctorch Scheduler Started\n", getFormattedTimeMs().c_str(), getTimestampMs());
    
    // Allocator 现在由 AllocatorManager::getAllocator() 延迟初始化，
    // 避免 Storage 构造早于 CtorchScheduler 单例时无法获取 MPS buffer。
    initKernels();
}

void CtorchScheduler::initKernels() {
    auto set_bin = [this](op o, DeviceType d, BinaryKernelFunc f) {
        binary_kernels_[static_cast<size_t>(o)][static_cast<size_t>(d)]
            .store(f, std::memory_order_relaxed);
    };
    auto set_unary = [this](op o, DeviceType d, UnaryKernelFunc f) {
        unary_kernels_[static_cast<size_t>(o)][static_cast<size_t>(d)]
            .store(f, std::memory_order_relaxed);
    };
    auto set_unary_inplace = [this](op o, DeviceType d, UnaryInplaceKernelFunc f) {
        unary_inplace_kernels_[static_cast<size_t>(o)][static_cast<size_t>(d)]
            .store(f, std::memory_order_relaxed);
    };
    auto set_softmax = [this](DeviceType d, Tensor (*f)(const Tensor&, int)) {
        softmax_kernels_[static_cast<size_t>(d)].store(f, std::memory_order_relaxed);
    };

    // CPU kernels (BASIC 作为 fallback)
    set_bin(op::Add, DeviceType::kCPU, Add_BASIC_kernel);
    set_bin(op::Sub, DeviceType::kCPU, Sub_BASIC_kernel);
    set_bin(op::Mul, DeviceType::kCPU, Mul_BASIC_kernel);
    set_bin(op::Div, DeviceType::kCPU, Div_BASIC_kernel);
    set_bin(op::MatMul, DeviceType::kCPU, MatMul_BASIC_kernel);
    set_bin(op::Dot, DeviceType::kCPU, Dot_BASIC_kernel);
    set_bin(op::MSE, DeviceType::kCPU, MSE_BASIC_kernel);
    set_bin(op::CE, DeviceType::kCPU, CrossEntropy_BASIC_kernel);
    set_bin(op::MAE, DeviceType::kCPU, MAE_BASIC_kernel);

    set_unary(op::Neg, DeviceType::kCPU, Neg_BASIC_kernel);
    set_unary(op::ReLU, DeviceType::kCPU, ReLU_BASIC_kernel);
    set_unary(op::Cos, DeviceType::kCPU, Cos_BASIC_kernel);
    set_unary(op::Sin, DeviceType::kCPU, Sin_BASIC_kernel);
    set_unary(op::Tanh, DeviceType::kCPU, Tanh_BASIC_kernel);
    set_unary(op::Sigmoid, DeviceType::kCPU, Sigmoid_BASIC_kernel);
    set_unary(op::GELU, DeviceType::kCPU, GELU_BASIC_kernel);
    set_unary(op::SiLU, DeviceType::kCPU, SiLU_BASIC_kernel);  // PEL25 Stage 3.4
    set_bin(op::SwiGLU, DeviceType::kCPU, SwiGLU_BASIC_kernel);  // PEL25 Stage 3.4 (双输入)
    set_unary(op::LReLU, DeviceType::kCPU, LReLU_BASIC_kernel);
    set_unary(op::Log, DeviceType::kCPU, Log_BASIC_kernel);
    set_unary(op::Exp, DeviceType::kCPU, Exp_BASIC_kernel);
    set_unary(op::Abs, DeviceType::kCPU, Abs_BASIC_kernel);

    // CPU in-place unary kernels (BASIC fallback)
    set_unary_inplace(op::Neg, DeviceType::kCPU, Neg_BASIC_inplace);
    set_unary_inplace(op::Cos, DeviceType::kCPU, Cos_BASIC_inplace);
    set_unary_inplace(op::Sin, DeviceType::kCPU, Sin_BASIC_inplace);
    set_unary_inplace(op::ReLU, DeviceType::kCPU, ReLU_BASIC_inplace);
    set_unary_inplace(op::Tanh, DeviceType::kCPU, Tanh_BASIC_inplace);
    set_unary_inplace(op::Sigmoid, DeviceType::kCPU, Sigmoid_BASIC_inplace);
    set_unary_inplace(op::GELU, DeviceType::kCPU, GELU_BASIC_inplace);
    set_unary_inplace(op::SiLU, DeviceType::kCPU, SiLU_BASIC_inplace);  // PEL25 Stage 3.4
    set_unary_inplace(op::LReLU, DeviceType::kCPU, LReLU_BASIC_inplace);
    set_unary_inplace(op::Log, DeviceType::kCPU, Log_BASIC_inplace);
    set_unary_inplace(op::Exp, DeviceType::kCPU, Exp_BASIC_inplace);
    set_unary_inplace(op::Abs, DeviceType::kCPU, Abs_BASIC_inplace);

    set_bin(op::Min, DeviceType::kCPU, Min_BASIC_kernel);
    set_bin(op::Max, DeviceType::kCPU, Max_BASIC_kernel);

    set_softmax(DeviceType::kCPU, Softmax_BASIC_kernel);

    // SIMD kernels (优先于 BASIC)
    set_bin(op::Add, DeviceType::kSIMD, Add_SIMD_kernel);
    set_bin(op::Sub, DeviceType::kSIMD, Sub_SIMD_kernel);
    set_bin(op::Mul, DeviceType::kSIMD, Mul_SIMD_kernel);
    set_bin(op::Div, DeviceType::kSIMD, Div_SIMD_kernel);

    set_unary(op::Neg, DeviceType::kSIMD, Neg_SIMD_kernel);
    set_unary(op::ReLU, DeviceType::kSIMD, ReLU_SIMD_kernel);
    set_unary(op::Tanh, DeviceType::kSIMD, Tanh_SIMD_kernel);
    set_unary(op::Sigmoid, DeviceType::kSIMD, Sigmoid_SIMD_kernel);
    set_unary(op::GELU, DeviceType::kSIMD, GELU_SIMD_kernel);
    set_unary(op::SiLU, DeviceType::kSIMD, SiLU_SIMD_kernel);  // PEL25 Stage 3.4
    set_bin(op::SwiGLU, DeviceType::kSIMD, SwiGLU_SIMD_kernel);  // PEL25 Stage 3.4 (双输入)
    set_unary(op::Log, DeviceType::kSIMD, Log_SIMD_kernel);
    set_unary(op::Exp, DeviceType::kSIMD, Exp_SIMD_kernel);
    set_unary(op::Abs, DeviceType::kSIMD, Abs_SIMD_kernel);

    set_bin(op::Min, DeviceType::kSIMD, Min_SIMD_kernel);
    set_bin(op::Max, DeviceType::kSIMD, Max_SIMD_kernel);
    set_bin(op::MatMul, DeviceType::kSIMD, MatMul_SIMD_kernel);
    set_bin(op::Dot, DeviceType::kSIMD, Dot_SIMD_kernel);
    set_bin(op::MSE, DeviceType::kSIMD, MSE_SIMD_kernel);
    set_bin(op::CE, DeviceType::kSIMD, CrossEntropy_SIMD_kernel);
    set_bin(op::MAE, DeviceType::kSIMD, MAE_SIMD_kernel);

    set_unary(op::Sin, DeviceType::kSIMD, Sin_SIMD_kernel);
    set_unary(op::Cos, DeviceType::kSIMD, Cos_SIMD_kernel);
    set_unary(op::LReLU, DeviceType::kSIMD, LReLU_SIMD_kernel);

    set_softmax(DeviceType::kSIMD, Softmax_SIMD_kernel);

    // AMX kernels（目前只有 MatMul 有 AMX 实现）
    // [Fix] v0.5.2 Linux build: AMX 是 macOS 专属 (Accelerate framework), Linux 跳过
    //   DeviceType::kAMX 在 Linux 上不存在, kernel 也不编, 不 register 即可
#ifdef __APPLE__
    set_bin(op::MatMul, DeviceType::kAMX, MatMul_AMX_kernel);
    set_unary(op::GELU, DeviceType::kAMX, GELU_AMX_kernel);
    set_unary(op::SiLU, DeviceType::kAMX, SiLU_AMX_kernel);  // PEL25 Stage 3.4 (AMX 降级)
#endif

    // MPS kernels
    set_bin(op::Add, DeviceType::kMPS, Add_MPS_kernel);
    set_bin(op::Sub, DeviceType::kMPS, Sub_MPS_kernel);
    set_bin(op::Mul, DeviceType::kMPS, Mul_MPS_kernel);
    set_bin(op::Div, DeviceType::kMPS, Div_MPS_kernel);
    set_bin(op::MatMul, DeviceType::kMPS, MatMul_MPS_kernel);
    set_bin(op::Dot, DeviceType::kMPS, Dot_MPS_kernel);
    set_bin(op::MSE, DeviceType::kMPS, MSE_MPS_kernel);
    set_bin(op::CE, DeviceType::kMPS, CrossEntropy_MPS_kernel);
    set_bin(op::MAE, DeviceType::kMPS, MAE_MPS_kernel);
    set_bin(op::Min, DeviceType::kMPS, Min_MPS_kernel);
    set_bin(op::Max, DeviceType::kMPS, Max_MPS_kernel);

    set_unary(op::Neg, DeviceType::kMPS, Neg_MPS_kernel);
    set_unary(op::ReLU, DeviceType::kMPS, ReLU_MPS_kernel);
    set_unary(op::Cos, DeviceType::kMPS, Cos_MPS_kernel);
    set_unary(op::Sin, DeviceType::kMPS, Sin_MPS_kernel);
    set_unary(op::Tanh, DeviceType::kMPS, Tanh_MPS_kernel);
    set_unary(op::Sigmoid, DeviceType::kMPS, Sigmoid_MPS_kernel);
    set_unary(op::GELU, DeviceType::kMPS, GELU_MPS_kernel);
    set_unary(op::LReLU, DeviceType::kMPS, LReLU_MPS_kernel);
    set_unary(op::Log, DeviceType::kMPS, Log_MPS_kernel);
    set_unary(op::Exp, DeviceType::kMPS, Exp_MPS_kernel);
    set_unary(op::Abs, DeviceType::kMPS, Abs_MPS_kernel);

    // MPS in-place unary kernels (P1-3)
    set_unary_inplace(op::Neg, DeviceType::kMPS, Neg_MPS_inplace);
    set_unary_inplace(op::Cos, DeviceType::kMPS, Cos_MPS_inplace);
    set_unary_inplace(op::Sin, DeviceType::kMPS, Sin_MPS_inplace);
    set_unary_inplace(op::ReLU, DeviceType::kMPS, ReLU_MPS_inplace);
    set_unary_inplace(op::Tanh, DeviceType::kMPS, Tanh_MPS_inplace);
    set_unary_inplace(op::Sigmoid, DeviceType::kMPS, Sigmoid_MPS_inplace);
    set_unary_inplace(op::GELU, DeviceType::kMPS, GELU_MPS_inplace);
    set_unary_inplace(op::LReLU, DeviceType::kMPS, LReLU_MPS_inplace);
    set_unary_inplace(op::Log, DeviceType::kMPS, Log_MPS_inplace);
    set_unary_inplace(op::Exp, DeviceType::kMPS, Exp_MPS_inplace);
    set_unary_inplace(op::Abs, DeviceType::kMPS, Abs_MPS_inplace);

    set_softmax(DeviceType::kMPS, Softmax_MPS_kernel);

    // ========== 线性代数专用 kernel 注册（2026-08-10）==========
    // Rot (Givens 旋转) — 用于 JacobiSVD 的列对正交化
    // ApplyHk (Householder 反射应用) — 用于 HouseholderQR 的反射器 apply
    // 优先 AMX 路径（Apple Accelerate cblas_srot / cblas_sgemv + cblas_sger）
    // 其它设备暂未实现（cblas_srot/cblas_sger 在非 Apple 平台也能用 cblas 接口，
    // 后续如果上 CUDA/CPU-x86 可在对应 backend 加 SIMD/BASIC 回退）
#ifdef __APPLE__
    register_rot_kernel(DeviceType::kAMX, Rot_AMX_kernel);
#endif
#ifdef __APPLE__
    register_rot_kernel(DeviceType::kMPS, Rot_AMX_kernel);  // MPS 内存是 shared，AMX kernel 可用 (但 MPS 也 macOS only)
    register_rot_kernel(DeviceType::kCPU, Rot_AMX_kernel);  // CPU 也走 AMX（Accelerate 框架）
    register_rot_kernel(DeviceType::kSIMD, Rot_AMX_kernel);
#endif

#ifdef __APPLE__
    register_applyhk_kernel(DeviceType::kAMX, ApplyHk_AMX_kernel);
#endif
#ifdef __APPLE__
    register_applyhk_kernel(DeviceType::kMPS, ApplyHk_AMX_kernel);
    register_applyhk_kernel(DeviceType::kCPU, ApplyHk_AMX_kernel);
    register_applyhk_kernel(DeviceType::kSIMD, ApplyHk_AMX_kernel);
#endif
}

// ============================================================
// 线性代数专用 dispatch（不走 op enum，2026-08-10）
// ============================================================

void CtorchScheduler::dispatch_rot(Tensor& x, Tensor& y, float c, float s) {
    // 按 device 选 kernel：MPS > AMX > SIMD > CPU > CPU_BASIC
    DeviceType dev = x.device();
    RotKernelFunc f = get_rot_kernel(dev);
    if (f == nullptr) {
        // 回退：MPS → AMX → SIMD → CPU
        if (isDeviceAvailable(DeviceType::kAMX)) {
            f = get_rot_kernel(DeviceType::kAMX);
        }
        if (f == nullptr) {
            f = get_rot_kernel(DeviceType::kSIMD);
        }
        if (f == nullptr) {
            f = get_rot_kernel(DeviceType::kCPU);
        }
    }
    if (f == nullptr) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DEVICE_COMPAT,
                                     "dispatch_rot: 没有可用的 Rot kernel");
    }
    f(x, y, c, s);
}

void CtorchScheduler::dispatch_applyhk(Tensor& M, const Tensor& v, float tau,
                                        std::size_t k_offset, std::size_t p_cols) {
    DeviceType dev = M.device();
    ApplyHkKernelFunc f = get_applyhk_kernel(dev);
    if (f == nullptr) {
        if (isDeviceAvailable(DeviceType::kAMX)) {
            f = get_applyhk_kernel(DeviceType::kAMX);
        }
        if (f == nullptr) {
            f = get_applyhk_kernel(DeviceType::kSIMD);
        }
        if (f == nullptr) {
            f = get_applyhk_kernel(DeviceType::kCPU);
        }
    }
    if (f == nullptr) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DEVICE_COMPAT,
                                     "dispatch_applyhk: 没有可用的 ApplyHk kernel");
    }
    f(M, v, tau, k_offset, p_cols);
}

BinaryKernelFunc CtorchScheduler::selectBestBinary(
    op op_type, DeviceType dev,
    const std::array<std::array<std::atomic<BinaryKernelFunc>, DEVICE_COUNT>, OP_COUNT>& table)
{
    size_t op_idx = static_cast<size_t>(op_type);

    // MPS 张量必须走 MPS kernel
    if (dev == DeviceType::kMPS) {
        BinaryKernelFunc func = table[op_idx][static_cast<size_t>(DeviceType::kMPS)]
            .load(std::memory_order_acquire);
        if (func != nullptr && isDeviceAvailable(DeviceType::kMPS)) {
            return func;
        }
        // [Fix 2026-08-13] MPS kernel 不可用时 fallback 到 CPU
        CtorchError::log(ErrorLevel::WARN, ErrorPlatform::kGENERAL, 
                         ErrorType::PLATFORM_API,
                         "selectBestBinary: no MPS kernel for op=" + std::to_string(static_cast<int>(op_type)) 
                         + ", fallback to CPU");
        return table[op_idx][static_cast<size_t>(DeviceType::kCPU)].load(std::memory_order_acquire);
    }

    // CPU/AMX/SIMD 张量：优先 AMX -> 直接到BASIC（AMX张量不能用SIMD kernel）
    // [Fix 2026-08-13] AMX 仅支持部分 op（如 MatMul），其他 op 需 fallback 到 CPU BASIC
    // 注意：AMX 张量不能调用 CPU-SIMD kernel（SIMD kernel 会拒绝非CPU设备），所以必须直接到BASIC
    if (isDeviceAvailable(DeviceType::kAMX) && dev == DeviceType::kAMX) {
        BinaryKernelFunc func = table[op_idx][static_cast<size_t>(DeviceType::kAMX)]
            .load(std::memory_order_acquire);
        if (func != nullptr) return func;
        // AMX kernel 不可用，优先 fallback 到 CPU SIMD（SIMD 支持 N-D 广播且速度更快）
        BinaryKernelFunc func_simd = table[op_idx][static_cast<size_t>(DeviceType::kSIMD)]
            .load(std::memory_order_acquire);
        if (func_simd != nullptr) return func_simd;
        // 否则 fallback 到 CPU BASIC
        return table[op_idx][static_cast<size_t>(DeviceType::kCPU)].load(std::memory_order_acquire);
    }

    BinaryKernelFunc func = table[op_idx][static_cast<size_t>(DeviceType::kSIMD)]
        .load(std::memory_order_acquire);
    if (func != nullptr) return func;

    func = table[op_idx][static_cast<size_t>(DeviceType::kCPU)].load(std::memory_order_acquire);
    return func;
}

UnaryKernelFunc CtorchScheduler::selectBestUnary(
    op op_type, DeviceType dev,
    const std::array<std::array<std::atomic<UnaryKernelFunc>, DEVICE_COUNT>, OP_COUNT>& table)
{
    size_t op_idx = static_cast<size_t>(op_type);

    // MPS 张量必须走 MPS kernel
    if (dev == DeviceType::kMPS) {
        UnaryKernelFunc func = table[op_idx][static_cast<size_t>(DeviceType::kMPS)]
            .load(std::memory_order_acquire);
        if (func != nullptr && isDeviceAvailable(DeviceType::kMPS)) {
            return func;
        }
        // [Fix 2026-08-13] MPS kernel 不可用时 fallback 到 CPU
        CtorchError::log(ErrorLevel::WARN, ErrorPlatform::kGENERAL, 
                         ErrorType::PLATFORM_API,
                         "selectBestUnary: no MPS kernel for op=" + std::to_string(static_cast<int>(op_type)) 
                         + ", fallback to CPU");
        return table[op_idx][static_cast<size_t>(DeviceType::kCPU)].load(std::memory_order_acquire);
    }

    // CPU/AMX/SIMD 张量：优先 AMX -> 直接到BASIC（AMX张量不能用SIMD kernel）
    // [Fix 2026-08-13] AMX 仅支持部分 op（如 GELU），其他 op 需 fallback 到 CPU BASIC
    // 注意：AMX 张量不能调用 CPU-SIMD kernel（SIMD kernel 会拒绝非CPU设备），所以必须直接到BASIC
    if (isDeviceAvailable(DeviceType::kAMX) && dev == DeviceType::kAMX) {
        UnaryKernelFunc func = table[op_idx][static_cast<size_t>(DeviceType::kAMX)]
            .load(std::memory_order_acquire);
        if (func != nullptr) return func;
        // AMX kernel 不可用，优先 fallback 到 CPU SIMD（速度更快）
        UnaryKernelFunc func_simd = table[op_idx][static_cast<size_t>(DeviceType::kSIMD)]
            .load(std::memory_order_acquire);
        if (func_simd != nullptr) return func_simd;
        // 否则 fallback 到 CPU BASIC
        return table[op_idx][static_cast<size_t>(DeviceType::kCPU)].load(std::memory_order_acquire);
    }

    UnaryKernelFunc func = table[op_idx][static_cast<size_t>(DeviceType::kSIMD)]
        .load(std::memory_order_acquire);
    if (func != nullptr) return func;

    func = table[op_idx][static_cast<size_t>(DeviceType::kCPU)].load(std::memory_order_acquire);
    return func;
}

UnaryInplaceKernelFunc CtorchScheduler::selectBestUnaryInplace(
    op op_type, DeviceType dev,
    const std::array<std::array<std::atomic<UnaryInplaceKernelFunc>, DEVICE_COUNT>, OP_COUNT>& table)
{
    size_t op_idx = static_cast<size_t>(op_type);

    // MPS 张量必须走 MPS kernel
    if (dev == DeviceType::kMPS) {
        UnaryInplaceKernelFunc func = table[op_idx][static_cast<size_t>(DeviceType::kMPS)]
            .load(std::memory_order_acquire);
        if (func != nullptr && isDeviceAvailable(DeviceType::kMPS)) {
            return func;
        }
        return nullptr;
    }

    // CPU/AMX/SIMD 张量：优先 AMX -> SIMD -> BASIC；当前仅 BASIC 实现 in-place，
    // 因此实际会回退到 CPU BASIC。未来可按算子补充 AMX/SIMD in-place 实现。
    if (isDeviceAvailable(DeviceType::kAMX)) {
        UnaryInplaceKernelFunc func = table[op_idx][static_cast<size_t>(DeviceType::kAMX)]
            .load(std::memory_order_acquire);
        if (func != nullptr) return func;
    }

    UnaryInplaceKernelFunc func = table[op_idx][static_cast<size_t>(DeviceType::kSIMD)]
        .load(std::memory_order_acquire);
    if (func != nullptr) return func;

    func = table[op_idx][static_cast<size_t>(DeviceType::kCPU)].load(std::memory_order_acquire);
    return func;
}

Tensor CtorchScheduler::dispatch(const Tensor& a, const Tensor& b, op op_type) {
    if (a.dtype() != b.dtype()) {
        CtorchError::throwException(ErrorPlatform::kGENERAL,ErrorType::DATATYPE,"Ctorch_Scheduler: Tensor类型不一致");
    }
    if (op_type != op::Add && op_type != op::Mul && op_type != op::Sub && op_type != op::Div && op_type != op::CE && op_type != op::MatMul && a.sizes() != b.sizes()) {
        CtorchError::throwException(ErrorPlatform::kGENERAL,ErrorType::DIMENSION,"Ctorch_Scheduler: Tensor形状不一致");
    }

    DeviceType target_dev = getTargetDevice(a, b);
    BinaryKernelFunc target_kernel = selectBestBinary(op_type, target_dev, binary_kernels_);

    if (target_kernel == nullptr) {
        CtorchError::throwException(ErrorPlatform::kGENERAL,ErrorType::PLATFORM_API,"Ctorch_Scheduler: 没有可用的Kernel");
    }
    return target_kernel(a, b);
}

Tensor CtorchScheduler::dispatch(const Tensor& a, op op_type) {
    DeviceType target_dev = a.device();
    UnaryKernelFunc target_kernel = selectBestUnary(op_type, target_dev, unary_kernels_);

    if (target_kernel == nullptr) {
        CtorchError::throwException(ErrorPlatform::kGENERAL,ErrorType::PLATFORM_API,"Ctorch_Scheduler: 没有可用的Kernel");
    }
    return target_kernel(a);
}

void CtorchScheduler::dispatch_inplace(Tensor& a, op op_type) {
    if (!supports_unary_memory_overlap(a.device(), op_type)) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::PLATFORM_API,
            "Ctorch_Scheduler: 该算子/设备不支持 in-place");
    }

    DeviceType target_dev = a.device();
    UnaryInplaceKernelFunc target_kernel = selectBestUnaryInplace(op_type, target_dev, unary_inplace_kernels_);

    if (target_kernel == nullptr) {
        CtorchError::throwException(ErrorPlatform::kGENERAL,ErrorType::PLATFORM_API,"Ctorch_Scheduler: 没有可用的 in-place Kernel");
    }
    target_kernel(a);
}

Tensor CtorchScheduler::dispatch_softmax(const Tensor& a, int dim) {
    DeviceType target_dev = a.device();

    if (target_dev == DeviceType::kMPS) {
        auto mps_kernel = softmax_kernels_[static_cast<size_t>(DeviceType::kMPS)]
            .load(std::memory_order_acquire);
        if (mps_kernel != nullptr && isDeviceAvailable(DeviceType::kMPS)) {
            return mps_kernel(a, dim);
        }
        CtorchError::throwException(ErrorPlatform::kGENERAL,ErrorType::PLATFORM_API,"Ctorch_Scheduler: 没有可用的MPS Softmax Kernel");
    }

    // CPU 路径：AMX -> SIMD -> BASIC
    if (isDeviceAvailable(DeviceType::kAMX)) {
        auto amx_kernel = softmax_kernels_[static_cast<size_t>(DeviceType::kAMX)]
            .load(std::memory_order_acquire);
        if (amx_kernel != nullptr) {
            return amx_kernel(a, dim);
        }
    }

    auto simd_kernel = softmax_kernels_[static_cast<size_t>(DeviceType::kSIMD)]
        .load(std::memory_order_acquire);
    if (simd_kernel != nullptr) {
        return simd_kernel(a, dim);
    }

    auto cpu_kernel = softmax_kernels_[static_cast<size_t>(DeviceType::kCPU)]
        .load(std::memory_order_acquire);
    if (cpu_kernel != nullptr) {
        return cpu_kernel(a, dim);
    }

    CtorchError::throwException(ErrorPlatform::kGENERAL,ErrorType::PLATFORM_API,"Ctorch_Scheduler: 没有可用的Softmax Kernel");
    return Tensor();
}

// ============================================================================
// Region Fusion 预走接口 tryRegionDispatch()
// ============================================================================

#ifndef CT_DISABLE_C3
// ---------------------------------------------------------------------------
// [Prewalk 归因] C3_PREWALK=0 停用 prewalk 首 op 启动路径（保留原“末尾 op 匹配”
// 路径），用于归因 prewalk 相对旧路径的净收益。C3_PREWALK_DIAG=1 打开物化采样。
static bool prewalkStartEnabled() {
    static const bool v = ([] {
        const char* e = std::getenv("C3_PREWALK");
        return !(e && std::string(e) == "0");
    })();
    return v;
}
static bool prewalkDiagEnabled() {
    static const bool v = ([] {
        const char* e = std::getenv("C3_PREWALK_DIAG");
        return e && std::string(e) == "1";
    })();
    return v;
}
// ============================================================================
// 惰性物化的 eager 重算实现
// ============================================================================
// 物化某个中间 op 的占位符：用 eager kernel 重算该 op 的真实值。
// 供 LazyMaterializer 闭包调用（Eager 重算的值仅供 backward 读 forward 中间值用，
// 唯一触发点是 placeholder 的 data_read()）。重算期间置位 g_in_recomputation，避免再次进入 C3
// region fusion / 热路径注入，形成递归死循环。
Tensor CtorchScheduler::eagerMaterializeOp(op op_type,
                                           const std::vector<Tensor>& inputs,
                                           DeviceType /*dev*/) {
    if (prewalkDiagEnabled()) {
        static std::atomic<size_t> cnt{0};
        if ((++cnt) % 1000 == 1)
            fprintf(stderr, "[PREWALK-DIAG] materialize op=%d inputs=%zu\n",
                    (int)op_type, inputs.size());
    }
    struct RecomputeGuard {
        RecomputeGuard() { g_in_recomputation = true; }
        ~RecomputeGuard() { g_in_recomputation = false; }
    } guard;

    // 二元算子（含 MatMul/Add/Sub/Mul）：走 eager 二元 kernel
    if (inputs.size() >= 2) {
        return dispatch(inputs[0], inputs[1], op_type);
    }
    // 一元算子（ReLU/Sigmoid/Tanh）：走 eager 一元 kernel
    if (inputs.size() == 1) {
        return dispatch(inputs[0], op_type);
    }
    // 输入不完整：返回空张量（data_read 会判定物化失败 => nullptr）
    return Tensor();
}
#endif // CT_DISABLE_C3
// Prewalk 状态机（三态）：
//   kIdle → 检查当前 op 是否是某 region 的首个 op → 设 kPrewalking + 返回 placeholder
//   kPrewalking → 中间 op: 验证 op_seq 匹配 + 缓存 external inputs + 返回 placeholder
//                → 末尾 op: 执行融合 kernel, 恢复 kIdle, 返回真实结果
//                → 不匹配: 设 kFallback, 返回 nullopt (dispatch 走 eager)
//   kFallback → 重置 kIdle, 返回 nullopt
//
// 同时保留原有的"末尾 op 匹配"路径作为 kIdle 时的 fallback（向后兼容）。
#ifndef CT_DISABLE_C3
std::optional<Tensor> CtorchScheduler::tryRegionDispatch(
    op op_type, const Tensor* inputs, size_t num_inputs, DeviceType /*dev*/) {
#ifdef CT_PROFILE_PERF
    auto _t0 = std::chrono::steady_clock::now();
    struct PerfGuard {
        std::chrono::steady_clock::time_point t0;
        PerfGuard(std::chrono::steady_clock::time_point t) : t0(t) {}
        ~PerfGuard() {
            auto t1 = std::chrono::steady_clock::now();
            ct::c3::C3KernelRegistry::getInstance().recordPerfRegionDispatch(
                (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
        }
    } _guard(_t0);
#endif
    // [RD-SEG 2026-08-27] env C3_RD_SEG=1：量化 tryRegionDispatch 各段耗时，定位 forward ~10µs 大头
    struct RdSeg {
        enum { EARLY=0, PREWALK_START=1, PREWALK_MID=2, PREWALK_END=3, TAIL=4, NUM=5 };
        static std::atomic<uint64_t>& N(int i){ static std::atomic<uint64_t> v[NUM]; return v[i]; }
        static std::atomic<uint64_t>& C(int i){ static std::atomic<uint64_t> v[NUM]; return v[i]; }
        static bool on(){ static bool e=[](){auto*p=std::getenv("C3_RD_SEG");return p&&*p=='1';}(); return e; }
    };
    int _rdseg = RdSeg::TAIL;  // 默认末尾 op 匹配路径
    struct RdSegGuard {
        bool on_; std::chrono::steady_clock::time_point t0; int* s;
        RdSegGuard(int* p): on_(RdSeg::on()), t0(), s(p) { if(on_) t0 = std::chrono::steady_clock::now(); }
        ~RdSegGuard(){
            if(!on_ || *s<0 || *s>=RdSeg::NUM) return;
            auto ns=(uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-t0).count();
            RdSeg::N(*s).fetch_add(ns,std::memory_order_relaxed); RdSeg::C(*s).fetch_add(1,std::memory_order_relaxed);
            static thread_local uint64_t acc=0;
            if((++acc)%20000==0){
                const char* nm[5]={"early","start","mid","end","tail"};
                fprintf(stderr,"[RD-SEG]");
                for(int i=0;i<5;++i){
                    uint64_t n=RdSeg::N(i).load(), c=RdSeg::C(i).load();
                    fprintf(stderr," %s=%.1fms/%llu(%.2fµs)", nm[i], n*1e-6,
                        (unsigned long long)c, c? (double)n*1e-3/(double)c : 0.0);
                }
                fprintf(stderr,"\n");
            }
        }
    } _rdseg_guard(&_rdseg);

    // [RD 优化 2026-08-27] backward 期间直接短路：反向传播由 MIMO 融合 kernel 全覆盖（fusion_hit=0），
    // backward 的候选 op（grad_W/grad_X/Add）尝试 forward region 全部落空，纯浪费的调度开销。
    // region 只在 forward 命中，短路安全（acc 验证 97.18% 零回归）。
    if (ct::detail::g_in_backward()) {
        _rdseg = RdSeg::EARLY;
        return std::nullopt;
    }
    auto& registry = ct::c3::RegionFusionRegistry::getInstance();
    if (registry.installedCountNoLock() == 0) {
        _rdseg = RdSeg::EARLY;
        return std::nullopt;
    }

    // ================================================================
    // Prewalk 状态机
    // ================================================================

    // --- kPrewalking: 正在预走 region ---
    if (prewalk_state_ == PrewalkState::kPrewalking && matched_region_) {
        _rdseg = RdSeg::PREWALK_MID;
        const auto& seq = matched_region_->op_seq;
        size_t next_pos = prewalk_pos_ + 1;

        // 检查 op 类型是否匹配 region 的下一个 op
        if (next_pos < seq.size() && seq[next_pos] == op_type) {
            prewalk_pos_ = next_pos;

            // 缓存 external inputs (非 chain 输入)
            // MatMul: 2 个 external (a, b)
            // 二元非 MatMul: 1 个 external (inputs[1], inputs[0] 是 chain)
            // 一元: 0 个 external
            if (op_type == op::MatMul) {
                for (size_t k = 0; k < num_inputs; ++k)
                    prewalk_external_inputs_.push_back(inputs[k].shallow());
            } else if (num_inputs > 1) {
                prewalk_external_inputs_.push_back(inputs[1].shallow());
            }

            // 如果是最后一个 op → 执行融合 kernel
            if (prewalk_pos_ == seq.size() - 1) {
                _rdseg = RdSeg::PREWALK_END;
                // 收集当前 op 的 external inputs
                if (op_type == op::MatMul) {
                    // 已经在上面加了
                } else if (num_inputs > 1) {
                    // 已经在上面加了
                }

                std::vector<size_t> out_shape = computeOutputShape(op_type, inputs, num_inputs);
                if (out_shape.empty()) {
                    prewalk_state_ = PrewalkState::kIdle;
                    matched_region_.reset();
                    prewalk_external_inputs_.clear();
                    return std::nullopt;
                }

                // 预读取 external inputs 验证 data pointer
                for (const auto& t : prewalk_external_inputs_) {
                    if (!t.data_read<float>()) {
                        prewalk_state_ = PrewalkState::kIdle;
                        matched_region_.reset();
                        prewalk_external_inputs_.clear();
                        return std::nullopt;
                    }
                }

                try {
                    ct::c3::KernelShapeInfo shapes;
                    if (!prewalk_external_inputs_.empty())
                        shapes.lhs_shape = prewalk_external_inputs_.front().shape();
                    if (prewalk_external_inputs_.size() > 1)
                        shapes.rhs_shape = prewalk_external_inputs_[1].shape();
                    shapes.out_shape = out_shape;
                    shapes.fused_pattern = "prewalk-fusion";

                    Tensor pre_act;
                    Tensor kernel_result = ct::c3::C3KernelRegistry::getInstance()
                        .executeFusedWithInputs(matched_region_->kernel,
                                                prewalk_external_inputs_, shapes,
                                                &pre_act);
                    // [Prewalk A+] 安全 preload preAct：融合 kernel 的 secondary 输出即为 pre-activation
                    // （Add 输出 = ReLU 输入 = z）。仅当与当前 op(ReLU) 输入占位符同形状且非空时才注入
                    // LazyBox，使 backward 的 data_read(z) 直接复用融合算出的 preAct，消除每次 eager 重算
                    // MatMul/Add（~87% 的 MIMO setup 开销）。形状守卫保证不会把错误段伪装成 z1；
                    // 融合 kernel 的平面输出 storage 由 shared_ptr 引用托管，preload 后仍保活。
                    if (num_inputs >= 1) {
                        const Tensor& relu_in = inputs[0];
                        auto relu_in_lz = relu_in.lazyMaterializer();
                        if (relu_in_lz && !pre_act.storage().empty() &&
                            pre_act.shape() == relu_in.shape() && pre_act.numel() == relu_in.numel()) {
                            relu_in_lz->preload(pre_act);
                            if (prewalkDiagEnabled()) {
                                static std::atomic<size_t> plcnt{0};
                                if ((++plcnt) % 1000 == 1)
                                    fprintf(stderr, "[PREWALK-DIAG] preAct preloaded into ReLU-in placeholder sz=%zu\n",
                                            (size_t)pre_act.numel());
                            }
                        }
                    }
                    // [Prewalk A]（既有诊断，仅记录不注入）
                    if (!pre_act.storage().empty() && prewalkDiagEnabled()) {
                        static std::atomic<size_t> pcnt{0};
                        if ((++pcnt) % 1000 == 1)
                            fprintf(stderr, "[PREWALK-DIAG] secondary preAct available sz=%zu (not preloaded)\n",
                                    (size_t)pre_act.numel());
                    }
                    // 恢复状态
                    prewalk_state_ = PrewalkState::kIdle;
                    matched_region_.reset();
                    prewalk_external_inputs_.clear();

                    if (kernel_result.storage().empty()) {
                        return std::nullopt;
                    }
                    // [forward 诊断] 统计 region prewalk 完成数（按 out_shape），受 C3_FWD_DIAG=1 控制，
                    // 默认关闭避免高频加锁/插入的性能开销。
                    static const bool fwd_pw_diag = [] {
                        const char* e = std::getenv("C3_FWD_DIAG");
                        return e && std::string(e) == "1";
                    }();
                    if (fwd_pw_diag) {
                        static std::mutex mu;
                        static std::unordered_map<std::string, uint64_t> g_pw_path;
                        static uint64_t g_pw_total = 0;
                        std::string key = std::to_string(out_shape[0]) + "x" +
                                          (out_shape.size() > 1 ? std::to_string(out_shape[1]) : "1");
                        uint64_t tot;
                        {
                            std::lock_guard<std::mutex> lk(mu);
                            g_pw_path[key]++;
                            tot = ++g_pw_total;
                        }
                        if (tot % 200 == 0 || tot <= 4) {
                            std::lock_guard<std::mutex> lk(mu);
                            std::string s;
                            for (auto& kv : g_pw_path) s += " " + kv.first + "=" + std::to_string(kv.second);
                            fprintf(stderr, "[PW-STAT] total=%llu:%s\n",
                                    (unsigned long long)(tot), s.c_str());
                        }
                    }
                    return kernel_result;
                } catch (...) {
                    prewalk_state_ = PrewalkState::kIdle;
                    matched_region_.reset();
                    prewalk_external_inputs_.clear();
                    return std::nullopt;
                }
            }

            // 中间 op: 返回 placeholder
            std::vector<size_t> ph_shape = computeOutputShape(op_type, inputs, num_inputs);
            if (ph_shape.empty()) {
                prewalk_state_ = PrewalkState::kFallback;
                return std::nullopt;
            }
            Tensor placeholder(PlaceholderTag{}, ph_shape, DType::kFloat, inputs[0].device());
            std::vector<Tensor> captured_inputs;
            for (size_t i = 0; i < num_inputs; ++i) {
                captured_inputs.push_back(inputs[i].shallow());
            }
            auto mat_fn = [this, op_type, captured_inputs = std::move(captured_inputs)]() -> Tensor {
                DeviceType dev = captured_inputs.empty()
                    ? DeviceType::kCPU : captured_inputs[0].device();
                return eagerMaterializeOp(op_type, captured_inputs, dev);
            };
            placeholder.setLazyMaterializer(std::make_shared<LazyMaterializer>(mat_fn));
            return placeholder;
        } else {
            // op 不匹配 region 序列 → 回退
            prewalk_state_ = PrewalkState::kFallback;
            matched_region_.reset();
            prewalk_external_inputs_.clear();
            return std::nullopt;
        }
    }

    // --- kFallback: 回退到 eager ---
    if (prewalk_state_ == PrewalkState::kFallback) {
        prewalk_state_ = PrewalkState::kIdle;
        matched_region_.reset();
        prewalk_external_inputs_.clear();
        // 继续走下面的"末尾 op 匹配"路径
    }

    // --- kIdle: 检查是否是 region 首个 op → 启动 prewalk ---
    // 直接开启 prewalk（训练+推理均生效）；C3_PREWALK=0 时退回到末尾 op 匹配路径
    if (prewalk_state_ == PrewalkState::kIdle && prewalkStartEnabled()) {
        if (registry.mayMatchAsFirstOp(op_type)) {
            // [perf 2026-09-05] 指针数组传参：避免热路径每次构造 vector<vector<size_t>> 拷贝堆分配
            std::vector<const std::vector<size_t>*> first_input_shapes;
            if (op_type == op::MatMul && num_inputs >= 2) {
                first_input_shapes.push_back(&inputs[0].shape());
                first_input_shapes.push_back(&inputs[1].shape());
            } else if (num_inputs > 0) {
                first_input_shapes.push_back(&inputs[0].shape());
            }

            auto region = registry.findRegionByFirstOp(op_type, first_input_shapes);
            // [forward 诊断] 打印每个首次出现的首个-op(MatMul) 形状与 region 命中/活跃状态，
            // 用于核对 L1(784→256) 与 L2(256→128) 是否都被注册为融合 region。
            static const bool fwd_mm_diag = []{ auto* e = std::getenv("C3_FWD_DIAG"); return e && std::string(e) == "1"; }();
            if (fwd_mm_diag && op_type == op::MatMul && !first_input_shapes.empty()) {
                static std::mutex mu;
                static std::unordered_map<std::string, bool> seen;
                std::string sk;
                for (auto& shp : first_input_shapes) { const auto& sh = *shp; sk += std::to_string(sh.size()) + ":"; for (auto d : sh) sk += std::to_string(d) + "x"; sk += "|"; }
                bool nfirst = false;
                {
                    std::lock_guard<std::mutex> lk(mu);
                    if (!seen.count(sk)) { seen[sk] = true; nfirst = true; }
                }
                if (nfirst) {
                    std::lock_guard<std::mutex> lk(mu);
                    fprintf(stderr, "[MM-REGION] shapes=%s found=%d active=%d cost=%d\n",
                            sk.c_str(), (region ? 1 : 0),
                            (region ? (region->active ? 1 : 0) : 0),
                            (region ? (region->cost.worthwhile ? 1 : 0) : 0));
                }
            }
            if (region && region->active && region->cost.worthwhile) {
                _rdseg = RdSeg::PREWALK_START;
                // 启动 prewalk!
                matched_region_ = region;
                prewalk_pos_ = 0;
                prewalk_state_ = PrewalkState::kPrewalking;
                prewalk_external_inputs_.clear();

                // 缓存首个 op 的 external inputs
                if (op_type == op::MatMul) {
                    for (size_t k = 0; k < num_inputs; ++k)
                        prewalk_external_inputs_.push_back(inputs[k].shallow());
                } else if (num_inputs > 1) {
                    prewalk_external_inputs_.push_back(inputs[1].shallow());
                }

                // 如果 region 只有 1 个 op（不可能，min=2），直接执行
                // 正常情况：返回 placeholder，等后续 op 到达后执行
                std::vector<size_t> ph_shape = computeOutputShape(op_type, inputs, num_inputs);
                if (ph_shape.empty()) {
                    prewalk_state_ = PrewalkState::kFallback;
                    matched_region_.reset();
                    prewalk_external_inputs_.clear();
                    return std::nullopt;
                }
                Tensor placeholder(PlaceholderTag{}, ph_shape, DType::kFloat, inputs[0].device());
                std::vector<Tensor> captured_inputs;
                for (size_t i = 0; i < num_inputs; ++i) {
                    captured_inputs.push_back(inputs[i].shallow());
                }
                auto mat_fn = [this, op_type, captured_inputs = std::move(captured_inputs)]() -> Tensor {
                    DeviceType dev = captured_inputs.empty()
                        ? DeviceType::kCPU : captured_inputs[0].device();
                    return eagerMaterializeOp(op_type, captured_inputs, dev);
                };
                placeholder.setLazyMaterializer(std::make_shared<LazyMaterializer>(mat_fn));
                return placeholder;
            }
        }
    }

    // ================================================================
    // 原有路径：末尾 op 匹配（向后兼容）
    // 当 prewalk 没启动或 region 首个 op 不匹配时，走此路径
    // ================================================================

    // 第二道: 当前 op 不可能作为任何已注册 region 的末尾 op 时 O(1) 返回
    if (!registry.mayMatchAsLastOp(op_type)) {
        return std::nullopt;
    }

    // 读取 trace 快照
    std::vector<op> trace_snapshot;
    {
        std::lock_guard<std::mutex> lk(region_trace_mutex_);
        trace_snapshot = region_trace_;
    }

    // 构建 extended trace: [current trace] + [current op]
    std::vector<op> extended = trace_snapshot;
    extended.push_back(op_type);
    auto extended_prefix = ct::c3::RollingHash::computePrefixHashes(extended);
    if (extended_prefix.size() < 2) {
        return std::nullopt;
    }

    size_t current_pos = extended.size() - 1;

    // DEBT-NEW-7:跟 installWithCost 一样,把 shape 混入 hash
    uint64_t shape_hash = 0;
    if (op_type == op::MatMul) {
        if (num_inputs > 0) {
            for (auto s : inputs[0].shape()) {
                shape_hash = shape_hash * 31 + s + 1;
            }
        }
    } else if (prewalk_cache_count_ > 0) {
        for (size_t li = prewalk_cache_count_; li-- > 0;) {
            const auto& e = prewalkAt(li);
            if (e.op_type == op::MatMul && !e.original_inputs.empty()) {
                for (auto s : e.original_inputs.front().shape()) {
                    shape_hash = shape_hash * 31 + s + 1;
                }
                break;
            }
        }
    }

    // 向后匹配:从最长可能长度到最短
    std::optional<ct::c3::RegionEntry> match;
    for (size_t len = std::min<size_t>(current_pos + 1, 8); len >= 2 && !match; --len) {
        size_t start = current_pos + 1 - len;
        uint64_t op_hash = ct::c3::RollingHash::getSubHash(extended_prefix, start, current_pos);
        uint64_t full_hash = op_hash ^ (shape_hash << 32);
        auto candidate = registry.find(full_hash);
        if (candidate && candidate->active && candidate->len == len) {
            match = candidate;
            break;
        }
    }
    if (!match) {
        return std::nullopt;
    }

    if (!match->cost.worthwhile) {
        return std::nullopt;
    }

    // 从 prewalk_cache_ 取 external inputs
    size_t needed = match->len - 1;
    if (prewalk_cache_count_ < needed) {
        return std::nullopt;
    }

    // shape 校验
    if (!match->first_input_shapes.empty() && needed > 0) {
        const auto& first_cached = prewalkAt(prewalk_cache_count_ - needed).original_inputs;
        if (!first_cached.empty()) {
            const auto& expected_shape = match->first_input_shapes.front();
            const auto& actual_shape = first_cached.front().shape();
            if (expected_shape != actual_shape) {
                return std::nullopt;
            }
        }
    }

    // 收集所有 external inputs
    std::vector<Tensor> external_inputs;
    for (size_t i = prewalk_cache_count_ - needed; i < prewalk_cache_count_; ++i) {
        for (const auto& t : prewalkAt(i).original_inputs) {
            external_inputs.push_back(t);
        }
    }
    if (op_type == op::MatMul) {
        for (size_t k = 0; k < num_inputs; ++k) external_inputs.push_back(inputs[k]);
    } else if (num_inputs > 1) {
        external_inputs.push_back(inputs[1]);
    }

    std::vector<size_t> out_shape = computeOutputShape(op_type, inputs, num_inputs);
    if (out_shape.empty()) {
        return std::nullopt;
    }

    for (const auto& t : external_inputs) {
        if (!t.data_read<float>()) return std::nullopt;
    }

    try {
        ct::c3::KernelShapeInfo shapes;
        if (!external_inputs.empty()) shapes.lhs_shape = external_inputs.front().shape();
        if (external_inputs.size() > 1) shapes.rhs_shape = external_inputs[1].shape();
        shapes.out_shape = out_shape;
        shapes.fused_pattern = "region-fusion";

        Tensor kernel_result = ct::c3::C3KernelRegistry::getInstance()
            .executeFusedWithInputs(match->kernel, external_inputs, shapes);
        if (kernel_result.storage().empty()) {
            return std::nullopt;
        }
        return kernel_result;
    } catch (...) {
        return std::nullopt;
    }
}

std::vector<size_t> CtorchScheduler::computeOutputShape(
    op op_type, const Tensor* inputs, size_t num_inputs) const {
    // DEBT-NEW-7 region fusion 配套：每个 op 的真实输出 shape 计算
    // 用于 prewalk 期间创建 placeholder Tensor(空 storage + LazyMaterializer)
    if (num_inputs == 0) return {};

    // 一元算子：输出 shape = 输入 shape
    // [位掩码 2026-08-11] 一元算子集合 → 单 uint64 位掩码, O(1) 查表,
    // 消除多条件 OR 的分支预测失败惩罚. computeOutputShape 在每次 dispatch
    // 末尾 (prewalk_cache 记录) 调用于热路径, 位掩码化有真实收益.
    // op 连续枚举 (0..kCount-1=27 < 64), 静态断言兜底.
    static_assert(static_cast<size_t>(op::kCount) <= 64,
                  "op::kCount exceeds uint64 bitmask capacity");
    static constexpr uint64_t kUnaryOpMask =
          (1ull << static_cast<size_t>(op::ReLU))
        | (1ull << static_cast<size_t>(op::Tanh))
        | (1ull << static_cast<size_t>(op::Sigmoid))
        | (1ull << static_cast<size_t>(op::Neg))
        | (1ull << static_cast<size_t>(op::Exp))
        | (1ull << static_cast<size_t>(op::Log))
        | (1ull << static_cast<size_t>(op::Abs))
        | (1ull << static_cast<size_t>(op::GELU))
        | (1ull << static_cast<size_t>(op::Softmax))
        | (1ull << static_cast<size_t>(op::SiLU));  // PEL25 #6: SiLU 是单输入 unary, 加进 bitmask
    auto isUnary = [](op t) {
        return (kUnaryOpMask >> static_cast<size_t>(t)) & 1ull;
    };

    if (isUnary(op_type) || num_inputs == 1) {
        return inputs[0].sizes();
    }

    // 二元算子：分两类
    const auto& a = inputs[0].sizes();
    const auto& b = num_inputs > 1 ? inputs[1].sizes() : a;

    if (op_type == op::MatMul) {
        // [M, K] @ [K, N] → [M, N]（最常见的 2D matmul；高维暂不支持）
        if (a.size() >= 2 && b.size() >= 2) {
            return {a[a.size() - 2], b[b.size() - 1]};
        }
        return a;  // 退化 fallback
    }

    // 元素级（Add/Sub/Mul/Div/CE）：广播到较大 shape
    if (a == b) return a;

    // 简化广播：取两边维度的最大值（与 broadCast 逻辑对齐）
    std::vector<size_t> out(std::max(a.size(), b.size()));
    for (size_t i = 0; i < out.size(); ++i) {
        size_t ad = (i < a.size()) ? a[a.size() - 1 - i] : 1;
        size_t bd = (i < b.size()) ? b[b.size() - 1 - i] : 1;
        out[out.size() - 1 - i] = std::max(ad, bd);
    }
    return out;
}

#ifndef CT_DISABLE_C3
#include "C3/C3BackwardCapture.h"
void CtorchScheduler::resetRegionFusion() {
    {
        std::lock_guard<std::mutex> lk(region_trace_mutex_);
        region_trace_.clear();
    }
    prewalk_state_ = PrewalkState::kIdle;
    prewalk_cache_count_ = 0;
    prewalk_cache_head_ = 0;
    matched_region_.reset();
    prewalk_pos_ = 0;
    prewalk_external_inputs_.clear();
    cached_region_ = nullptr;
    cached_hash_ = 0;
    // 清理所有已注册单算子及融合 JIT 内核，确保测试间“环境隔离”
    ct::c3::C3KernelRegistry::getInstance().uninstallAll();
    // 清除反向融合捕获器的所有中间状态，避免跨测试用例残留干扰
    ct::c3::C3BackwardCapture::getInstance().clear();
}
#endif // CT_DISABLE_C3
#endif // CT_DISABLE_C3
