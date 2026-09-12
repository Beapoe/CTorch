/**
 * @file SiLU_AMX_kernel.cpp
 * @brief CPU-AMX SiLU 算子 (PEL25 Stage 3.1)
 * @details silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
 *          AMX 槽位对 unary/elementwise 操作显式降级到 SIMD (PEL23 §13 NR-3)
 * @author Mavis (PEL25 §6 SwiGLU 算子开发协议)
 * @date 2026-09-06
 */

#include "../kernels.h"
#include "../../../include/CtorchError.h"
#include "../../../include/Tensor.h"
#include <cmath>

namespace {
inline float silu_scalar(float x) {
    return x / (1.0f + std::exp(-x));
}
}

// AMX 槽位: SiLU 不是 AMX 原生算子 (AMX 主要支持 MatMul-like 矩阵运算),
// 按 performance-optimization-prompt §13 NR-3 AMX 降级规则, 直接调用 SIMD kernel
Tensor SiLU_AMX_kernel(const Tensor& a) {
    if (a.device() != DeviceType::kCPU) {
        CtorchError::log(ErrorLevel::ERROR, DeviceTypeToErrorPlatform(a.device()), ErrorType::DEVICE_COMPAT,
                          "CPU-AMX SiLU_Kernel: 仅在CPU支持");
    }

    // [Fix §4.95 P2] 注释称"直接调用 SIMD kernel"实为标量 exp 循环(潜在 ~10x 退化陷阱);
    // 改为真正委托 SIMD(与 GELU_AMX 降级范式一致)
    CtorchError::log(ErrorLevel::WARN, ErrorPlatform::kAMX, ErrorType::DEVICE_COMPAT,
                      "AMX SiLU_Kernel: 无专用实现，降级到 SIMD");
    return SiLU_SIMD_kernel(a);
}
