/**
 * @file Exp_SIMD_kernel.cpp
 * @brief CPU-SIMD Exp算子（指数函数，集成 SIMDMath 向量化实现）
 * @author GhostFace
 * @date 2026/08/03 - 2026/06/30
 * @see SIMDMath.h 向量化超越函数库
 */

#include "../kernels.h"
#include "../../../include/CtorchError.h"
#include "../../../include/Tensor.h"
#include "../../../include/CoreDefs.h"
#include "../../../include/kernels/SIMDMath.h"
#include <cmath>

#ifdef __x86_64__
#include <immintrin.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#endif

CT_HOT Tensor Exp_SIMD_kernel(const Tensor& a) {
    if (a.device() != DeviceType::kCPU) [[unlikely]] {
        CtorchError::log(ErrorLevel::ERROR, DeviceTypeToErrorPlatform(a.device()), ErrorType::DEVICE_COMPAT,
                          "CPU-SIMD Exp_Kernel: 仅在CPU支持");
    }

    Tensor result(ShapeTag{}, a.sizes(), a.dtype(), a.device(), false);
    size_t count = a.numel();
    if (count == 0) return result;

    const float* CT_RESTRICT a_data = a.data_read<float>();
    float* CT_RESTRICT result_data = result.data_write<float>();

#if defined(__x86_64__) && defined(__AVX__)
    size_t i = 0;
    // Intel 编译器 / GCC + SVML：使用硬件加速的 _mm256_exp_ps
    #if defined(__INTEL_COMPILER) || (defined(__GNUC__) && defined(__AVX__) && defined(__SVML__))
    for (; i + 7 < count; i += 8) {
        __m256 x = _mm256_loadu_ps(&a_data[i]);
        __m256 exp_x = _mm256_exp_ps(x);
        _mm256_storeu_ps(&result_data[i], exp_x);
    }
    #endif
    // 主路径：用 SIMDMath 的 exp256_ps（保证 Apple Clang 等无 SVML 环境下也向量化）
    for (; i + 7 < count; i += 8) {
        __m256 x = _mm256_loadu_ps(&a_data[i]);
        _mm256_storeu_ps(&result_data[i], ct::kernels::simd::exp256_ps(x));
    }
    // 处理剩余部分（标量 + clamp，匹配 SIMDMath 内部精度）
    for (; i < count; ++i) {
        float x = a_data[i];
        x = std::min(x, 87.0f);
        x = std::max(x, -87.0f);
        result_data[i] = std::exp(x);
    }
#elif defined(__x86_64__)
    // [Fix §4.95 P2] 无 AVX 的 x86: 原无条件引用仅 __AVX__ 声明的 exp256_ps 会编译失败;
    // 走通用 vexp(内部含标量兜底)
    ct::kernels::simd::vexp(a_data, result_data, count);
#elif defined(__aarch64__)
    // Apple Silicon / ARM64: 调用 SIMDMath 的 exp_neon_f32（4-wide）
    size_t i = 0;
    for (; i + 3 < count; i += 4) {
        float32x4_t x = vld1q_f32(&a_data[i]);
        vst1q_f32(&result_data[i], ct::kernels::simd::exp_neon_f32(x));
    }
    for (; i < count; ++i) {
        float x = a_data[i];
        x = std::min(x, 87.0f);
        x = std::max(x, -87.0f);
        result_data[i] = std::exp(x);
    }
#else
    ct::kernels::simd::vexp(a_data, result_data, count);
#endif

    return result;
}
