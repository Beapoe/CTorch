/**
 * @file Log_SIMD_kernel.cpp
 * @brief CPU-SIMD Log算子（自然对数，集成 SIMDMath 向量化实现）
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

CT_HOT Tensor Log_SIMD_kernel(const Tensor& a) {
    if (a.device() != DeviceType::kCPU) [[unlikely]] {
        CtorchError::log(ErrorLevel::ERROR, DeviceTypeToErrorPlatform(a.device()), ErrorType::DEVICE_COMPAT,
                          "CPU-SIMD Log_Kernel: 仅在CPU支持");
    }

    Tensor result(ShapeTag{}, a.sizes(), a.dtype(), a.device(), false);
    size_t count = a.numel();
    if (count == 0) return result;

    const float* CT_RESTRICT a_data = a.data_read<float>();
    float* CT_RESTRICT result_data = result.data_write<float>();

#if defined(__x86_64__) && defined(__AVX__)
    size_t i = 0;
    // Intel 编译器 / GCC + SVML：使用硬件加速的 _mm256_log_ps
    #if defined(__INTEL_COMPILER) || (defined(__GNUC__) && defined(__AVX__) && defined(__SVML__))
    for (; i + 7 < count; i += 8) {
        __m256 x = _mm256_loadu_ps(&a_data[i]);
        __m256 log_x = _mm256_log_ps(x);
        _mm256_storeu_ps(&result_data[i], log_x);
    }
    #endif
    // 主路径：用 SIMDMath 的 log256_ps
    for (; i + 7 < count; i += 8) {
        __m256 x = _mm256_loadu_ps(&a_data[i]);
        _mm256_storeu_ps(&result_data[i], ct::kernels::simd::log256_ps(x));
    }
    // 标量 tail（调用方需保证 x > 0）
    for (; i < count; ++i) {
        result_data[i] = std::log(a_data[i]);
    }
#elif defined(__x86_64__)
    // [Fix §4.95 P2] 无 AVX 的 x86: 原无条件引用仅 __AVX__ 声明的 log256_ps 会编译失败;
    // 走通用 vlog(内部含标量兜底)
    ct::kernels::simd::vlog(a_data, result_data, count);
#elif defined(__aarch64__)
    size_t i = 0;
    for (; i + 3 < count; i += 4) {
        float32x4_t x = vld1q_f32(&a_data[i]);
        vst1q_f32(&result_data[i], ct::kernels::simd::log_neon_f32(x));
    }
    for (; i < count; ++i) {
        result_data[i] = std::log(a_data[i]);
    }
#else
    ct::kernels::simd::vlog(a_data, result_data, count);
#endif

    return result;
}
