/**
 * @file Softmax_SIMD_kernel.cpp
 * @brief CPU-SIMD Softmax 算子（集成 SIMDMath 向量化实现）
 * @author GhostFace
 * @date 2026/08/03
 * @see SIMDMath.h 向量化超越函数库
 *
 * 算法：
 *   softmax(x_i) = exp(x_i - max) / sum_j exp(x_j - max)
 *
 * 数值稳定性：减去 axis 上的最大值，避免 exp 溢出。
 */

#include "../kernels.h"
#include "../../../include/CtorchError.h"
#include "../../../include/Tensor.h"
#include "../../../include/CoreDefs.h"
#include "../../../include/kernels/SIMDMath.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <cstring>
#include <memory>

CT_HOT Tensor Softmax_SIMD_kernel(const Tensor& a, int dim) {
    if (a.device() != DeviceType::kCPU) [[unlikely]] {
        CtorchError::log(ErrorLevel::ERROR, DeviceTypeToErrorPlatform(a.device()),
                         ErrorType::DEVICE_COMPAT,
                         "CPU-SIMD Softmax_Kernel: 仅在CPU支持");
    }

    int actual_dim = dim;
    if (actual_dim < 0) {
        actual_dim = static_cast<int>(a.sizes().size()) + actual_dim;
    }
    if (actual_dim < 0 || actual_dim >= static_cast<int>(a.sizes().size())) {
        CtorchError::log(ErrorLevel::ERROR, ErrorPlatform::kGENERAL, ErrorType::DIMENSION,
                         "CPU-SIMD Softmax_Kernel: dim 超出范围");
        return Tensor();
    }

    // [Fix 2026-09-10 §4.95 P1-10] 非连续视图(transpose/slice)的 strides 此前被忽略
    // (线性寻址错读), 先物化再计算
    Tensor work = a.is_contiguous() ? a : a.contiguous();
    const auto& shape = work.sizes();
    Tensor result(ShapeTag{}, a.sizes(), a.dtype(), a.device(), false);
    const float* src = work.data_read<float>();
    float* dst = result.data_write<float>();

    if (work.numel() == 0) return result;

    size_t outer = 1;
    for (int i = 0; i < actual_dim; ++i) outer *= shape[i];
    size_t inner = 1;
    for (size_t i = actual_dim + 1; i < shape.size(); ++i) inner *= shape[i];
    size_t axis_size = shape[actual_dim];
    size_t axis_stride = inner;

    // 当 inner==1（即 contiguous 沿 axis）时，可以直接对 axis 维做向量化 exp
    // [HPC 2026-08-11] 每行临时 buffer 用固定栈数组（axis_size<=256 时零 malloc），
    // 仅大 axis 回退堆。消除 OMP parallel-for 循环内每行的堆分配。
    static constexpr size_t kStackBufElems = 256;
    if (inner == 1) {
        // axis 维连续存储，每个 (outer) 行独立处理
        #pragma omp parallel for
        for (size_t o = 0; o < outer; ++o) {
            const float* row_in = src + o * axis_size;
            float* row_out = dst + o * axis_size;

            // 1. max（标量循环，axis_size 通常较小，如 768/1024）
            float max_val = row_in[0];
            for (size_t j = 1; j < axis_size; ++j) {
                max_val = std::max(max_val, row_in[j]);
            }

            // 2. 先 shift by max，再 vexp（exp(x - max) 而非 exp(x) - max）
            float stack_buf[kStackBufElems];
            float* tmp = stack_buf;
            std::unique_ptr<float[]> heap_buf;
            if (axis_size > kStackBufElems) { heap_buf = std::make_unique<float[]>(axis_size); tmp = heap_buf.get(); }
            for (size_t j = 0; j < axis_size; ++j) tmp[j] = row_in[j] - max_val;
            ct::kernels::simd::vexp(tmp, row_out, axis_size);

            // 3. sum row_out（horizontal reduction）
            float exp_sum = 0.0f;
#if defined(__AVX2__)
            size_t j = 0;
            __m256 acc = _mm256_setzero_ps();
            for (; j + 7 < axis_size; j += 8) {
                __m256 v = _mm256_loadu_ps(&row_out[j]);
                acc = _mm256_add_ps(acc, v);
            }
            __m128 hi = _mm256_extractf128_ps(acc, 1);
            __m128 lo = _mm256_castps256_ps128(acc);
            __m128 s = _mm_add_ps(lo, hi);
            s = _mm_hadd_ps(s, s);
            s = _mm_hadd_ps(s, s);
            exp_sum = _mm_cvtss_f32(s);
            for (; j < axis_size; ++j) exp_sum += row_out[j];
#elif defined(__aarch64__)
            size_t j = 0;
            float32x4_t acc = vdupq_n_f32(0.0f);
            for (; j + 3 < axis_size; j += 4) {
                float32x4_t v = vld1q_f32(&row_out[j]);
                acc = vaddq_f32(acc, v);
            }
            exp_sum = vgetq_lane_f32(acc, 0) + vgetq_lane_f32(acc, 1)
                    + vgetq_lane_f32(acc, 2) + vgetq_lane_f32(acc, 3);
            for (; j < axis_size; ++j) exp_sum += row_out[j];
#else
            for (size_t j = 0; j < axis_size; ++j) exp_sum += row_out[j];
#endif

            float inv_sum = 1.0f / exp_sum;

            // 3. 乘 inv_sum 完成归一化
#if defined(__AVX2__)
            __m256 iv = _mm256_set1_ps(inv_sum);
            size_t k = 0;
            for (; k + 7 < axis_size; k += 8) {
                __m256 v = _mm256_loadu_ps(&row_out[k]);
                _mm256_storeu_ps(&row_out[k], _mm256_mul_ps(v, iv));
            }
            for (; k < axis_size; ++k) row_out[k] *= inv_sum;
#elif defined(__aarch64__)
            float32x4_t iv = vdupq_n_f32(inv_sum);
            size_t k = 0;
            for (; k + 3 < axis_size; k += 4) {
                float32x4_t v = vld1q_f32(&row_out[k]);
                vst1q_f32(&row_out[k], vmulq_f32(v, iv));
            }
            for (; k < axis_size; ++k) row_out[k] *= inv_sum;
#else
            for (size_t k = 0; k < axis_size; ++k) row_out[k] *= inv_sum;
#endif
        }
    } else {
        // 非 contiguous（inner > 1）：保留原 OMP simd 实现（向量化 axis_stride 维收益小）
        #pragma omp parallel for
        for (size_t o = 0; o < outer; ++o) {
            for (size_t i = 0; i < inner; ++i) {
                size_t base = o * axis_size * axis_stride + i;
                float max_val = src[base];
                for (size_t j = 1; j < axis_size; ++j) {
                    max_val = std::max(max_val, src[base + j * axis_stride]);
                }
                float exp_sum = 0.0f;
                #pragma omp simd reduction(+:exp_sum)
                for (size_t j = 0; j < axis_size; ++j) {
                    float e = std::exp(src[base + j * axis_stride] - max_val);
                    dst[base + j * axis_stride] = e;
                    exp_sum += e;
                }
                float inv_sum = 1.0f / exp_sum;
                #pragma omp simd
                for (size_t j = 0; j < axis_size; ++j) {
                    dst[base + j * axis_stride] *= inv_sum;
                }
            }
        }
    }

    return result;
}
