/**
 * @file CrossEntropy_SIMD_kernel.cpp
 * @brief CPU-SIMD 交叉熵算子（集成 SIMDMath 向量化实现）
 * @author GhostFace
 * @date 2026/08/03
 * @see SIMDMath.h 向量化超越函数库
 *
 * 算法：
 *   CE = -1/N * sum_i sum_c y_ic * log(softmax(x)_ic)
 *       = -1/N * sum_i sum_c y_ic * (x_ic - logsumexp(x_i))
 *
 *   logsumexp(x_i) = log(sum_c exp(x_ic - max_c)) + max_c
 *
 * 数值稳定性：max-subtraction trick。
 */

#include "../kernels.h"
#include "../../../include/CtorchError.h"
#include "../../../include/Tensor.h"
#include "../../../include/CoreDefs.h"
#include "../../../include/kernels/SIMDMath.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <memory>

CT_HOT Tensor CrossEntropy_SIMD_kernel(const Tensor& a, const Tensor& b) {
    if (a.device() != DeviceType::kCPU || b.device() != DeviceType::kCPU) [[unlikely]] {
        CtorchError::log(ErrorLevel::ERROR, DeviceTypeToErrorPlatform(a.device()),
                         ErrorType::DEVICE_COMPAT,
                         "CPU-SIMD CrossEntropy_Kernel: 仅在CPU支持");
    }
    if (a.dtype() != b.dtype()) [[unlikely]] {
        CtorchError::log(ErrorLevel::ERROR, ErrorPlatform::kGENERAL, ErrorType::DATATYPE,
                         "CPU-SIMD CrossEntropy_Kernel: 张量数据类型不一致");
    }

    // [Fix 2026-09-10 §4.95 P1-10] 非连续视图(transpose/slice)先物化,
    // 此前线性寻址忽略 strides 会错读
    Tensor work_a = a.is_contiguous() ? a : a.contiguous();
    Tensor work_b = b.is_contiguous() ? b : b.contiguous();
    const float* CT_RESTRICT data_a = work_a.data_read<float>();
    const float* CT_RESTRICT data_b = work_b.data_read<float>();
    float cross_entropy = 0.0f;

    if (a.numel() == 0) {
        return Tensor(0.0f);
    }

    // 临时 buffer：用于 vexp 写入中间 exp 结果
    // [HPC 2026-08-11] 消除每行堆分配：num_classes 小时用固定栈数组（零 malloc），
    // 仅大 num_classes 回退堆。避免交叉熵训练热路径上 batch_size 次 allocation。
    static constexpr size_t kStackBufElems = 256;  // 覆盖 MNIST(10)/常见分类(<=256)类
    auto compute_logsumexp = [](const float* row, size_t n) -> float {
        // 1. max
        float m = row[0];
        for (size_t j = 1; j < n; ++j) m = std::max(m, row[j]);

        // 2. exp(x - m) 累加：先 shift by max，再 vexp，再 horizontal sum
        float stack_buf[kStackBufElems];
        float* tmp = stack_buf;
        std::unique_ptr<float[]> heap_buf;
        if (n > kStackBufElems) { heap_buf = std::make_unique<float[]>(n); tmp = heap_buf.get(); }
        for (size_t j = 0; j < n; ++j) tmp[j] = row[j] - m;
        ct::kernels::simd::vexp(tmp, tmp, n);
        float s = 0.0f;
#if defined(__AVX2__)
        size_t j = 0;
        __m256 acc = _mm256_setzero_ps();
        for (; j + 7 < n; j += 8) {
            __m256 v = _mm256_loadu_ps(&tmp[j]);
            acc = _mm256_add_ps(acc, v);
        }
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 ss = _mm_add_ps(lo, hi);
        ss = _mm_hadd_ps(ss, ss);
        ss = _mm_hadd_ps(ss, ss);
        s = _mm_cvtss_f32(ss);
        for (; j < n; ++j) s += tmp[j];
#elif defined(__aarch64__)
        size_t j = 0;
        float32x4_t acc = vdupq_n_f32(0.0f);
        for (; j + 3 < n; j += 4) {
            float32x4_t v = vld1q_f32(&tmp[j]);
            acc = vaddq_f32(acc, v);
        }
        s = vgetq_lane_f32(acc, 0) + vgetq_lane_f32(acc, 1)
          + vgetq_lane_f32(acc, 2) + vgetq_lane_f32(acc, 3);
        for (; j < n; ++j) s += tmp[j];
#else
        for (size_t j = 0; j < n; ++j) s += tmp[j];
#endif
        return m + std::log(s);
    };

    if (a.sizes() == b.sizes()) {
        if (a.sizes().size() == 2) {
            size_t batch_size = a.sizes()[0];
            size_t num_classes = a.sizes()[1];

            #pragma omp parallel for reduction(+:cross_entropy)
            for (size_t i = 0; i < batch_size; ++i) {
                const float* row = data_a + i * num_classes;
                float lse = compute_logsumexp(row, num_classes);
                // CE_i = -sum_c y_ic * (x_ic - lse)
                // 简化：先算 sum_c y_ic * x_ic，再减 lse * sum_c y_ic
                float sum_yx = 0.0f;
                float sum_y  = 0.0f;
                for (size_t j = 0; j < num_classes; ++j) {
                    sum_yx += data_b[i * num_classes + j] * data_a[i * num_classes + j];
                    sum_y  += data_b[i * num_classes + j];
                }
                cross_entropy -= sum_yx - lse * sum_y;
            }
        } else if (a.sizes().size() == 1) {
            size_t num_classes = a.sizes()[0];
            float lse = compute_logsumexp(data_a, num_classes);
            float sum_yx = 0.0f;
            float sum_y  = 0.0f;
            for (size_t j = 0; j < num_classes; ++j) {
                sum_yx += data_b[j] * data_a[j];
                sum_y  += data_b[j];
            }
            cross_entropy -= sum_yx - lse * sum_y;
        } else {
            CtorchError::log(ErrorLevel::ERROR, ErrorPlatform::kGENERAL, ErrorType::DIMENSION,
                             "CPU-SIMD CrossEntropy_Kernel: one-hot 仅支持 1D/2D");
            return Tensor();
        }
    } else if (b.sizes().size() == 1 && a.sizes().size() == 2 && b.sizes()[0] == a.sizes()[0]) {
        // class indices 形式：CE = -1/N * sum_i log(exp(x_i_c) / sum_c exp(x_i_c))
        //                       = -1/N * sum_i (lse(x_i) - x_i_c)
        size_t batch_size = b.sizes()[0];
        size_t num_classes = a.sizes()[1];

        #pragma omp parallel for reduction(+:cross_entropy)
        for (size_t i = 0; i < batch_size; ++i) {
            int class_idx = static_cast<int>(data_b[i]);
            if (class_idx < 0 || class_idx >= static_cast<int>(num_classes)) {
                CtorchError::log(ErrorLevel::ERROR, ErrorPlatform::kGENERAL, ErrorType::TENSOR_STATE,
                                 "CPU-SIMD CrossEntropy_Kernel: 类别索引超出范围");
                continue;
            }
            const float* row = data_a + i * num_classes;
            float lse = compute_logsumexp(row, num_classes);
            cross_entropy -= lse - data_a[i * num_classes + class_idx];
        }
    } else {
        CtorchError::log(ErrorLevel::ERROR, ErrorPlatform::kGENERAL, ErrorType::DIMENSION,
                         "CPU-SIMD CrossEntropy_Kernel: 张量形状不兼容");
        return Tensor();
    }

    size_t batch_size = (a.sizes().size() == 2) ? a.sizes()[0] : 1;
    float avg_loss = cross_entropy / static_cast<float>(batch_size);
    return Tensor(avg_loss);
}
