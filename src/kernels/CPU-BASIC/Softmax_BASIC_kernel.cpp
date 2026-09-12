/**
 * @file Softmax_BASIC_kernel.cpp
 * @brief CPU-BASIC Softmax算子
 * @author GhostFace
 * @date 2026/02/09
 */

#include <cmath>
#include "./../kernels.h"
#include "./../../../include/CtorchError.h"
#include "./../../../include/Tensor.h"
#include "./../../../include/CoreDefs.h"

CT_HOT Tensor Softmax_BASIC_kernel(const Tensor &a, int dim) {
    if (a.device() != DeviceType::kCPU) [[unlikely]] {
        CtorchError::throwException(DeviceTypeToErrorPlatform(a.device()),
                                     ErrorType::DEVICE_COMPAT,
                                     "CPU-BASIC Softmax_Kernel: 仅在CPU支持");
    }

    if (a.dtype() != DType::kFloat) [[unlikely]] {
        CtorchError::throwException(ErrorPlatform::kCPU, ErrorType::DATATYPE,
                                     "CPU-BASIC Softmax_Kernel: 目前仅支持 float");
    }

    const auto &shape = a.sizes();
    size_t rank = shape.size();
    if (rank == 0) [[unlikely]] {
        CtorchError::throwException(ErrorPlatform::kCPU, ErrorType::DIMENSION,
                                     "CPU-BASIC Softmax_Kernel: 不支持标量");
    }

    int d = dim;
    if (d < 0) d += static_cast<int>(rank);
    if (d < 0 || d >= static_cast<int>(rank)) [[unlikely]] {
        CtorchError::throwException(ErrorPlatform::kCPU, ErrorType::DIMENSION,
                                     "CPU-BASIC Softmax_Kernel: dim 越界");
    }
    size_t softmax_dim = static_cast<size_t>(d);

    // [Fix 2026-09-10 §4.95 P1-10] 非连续视图先物化(与 SIMD 版一致)
    Tensor work = a.is_contiguous() ? a : a.contiguous();

    Tensor result(ShapeTag{}, shape, work.dtype(), work.device(), false);
    const float* CT_RESTRICT in = work.data_read<float>();
    float* CT_RESTRICT out      = result.data_write<float>();

    size_t outer_size = 1;
    for (size_t i = 0; i < softmax_dim; ++i) {
        outer_size *= shape[i];
    }
    size_t inner_size = 1;
    for (size_t i = softmax_dim + 1; i < rank; ++i) {
        inner_size *= shape[i];
    }
    size_t dim_size = shape[softmax_dim];

    for (size_t outer = 0; outer < outer_size; ++outer) {
        for (size_t inner = 0; inner < inner_size; ++inner) {
            size_t base = outer * dim_size * inner_size + inner;

            float max_val = in[base];
            for (size_t j = 1; j < dim_size; ++j) {
                float v = in[base + j * inner_size];
                if (v > max_val) max_val = v;
            }

            float sum = 0.0f;
            for (size_t j = 0; j < dim_size; ++j) {
                float e = std::exp(in[base + j * inner_size] - max_val);
                out[base + j * inner_size] = e;
                sum += e;
            }

            if (sum > 0.0f) {
                for (size_t j = 0; j < dim_size; ++j) {
                    out[base + j * inner_size] /= sum;
                }
            }
        }
    }

    return result;
}
