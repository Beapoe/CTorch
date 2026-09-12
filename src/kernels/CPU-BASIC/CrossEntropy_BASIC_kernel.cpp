/**
 * @file CrossEntropy_BASIC_kernel.cpp
 * @brief CPU-BASIC CrossEntropy（交叉熵）算子
 * @author GhostFace
 * @date 2026/02/09
 */

#include "./../kernels.h"
#include "./../../../include/CtorchError.h"
#include "./../../../include/Tensor.h"
#include "./../../../include/CoreDefs.h"
#include <cmath>
#include <algorithm>

CT_HOT Tensor CrossEntropy_BASIC_kernel(const Tensor& a, const Tensor& b) {
    // 校验设备：仅支持CPU张量
    if (a.device() != DeviceType::kCPU || b.device() != DeviceType::kCPU) [[unlikely]] {
        CtorchError::log(ErrorLevel::ERROR, DeviceTypeToErrorPlatform(a.device()), ErrorType::DEVICE_COMPAT,
                          "CPU-BASIC CrossEntropy_Kernel: 仅在CPU支持");
    }
    
    // 校验数据类型
    if (a.dtype() != b.dtype()) [[unlikely]] {
        CtorchError::log(ErrorLevel::ERROR, ErrorPlatform::kGENERAL, ErrorType::DATATYPE,
                          "CPU-BASIC CrossEntropy_Kernel: 张量数据类型不一致");
    }
    
    // 实现CrossEntropy损失函数
    // [Fix 2026-09-10 §4.95 P1-10] 非连续视图先物化(与 SIMD 版一致)
    Tensor work_a = a.is_contiguous() ? a : a.contiguous();
    Tensor work_b = b.is_contiguous() ? b : b.contiguous();
    const float* CT_RESTRICT data_a = work_a.data_read<float>();
    const float* CT_RESTRICT data_b = work_b.data_read<float>();
    
    float cross_entropy = 0.0f;
    
    // 支持两种标签格式：
    // 1. 独热编码：形状与模型输出相同 [batch_size, num_classes]
    // 2. 类别索引：形状为 [batch_size]
    if (a.sizes() == b.sizes()) {
        // 独热编码情况
        // 支持两种形状：
        // - [batch_size, num_classes]
        // - [num_classes]（视为 batch_size=1）
        if (a.sizes().size() == 2) {
            size_t batch_size = a.sizes()[0];
            size_t num_classes = a.sizes()[1];
            
            for (size_t i = 0; i < batch_size; ++i) {
                // 对每个样本应用softmax
                float max_val = data_a[i * num_classes];
                for (size_t j = 1; j < num_classes; ++j) {
                    if (data_a[i * num_classes + j] > max_val) {
                        max_val = data_a[i * num_classes + j];
                    }
                }
                
                float exp_sum = 0.0f;
                for (size_t j = 0; j < num_classes; ++j) {
                    exp_sum += std::exp(data_a[i * num_classes + j] - max_val);
                }
                
                for (size_t j = 0; j < num_classes; ++j) {
                    // 计算softmax值
                    float pred = std::exp(data_a[i * num_classes + j] - max_val) / exp_sum;
                    // 确保预测值不为0，避免log(0)的情况
                    pred = std::max(pred, 1e-10f);
                    cross_entropy -= data_b[i * num_classes + j] * std::log(pred);
                }
            }
        } else if (a.sizes().size() == 1) {
            // 单样本：a,b 形状为 [num_classes]
            size_t num_classes = a.sizes()[0];
            float max_val = data_a[0];
            for (size_t j = 1; j < num_classes; ++j) {
                if (data_a[j] > max_val) max_val = data_a[j];
            }
            float exp_sum = 0.0f;
            for (size_t j = 0; j < num_classes; ++j) {
                exp_sum += std::exp(data_a[j] - max_val);
            }
            for (size_t j = 0; j < num_classes; ++j) {
                float pred = std::exp(data_a[j] - max_val) / exp_sum;
                pred = std::max(pred, 1e-10f);
                cross_entropy -= data_b[j] * std::log(pred);
            }
        } else {
            CtorchError::log(ErrorLevel::ERROR, ErrorPlatform::kGENERAL, ErrorType::DIMENSION,
                              "CPU-BASIC CrossEntropy_Kernel: one-hot 仅支持 1D/2D");
            return Tensor();
        }
    } else if (b.sizes().size() == 1 && a.sizes().size() == 2 && b.sizes()[0] == a.sizes()[0]) {
        // 类别索引情况：b是 [batch_size]，a是 [batch_size, num_classes]
        size_t batch_size = b.sizes()[0];
        size_t num_classes = a.sizes()[1];
        
        for (size_t i = 0; i < batch_size; ++i) {
            // 获取类别索引
            int class_idx = static_cast<int>(data_b[i]);
            if (class_idx < 0 || class_idx >= static_cast<int>(num_classes)) {
                CtorchError::log(ErrorLevel::ERROR, ErrorPlatform::kGENERAL, ErrorType::TENSOR_STATE,
                                  "CPU-BASIC CrossEntropy_Kernel: 类别索引超出范围");
                return Tensor();
            }
            
            // 对当前样本应用softmax
            float max_val = data_a[i * num_classes];
            for (size_t j = 1; j < num_classes; ++j) {
                if (data_a[i * num_classes + j] > max_val) {
                    max_val = data_a[i * num_classes + j];
                }
            }
            
            float exp_sum = 0.0f;
            for (size_t j = 0; j < num_classes; ++j) {
                exp_sum += std::exp(data_a[i * num_classes + j] - max_val);
            }
            
            // 获取对应类别的softmax值
            float pred = std::exp(data_a[i * num_classes + class_idx] - max_val) / exp_sum;
            // 确保预测值不为0，避免log(0)的情况
            pred = std::max(pred, 1e-10f);
            cross_entropy -= std::log(pred);
        }
    } else {
        CtorchError::log(ErrorLevel::ERROR, ErrorPlatform::kGENERAL, ErrorType::DIMENSION,
                          "CPU-BASIC CrossEntropy_Kernel: 张量形状不兼容");
        return Tensor();
    }
    
    size_t batch_size = (a.sizes().size() == 2) ? a.sizes()[0] : 1;
    float avg_loss = cross_entropy / static_cast<float>(batch_size);
    
    Tensor result(avg_loss);
    return result;
}