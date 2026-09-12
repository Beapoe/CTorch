#include <metal_stdlib>
using namespace metal;

kernel void add_kernel(device float* a [[buffer(0)]],
                       device float* b [[buffer(1)]],
                       device float* result [[buffer(2)]],
                       uint idx [[thread_position_in_grid]]) {
    result[idx] = a[idx] + b[idx];
}

kernel void mul_kernel(device float* a [[buffer(0)]],
                       device float* b [[buffer(1)]],
                       device float* result [[buffer(2)]],
                       uint idx [[thread_position_in_grid]]) {
    result[idx] = a[idx] * b[idx];
}

kernel void relu_kernel(device float* a [[buffer(0)]],
                        device float* result [[buffer(1)]],
                        uint idx [[thread_position_in_grid]]) {
    result[idx] = max(a[idx], 0.0f);
}

kernel void lrelu_kernel(device float* a [[buffer(0)]],
                         device float* result [[buffer(1)]],
                         uint idx [[thread_position_in_grid]]) {
    result[idx] = a[idx] > 0.0f ? a[idx] : a[idx] * 0.01f;
}

kernel void lrelu_grad_kernel(device float* x [[buffer(0)]],
                              device float* grad_out [[buffer(1)]],
                              device float* grad_x [[buffer(2)]],
                              uint idx [[thread_position_in_grid]]) {
    grad_x[idx] = x[idx] > 0.0f ? grad_out[idx] : grad_out[idx] * 0.01f;
}

kernel void sigmoid_kernel(device float* a [[buffer(0)]],
                           device float* result [[buffer(1)]],
                           uint idx [[thread_position_in_grid]]) {
    float x = a[idx];
    if (x >= 0.0f) {
        result[idx] = 1.0f / (1.0f + exp(-x));
    } else {
        float exp_x = exp(x);
        result[idx] = exp_x / (1.0f + exp_x);
    }
}

kernel void gelu_kernel(device float* a [[buffer(0)]],
                        device float* result [[buffer(1)]],
                        uint idx [[thread_position_in_grid]]) {
    float x = a[idx];
    float v = 0.7978845608f * (x + 0.044715f * x * x * x);
    result[idx] = 0.5f * x * (1.0f + tanh(v));
}

kernel void matmul_kernel(device float* a [[buffer(0)]],
                          device float* b [[buffer(1)]],
                          device float* result [[buffer(2)]],
                          uint3 pos [[thread_position_in_grid]],
                          constant uint& m [[buffer(3)]],
                          constant uint& k [[buffer(4)]],
                          constant uint& n [[buffer(5)]],
                          constant uint& a_stride0 [[buffer(6)]],
                          constant uint& a_stride1 [[buffer(7)]],
                          constant uint& b_stride0 [[buffer(8)]],
                          constant uint& b_stride1 [[buffer(9)]],
                          constant uint& result_stride0 [[buffer(10)]],
                          constant uint& result_stride1 [[buffer(11)]]) {
    uint row = pos.x;
    uint col = pos.y;
    
    if (row >= m || col >= n) return;
    
    float sum = 0.0f;
    for (uint i = 0; i < k; ++i) {
        sum += a[row * a_stride0 + i * a_stride1] * b[i * b_stride0 + col * b_stride1];
    }
    result[row * result_stride0 + col * result_stride1] = sum;
}

kernel void softmax_kernel(device float* a [[buffer(0)]],
                           device float* result [[buffer(1)]],
                           uint idx [[thread_position_in_grid]],
                           constant uint& batch_size [[buffer(2)]],
                           constant uint& hidden_size [[buffer(3)]]) {
    uint batch_idx = idx / hidden_size;
    uint elem_idx = idx % hidden_size;
    
    float max_val = a[batch_idx * hidden_size];
    for (uint i = 1; i < hidden_size; ++i) {
        max_val = max(max_val, a[batch_idx * hidden_size + i]);
    }
    
    float exp_val = exp(a[idx] - max_val);
    
    float sum_exp = 0.0f;
    for (uint i = 0; i < hidden_size; ++i) {
        sum_exp += exp(a[batch_idx * hidden_size + i] - max_val);
    }
    
    result[idx] = exp_val / sum_exp;
}

kernel void sub_kernel(device float* a [[buffer(0)]],
                       device float* b [[buffer(1)]],
                       device float* result [[buffer(2)]],
                       uint idx [[thread_position_in_grid]]) {
    result[idx] = a[idx] - b[idx];
}

kernel void div_kernel(device float* a [[buffer(0)]],
                       device float* b [[buffer(1)]],
                       device float* result [[buffer(2)]],
                       uint idx [[thread_position_in_grid]]) {
    result[idx] = a[idx] / b[idx];
}

kernel void neg_kernel(device float* a [[buffer(0)]],
                       device float* result [[buffer(1)]],
                       uint idx [[thread_position_in_grid]]) {
    result[idx] = -a[idx];
}

kernel void sin_kernel(device float* a [[buffer(0)]],
                       device float* result [[buffer(1)]],
                       uint idx [[thread_position_in_grid]]) {
    result[idx] = sin(a[idx]);
}

kernel void cos_kernel(device float* a [[buffer(0)]],
                       device float* result [[buffer(1)]],
                       uint idx [[thread_position_in_grid]]) {
    result[idx] = cos(a[idx]);
}

kernel void tanh_kernel(device float* a [[buffer(0)]],
                        device float* result [[buffer(1)]],
                        uint idx [[thread_position_in_grid]]) {
    float x = a[idx];
    // [Fix §4.95 P2] 原 (e^x-e^-x)/(e^x+e^-x) 在 |x|>88 时 inf/inf=NaN;
    // 改对称公式: e^-2|x| 不溢出, t 大时 → ±1(与 CPU 侧 std::tanh/clamp 饱和一致)
    float t = fabs(x);
    float e = exp(-2.0f * t);
    float th = (1.0f - e) / (1.0f + e);
    result[idx] = (x < 0.0f) ? -th : th;
}

kernel void log_kernel(device float* a [[buffer(0)]],
                       device float* result [[buffer(1)]],
                       uint idx [[thread_position_in_grid]]) {
    result[idx] = log(a[idx]);
}

kernel void exp_kernel(device float* a [[buffer(0)]],
                       device float* result [[buffer(1)]],
                       uint idx [[thread_position_in_grid]]) {
    result[idx] = exp(a[idx]);
}

kernel void abs_kernel(device float* a [[buffer(0)]],
                       device float* result [[buffer(1)]],
                       uint idx [[thread_position_in_grid]]) {
    result[idx] = abs(a[idx]);
}

kernel void min_kernel(device float* a [[buffer(0)]],
                       device float* b [[buffer(1)]],
                       device float* result [[buffer(2)]],
                       uint idx [[thread_position_in_grid]]) {
    result[idx] = min(a[idx], b[idx]);
}

kernel void max_kernel(device float* a [[buffer(0)]],
                       device float* b [[buffer(1)]],
                       device float* result [[buffer(2)]],
                       uint idx [[thread_position_in_grid]]) {
    result[idx] = max(a[idx], b[idx]);
}

kernel void mse_kernel(device float* a [[buffer(0)]],
                       device float* b [[buffer(1)]],
                       device float* result [[buffer(2)]],
                       uint idx [[thread_position_in_grid]]) {
    float diff = a[idx] - b[idx];
    result[idx] = diff * diff;
}

kernel void mae_kernel(device float* a [[buffer(0)]],
                       device float* b [[buffer(1)]],
                       device float* result [[buffer(2)]],
                       uint idx [[thread_position_in_grid]]) {
    result[idx] = abs(a[idx] - b[idx]);
}

kernel void cross_entropy_kernel(device float* logits [[buffer(0)]],
                                 device float* targets [[buffer(1)]],
                                 device float* result [[buffer(2)]],
                                 uint idx [[thread_position_in_grid]],
                                 constant uint& batch_size [[buffer(3)]],
                                 constant uint& num_classes [[buffer(4)]]) {
    uint batch_idx = idx / num_classes;
    uint class_idx = idx % num_classes;
    
    float max_val = logits[batch_idx * num_classes];
    for (uint i = 1; i < num_classes; ++i) {
        max_val = max(max_val, logits[batch_idx * num_classes + i]);
    }
    
    float sum_exp = 0.0f;
    for (uint i = 0; i < num_classes; ++i) {
        sum_exp += exp(logits[batch_idx * num_classes + i] - max_val);
    }
    
    float prob = exp(logits[idx] - max_val) / sum_exp;
    
    if (class_idx == uint(targets[batch_idx])) {
        result[idx] = -log(prob);
    } else {
        result[idx] = 0.0f;
    }
}