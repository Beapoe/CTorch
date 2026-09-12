/**
 *@file GradAccumulator.cpp
 *@brief 梯度累加器(叶子节点)
 *@author Beapoe
 *@date 2026/4/3
 **/

#include "../include/AutoGrad/Nodes/GradAccumulator.h"
#include "../include/Tensor.h"
#include "../../../src/kernels/kernels.h"

GradAccumulator::GradAccumulator(std::weak_ptr<Tensor> tensor) : _tensor(std::move(tensor)) {
    _upStreamNodes = std::vector<std::shared_ptr<Node>>();
    _inputs = std::vector<Tensor>();
    if (auto t = _tensor.lock()) {
        CTORCH_TRACE(ErrorPlatform::kAutoDiff, "GradAccumulator::GradAccumulator - Created for tensor with requires_grad: " + std::to_string(t->requires_grad()));
    }
}

std::vector<GradPack> GradAccumulator::backward(const std::vector<Tensor>& downStreamGrads) {
    if (downStreamGrads.empty()) {
        return {};
    }

    if (auto tensor = _tensor.lock()) {
        if (tensor->device() == DeviceType::kMPS) {
#ifdef __APPLE__
            MPS_flush_wait(true);
#endif
            Tensor accumulated = downStreamGrads[0];

            for (size_t i = 1; i < downStreamGrads.size(); ++i) {
                if (downStreamGrads[i].numel() > 0) {
                    Tensor add_result = accumulated + downStreamGrads[i];
                    accumulated = std::move(add_result);
                }
            }

            auto existing_grad = tensor->grad();
            if (existing_grad.numel() > 0 && existing_grad.storage().data<float>() != nullptr) {
                Tensor add_result = accumulated + existing_grad;
                accumulated = std::move(add_result);
            }

            tensor->setGrad(std::make_shared<Tensor>(std::move(accumulated)));
        } else {
            // 用调度器走 SIMD/AMX 加法，替代标量循环累加
            Tensor accumulated;
            // 单梯度快速路径：绝大多数场景（SGD、单消费）只有一个下游梯度
            if (downStreamGrads.size() == 1 && downStreamGrads[0].numel() > 0) {
                accumulated = downStreamGrads[0];
            } else {
                size_t start_idx = 0;
                while (start_idx < downStreamGrads.size() && downStreamGrads[start_idx].numel() == 0) {
                    ++start_idx;
                }
                if (start_idx < downStreamGrads.size()) {
                    accumulated = downStreamGrads[start_idx];
                    for (size_t i = start_idx + 1; i < downStreamGrads.size(); ++i) {
                        if (downStreamGrads[i].numel() > 0) {
                            accumulated = Add_SIMD_kernel(accumulated, downStreamGrads[i]);
                        }
                    }
                } else {
                    accumulated = Tensor(ShapeTag{}, tensor->shape(), tensor->dtype(), tensor->device());
                    accumulated.zero();
                }
            }

            // 仅当确有已有梯度时累加：grad_ptr() 探测避免 grad() 返回整 Tensor 拷贝
            // [Eager/C3 优化 2026-08-27] 直接调 Add_SIMD_kernel，绕开 operator+/dispatch。
            // [perf 2026-09-05] 升级为原地累加：Add_SIMD_kernel 会新建 Tensor(malloc+zero 整个
            //   梯度 buffer，W1 梯度 800KB 每次白做)。改为直接 SIMD 循环写入已有 grad 的
            //   storage（g[i] += a[i]），零分配、零构造，语义等价（浮点加可交换）。
            if (tensor->grad_ptr() != nullptr) {
                float* g = tensor->grad_ptr();
                const float* a = accumulated.data_read<float>();
                const size_t n = accumulated.numel();
                // [Fix §4.95 P2] 别名检查由指针相等升级为共享 storage 判定:
                // 视图(同 storage 不同 offset)重叠时原地累加会互相污染
                bool same_storage = (accumulated.storage().data<float>() ==
                                     tensor->storage().data<float>());
                if (!same_storage && g != a && n == tensor->numel()) {
                    size_t i = 0;
#if defined(__x86_64__) || defined(__i386__)
                    #if defined(__AVX512F__) && defined(__AVX512DQ__)
                    for (; i + 16 <= n; i += 16) {
                        __m512 vg = _mm512_loadu_ps(g + i);
                        __m512 va = _mm512_loadu_ps(a + i);
                        _mm512_storeu_ps(g + i, _mm512_add_ps(vg, va));
                    }
                    #else
                    for (; i + 8 <= n; i += 8) {
                        __m256 vg = _mm256_loadu_ps(g + i);
                        __m256 va = _mm256_loadu_ps(a + i);
                        _mm256_storeu_ps(g + i, _mm256_add_ps(vg, va));
                    }
                    #endif
#elif defined(__aarch64__)
                    for (; i + 4 <= n; i += 4) {
                        float32x4_t vg = vld1q_f32(g + i);
                        float32x4_t va = vld1q_f32(a + i);
                        vst1q_f32(g + i, vaddq_f32(vg, va));
                    }
#endif
                    for (; i < n; ++i) g[i] += a[i];
                } else {
                    // 别名或形状不匹配的异常路径：保守回退到原实现
                    accumulated = Add_SIMD_kernel(accumulated, tensor->grad());
                    tensor->setGrad(std::make_shared<Tensor>(std::move(accumulated)));
                }
            } else {
                tensor->setGrad(std::make_shared<Tensor>(std::move(accumulated)));
            }
        }
    }
    return {};
}
