/**
 *@file CrossEntropyNode.cpp
 *@author Beapoe
 *@brief 交叉熵损失节点实现
 *@date 2026/4/5
 **/

#include "AutoGrad/Nodes/CrossEntropyNode.h"
#include "../../../src/kernels/kernels.h"

CrossEntropyNode::CrossEntropyNode(const std::vector<std::shared_ptr<Node>> &upStreamNodes,
                                   const std::vector<Tensor> &inputs)
    : Node(upStreamNodes, inputs) {
    set_requireAccelerate(true);
}

CrossEntropyNode::CrossEntropyNode(std::vector<std::shared_ptr<Node>> &&upStreamNodes,
                                   std::vector<Tensor> &&inputs)
    : Node(std::move(upStreamNodes), std::move(inputs)) {
    set_requireAccelerate(true);
}

CrossEntropyNode::CrossEntropyNode(const std::vector<std::shared_ptr<Node>> &upStreamNodes,
                                   const std::vector<Tensor> &inputs,
                                   const std::weak_ptr<Tensor> &result)
    : Node(upStreamNodes, inputs, result) {
    set_requireAccelerate(true);
}

CrossEntropyNode::CrossEntropyNode(std::vector<std::shared_ptr<Node>> &&upStreamNodes,
                                   std::vector<Tensor> &&inputs,
                                   const std::weak_ptr<Tensor> &result)
    : Node(std::move(upStreamNodes), std::move(inputs), result) {
    set_requireAccelerate(true);
}

std::vector<GradPack> CrossEntropyNode::backward(const std::vector<Tensor> &downStreamGrads) {
    std::vector<GradPack> ret;

    if (_inputs.size() != 2) {
        CtorchError::error(ErrorPlatform::kAutoDiff, ErrorType::UNKNOWN,
                           "CrossEntropyNode: 输入数量错误");
        return ret;
    }

    const Tensor &logits = _inputs[0];
    const Tensor &target = _inputs[1];

    if (downStreamGrads.empty()) {
        CtorchError::error(ErrorPlatform::kAutoDiff, ErrorType::UNKNOWN,
                           "CrossEntropyNode: downStreamGrads is empty");
        return ret;
    }

    const Tensor &grad = downStreamGrads[0];

    Tensor softmax_logits = logits.softmax(1);
    Tensor diff           = softmax_logits - target;
#ifdef __APPLE__
    if (logits.device() == DeviceType::kMPS) {
        MPS_flush_wait(true);
    }
#endif

    // [Fix 2026-09-10 §4.95 P1-02] mean loss(forward 除以 batch_size)的反向必须补 1/N:
    // 此前缺 1/N, 梯度被隐式放大 N 倍 ⇒ 等效 lr×N。修复后依赖此梯度的训练配置
    // 需按 N 倍同步调整 lr(见 mnist.cpp: 0.001 → 0.128, 维持与修复前相同的优化轨迹)。
    const size_t batch = (logits.shape().size() > 0) ? logits.shape()[0] : 1;
    Tensor grad_logits = grad * diff * (1.0f / static_cast<float>(batch > 0 ? batch : 1));

#ifdef __APPLE__
    if (logits.device() == DeviceType::kMPS) {
        MPS_flush_wait(true);
    }
#endif

    ret.push_back(GradPack{_upStreamNodes[0], std::vector({grad_logits}), 0});

    return ret;
}