/**
 * @file DotNode.cpp
 * @brief 点积节点实现 (dot 反向断链修复, §4.97 待办 ⑥)
 * @date 2026-09-12
 **/

#include "AutoGrad/Nodes/DotNode.h"
#include "Ctools.h"
#include "CtorchError.h"
#include "Tensor.h"

DotNode::DotNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes, const std::vector<Tensor>& inputs)
    : Node(upStreamNodes, inputs) {}

DotNode::DotNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs)
    : Node(std::move(upStreamNodes), std::move(inputs)) {}

DotNode::DotNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes, const std::vector<Tensor>& inputs,
                 const std::weak_ptr<Tensor>& result)
    : Node(upStreamNodes, inputs, result) {}

DotNode::DotNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs,
                 const std::weak_ptr<Tensor>& result)
    : Node(std::move(upStreamNodes), std::move(inputs), result) {}

std::vector<GradPack> DotNode::backward(const std::vector<Tensor>& downStreamGrads) {
    std::vector<GradPack> ret;

    if (_inputs.size() != 2) {
        CtorchError::error(ErrorPlatform::kAutoDiff, ErrorType::UNKNOWN, "DotNode: 输入数量错误");
        return ret;
    }
    if (downStreamGrads.empty()) {
        return ret;
    }

    const Tensor& x = _inputs[0];
    const Tensor& w = _inputs[1];
    const Tensor& grad = downStreamGrads[0];

    // d(dot)/dx = w * grad;  d(dot)/dw = x * grad
    // 逐元素实现(避免 0D 标量 grad 进入 kernel 的边界问题, 与 MatMul backward
    // 的 0D 处理同思路): grad_x[i] = g * w[i], grad_w[i] = g * x[i]
    const float g = grad.item<float>();
    const size_t n = x.numel();

    Tensor grad_x(ShapeTag{}, x.shape(), x.dtype(), x.device(), false);
    Tensor grad_w(ShapeTag{}, w.shape(), w.dtype(), w.device(), false);
    const float* xp = x.data_read<float>();
    const float* wp = w.data_read<float>();
    float* gxp = grad_x.data_write<float>();
    float* gwp = grad_w.data_write<float>();
    for (size_t i = 0; i < n; ++i) {
        gxp[i] = g * wp[i];
        gwp[i] = g * xp[i];
    }

    ret.push_back(GradPack{_upStreamNodes[0], {grad_x}, 0});
    ret.push_back(GradPack{_upStreamNodes[1], {grad_w}, 1});
    return ret;
}
