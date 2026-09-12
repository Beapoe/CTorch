/**
 * @file DotNode.h
 * @brief 点积节点 (dot(x, w) 反向断链修复, §4.97 待办 ⑥)
 * @details 前向: y = Σ x[i] * w[i] (一维点积 → 标量)
 *          反向: grad_x[i] = grad * w[i];  grad_w[i] = grad * x[i]
 * @date 2026-09-12
 **/

#ifndef CTORCH_DOTNODE_H
#define CTORCH_DOTNODE_H

#include "AutoGrad/Node.h"

class DotNode final : public Node {
public:
    DotNode() = default;
    DotNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes, const std::vector<Tensor>& inputs);
    DotNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs);
    DotNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes, const std::vector<Tensor>& inputs,
            const std::weak_ptr<Tensor>& result);
    DotNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs,
            const std::weak_ptr<Tensor>& result);

    std::vector<GradPack> backward(const std::vector<Tensor>& downStreamGrads) override;
};

#endif  // CTORCH_DOTNODE_H
