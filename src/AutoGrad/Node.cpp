/**
 *@file Node.cpp
 *@author Beapoe
 *@brief 节点实现
 *@date 2026/2/17
 **/

#include "../../include/AutoGrad/Node.h"
#include "../../include/CtorchError.h"
#include "../include/Tensor.h"

void Node::increase() {
    _count.fetch_add(1,std::memory_order_acq_rel);
    _dependencies++;
    // [2026-09-07] 稳定 fanout: 记录被下游注册引用次数(构建期一次, 不随 backward 递减)。
    // MIMO 反向融合等用 getDownstreamCount()==1 做"单消费者"守卫, 防多消费者共享中间丢梯度。
    _downstreamCount.fetch_add(1, std::memory_order_acq_rel);
}

size_t Node::getDownstreamCount() const { return _downstreamCount.load(std::memory_order_acquire); }

bool Node::decrease() {
    size_t old = _count.load(std::memory_order_acquire);
    while (old > 0) {
        if (_count.compare_exchange_strong(old, old - 1, std::memory_order_acq_rel)) {
            return old == 1;
        }
    }
    CtorchError::error(ErrorPlatform::kAutoDiff,ErrorType::UNKNOWN,"Dependency count is negative");
    return false;
}


void Node::restore() { _count.store(_dependencies,std::memory_order_relaxed); }

size_t Node::getDependencies() const { return _dependencies; }

void Node::setDependencies(size_t dependencies) {
    _dependencies = dependencies;
}

void Node::setCount(const size_t count) { _count = count; }

size_t Node::getCount() const { return _count.load(std::memory_order_acquire); }

Node::Node(const std::vector<std::shared_ptr<Node>> &upStreamNodes,
           const std::vector<Tensor> &inputs)
               :_upStreamNodes(upStreamNodes),_inputs(inputs), _dependencies(upStreamNodes.size()) {
    // 初始依赖计数等于上游节点数，保证 Leaf 节点也能正确入队
}

Node::Node(std::vector<std::shared_ptr<Node>>&& upStreamNodes,
           std::vector<Tensor>&& inputs)
               :_upStreamNodes(std::move(upStreamNodes)),_inputs(std::move(inputs)),
                _dependencies(_upStreamNodes.size()) {
    // 初始依赖计数等于上游节点数，保证 Leaf 节点也能正确入队。
    // 注意：必须在 move 完成后从被赋值成员读取 size，源参数已被 move 置空。
}

Node::Node(const std::vector<std::shared_ptr<Node>> &upStreamNodes, const std::vector<Tensor> &inputs,
           const std::weak_ptr<Tensor> &result)
               :_upStreamNodes(upStreamNodes),_inputs(inputs),_result(result), _dependencies(upStreamNodes.size())
{
    if (auto t = result.lock()) {
        _resultShape = t->sizes();
    }
}

Node::Node(std::vector<std::shared_ptr<Node>>&& upStreamNodes,
           std::vector<Tensor>&& inputs,
           const std::weak_ptr<Tensor>& result)
               :_upStreamNodes(std::move(upStreamNodes)),_inputs(std::move(inputs)),_result(result),
                _dependencies(_upStreamNodes.size())
{
    // 注意：必须在 move 完成后从被赋值成员读取 size，源参数已被 move 置空。
    if (auto t = result.lock()) {
        _resultShape = t->sizes();
    }
}

Node::Node(const std::weak_ptr<Tensor> &result)
    :_upStreamNodes(std::vector<std::shared_ptr<Node>>()),_inputs(std::vector<Tensor>()),_result(result), _dependencies(0)
{
    // Leaf 节点：无上游节点，依赖计数为 0，由注册时 increase() 设置
    if (auto t = result.lock()) {
        _resultShape = t->sizes();
    }
}


const std::vector<std::shared_ptr<Node>>& Node::getUpStreamNodes() const { return _upStreamNodes; }

bool Node::requireAccelerate() const { return _requireAccelerate; }

void Node::set_requireAccelerate(bool requireAccelerate) {_requireAccelerate = requireAccelerate;}

void Node::restoreRecursive(std::unordered_set<Node *>& visited) {
    if (!visited.count(this)) {
        visited.insert(this);
        restore();
        for (auto &node : _upStreamNodes)
            if (node) node->restoreRecursive(visited);
    }
}

void Node::clearRecursive(std::unordered_set<Node *>& visited) {
    if (!visited.count(this)) {
        visited.insert(this);
        clearResultOwner();
        auto upstream_copy = _upStreamNodes;
        auto result_copy = _result.lock();
        for (auto &node : upstream_copy)
            if (node) node->clearRecursive(visited);
        _upStreamNodes.clear();
        _inputs.clear();
        if (result_copy) {
            result_copy->detach_autograd();
        }
    }
}
