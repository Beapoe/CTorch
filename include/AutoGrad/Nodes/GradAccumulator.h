/**
 * @file GradAccumulator.h
 * @brief 梯度累加器(叶子节点)
 * @author Beapoe
 * @date 2026/4/3
 **/

#ifndef CTORCH_GRADACCUMULATOR_H
#define CTORCH_GRADACCUMULATOR_H
#include "../Node.h"

class GradAccumulator final: public Node {
public:
    explicit GradAccumulator(std::weak_ptr<Tensor> tensor);
    std::vector<GradPack> backward(const std::vector<Tensor>& downStreamGrads) override;
    /// [Fix §4.95 P1-03] rebind: 同步更新 _tensor 与基类 _result
    void rebind(const std::weak_ptr<Tensor>& result) override {
        Node::rebind(result);
        _tensor = result;
    }
private:
    std::weak_ptr<Tensor> _tensor;
};

#endif // CTORCH_GRADACCUMULATOR_H
