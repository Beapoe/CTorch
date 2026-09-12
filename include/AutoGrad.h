/**
 * @file AutoGrad.h
 * @author Beapoe
 * @brief 自动微分类接口
 * @date 2026/4/4
 **/

#ifndef CTORCH_AUTOGRAD_H
#define CTORCH_AUTOGRAD_H

// 前向声明
class Tensor;
class Node;

// 包含必要的头文件
#include "AutoGrad/DataCore.h"
#include "AutoGrad/ComputeCore.h"
#include "CtorchScheduler.h"
#include "AutoGrad/Nodes/AddNode.h"
#include "AutoGrad/Nodes/SubNode.h"
#include "AutoGrad/Nodes/MulNode.h"
#include "AutoGrad/Nodes/DivNode.h"
#include "AutoGrad/Nodes/NegNode.h"
#include "AutoGrad/Nodes/ReLUNode.h"
#include "AutoGrad/Nodes/LReLUNode.h"
#include "AutoGrad/Nodes/DotNode.h"
#include "AutoGrad/Nodes/CosNode.h"
#include "AutoGrad/Nodes/SinNode.h"
#include "AutoGrad/Nodes/TanhNode.h"
#include "AutoGrad/Nodes/SigmoidNode.h"
#include "AutoGrad/Nodes/GELUNode.h"
#include "AutoGrad/Nodes/SiLUNode.h"  // PEL25 Stage 5.1: SiLU dispatch
#include "AutoGrad/Nodes/SwiGLUNode.h"  // PEL25 Stage 5.1: SwiGLU dispatch (双输入)
#include "AutoGrad/Nodes/MatMulNode.h"
#include "AutoGrad/Nodes/CrossEntropyNode.h"
#include "AutoGrad/Nodes/SoftmaxNode.h"
#include "AutoGrad/Nodes/GradAccumulator.h"
#include "AutoGrad/Nodes/LogNode.h"
#include "AutoGrad/Nodes/ExpNode.h"
#include "AutoGrad/Nodes/AbsNode.h"
#include "AutoGrad/Nodes/MinNode.h"
#include "AutoGrad/Nodes/MaxNode.h"
#include "Arena.h"

namespace AutoGrad {

    //线程本地的全局变量，用于控制是否记录计算图
    inline thread_local bool EnableGrad{true};

    template <typename T>
    void registerNode(const Tensor& input, std::weak_ptr<Tensor> result) {
        if (EnableGrad) {
            DataCore::registerNode<T>(input, result);
        }
    }

    template <typename T>
    void registerNode(const std::vector<Tensor>& inputs, std::weak_ptr<Tensor> result) {
        if (EnableGrad) {
            if (inputs.size() == 1) {
                DataCore::registerNode<T>(inputs[0], result);
            }
            else if (inputs.size() == 2) {
                DataCore::registerNode<T>(inputs[0], inputs[1], result);
            }
        }
    }

    template <typename T>
    void registerNode(const Tensor& a, const Tensor& b, std::weak_ptr<Tensor> result) {
        if (EnableGrad) {
            DataCore::registerNode<T>(a, b, result);
        }
    }

    void backward(std::shared_ptr<Node> root, bool retainGraph);

    inline Tensor dispatch(const Tensor& a, const Tensor& b, op op_type) {
        Tensor result = CtorchScheduler::getInstance().dispatch(a, b, op_type);

        if (EnableGrad && (a.requires_grad() || b.requires_grad())) {
            result.requires_grad(true);
            auto result_ptr = std::make_shared<Tensor>(result);
            std::weak_ptr<Tensor> result_weak = result_ptr;
            switch (op_type) {
            case op::Add:
                registerNode<AddNode>(a, b, result_weak);
                break;
            case op::Sub:
                registerNode<SubNode>(a, b, result_weak);
                break;
            case op::Mul:
                registerNode<MulNode>(a, b, result_weak);
                break;
            case op::Div:
                registerNode<DivNode>(a, b, result_weak);
                break;
            case op::MatMul:
                registerNode<MatMulNode>(a, b, result_weak);
                break;
            case op::CE:
                registerNode<CrossEntropyNode>(a, b, result_weak);
                break;
            default:
                break;
            }
            if (result_ptr->getRelatedNode()) {
                result.setRelatedNode(result_ptr->getRelatedNode());
                result_ptr->getRelatedNode()->setResultOwner(result_ptr);
            }
        }

        return result;
    }

    template <op OpType>
    inline Tensor dispatch(const Tensor& a, const Tensor& b) {
        Tensor result = CtorchScheduler::getInstance().dispatch<OpType>(a, b);

        if (EnableGrad && (a.requires_grad() || b.requires_grad())) {
            result.requires_grad(true);
            auto result_ptr = std::make_shared<Tensor>(result);
            std::weak_ptr<Tensor> result_weak = result_ptr;
            if constexpr (OpType == op::Add) {
                registerNode<AddNode>(a, b, result_weak);
            } else if constexpr (OpType == op::Sub) {
                registerNode<SubNode>(a, b, result_weak);
            } else if constexpr (OpType == op::Mul) {
                registerNode<MulNode>(a, b, result_weak);
            } else if constexpr (OpType == op::Div) {
                registerNode<DivNode>(a, b, result_weak);
            } else if constexpr (OpType == op::MatMul) {
                registerNode<MatMulNode>(a, b, result_weak);
            } else if constexpr (OpType == op::CE) {
                registerNode<CrossEntropyNode>(a, b, result_weak);
            } else if constexpr (OpType == op::Min) {
                registerNode<MinNode>(a, b, result_weak);
            } else if constexpr (OpType == op::Max) {
                registerNode<MaxNode>(a, b, result_weak);
            } else if constexpr (OpType == op::SwiGLU) {  // PEL25 Stage 5.1: SwiGLU dispatch
                registerNode<SwiGLUNode>(a, b, result_weak);
            } else if constexpr (OpType == op::Dot) {  // [§4.97 ⑥] dot 反向断链修复
                registerNode<DotNode>(a, b, result_weak);
            }
            if (result_ptr->getRelatedNode()) {
                result.setRelatedNode(result_ptr->getRelatedNode());
                result_ptr->getRelatedNode()->setResultOwner(result_ptr);
            }
        }

        return result;
    }

    inline Tensor dispatch(const Tensor& a, op op_type) {
        Tensor result = CtorchScheduler::getInstance().dispatch(a, op_type);

        if (EnableGrad && a.requires_grad()) {
            result.requires_grad(true);
            auto result_ptr = std::make_shared<Tensor>(result);
            std::weak_ptr<Tensor> result_weak = result_ptr;
            switch (op_type) {
                case op::Neg:
                    registerNode<NegNode>(a, result_weak);
                    break;
                case op::ReLU:
                    registerNode<ReLUNode>(a, result_weak);
                    break;
                case op::LReLU:
                    registerNode<LReLUNode>(a, result_weak);
                    break;
                case op::Cos:
                    registerNode<CosNode>(a, result_weak);
                    break;
                case op::Sin:
                    registerNode<SinNode>(a, result_weak);
                    break;
                case op::Tanh:
                    registerNode<TanhNode>(a, result_weak);
                    break;
                case op::Sigmoid:
                registerNode<SigmoidNode>(a, result_weak);
                break;
            case op::GELU:
                registerNode<GELUNode>(a, result_weak);
                break;
            default:
                break;
            }
            if (result_ptr->getRelatedNode()) {
                result.setRelatedNode(result_ptr->getRelatedNode());
                result_ptr->getRelatedNode()->setResultOwner(result_ptr);
            }
        }

        return result;
    }

    template <op OpType>
    inline Tensor dispatch(const Tensor& a) {
        Tensor result = CtorchScheduler::getInstance().dispatch<OpType>(a);

        if (EnableGrad && a.requires_grad()) {
            result.requires_grad(true);
            auto result_ptr = std::make_shared<Tensor>(result);
            std::weak_ptr<Tensor> result_weak = result_ptr;
            if constexpr (OpType == op::Neg) {
                registerNode<NegNode>(a, result_weak);
            } else if constexpr (OpType == op::ReLU) {
                registerNode<ReLUNode>(a, result_weak);
            } else if constexpr (OpType == op::LReLU) {
                registerNode<LReLUNode>(a, result_weak);
            } else if constexpr (OpType == op::Cos) {
                registerNode<CosNode>(a, result_weak);
            } else if constexpr (OpType == op::Sin) {
                registerNode<SinNode>(a, result_weak);
            } else if constexpr (OpType == op::Tanh) {
                registerNode<TanhNode>(a, result_weak);
            } else if constexpr (OpType == op::Sigmoid) {
                registerNode<SigmoidNode>(a, result_weak);
            } else if constexpr (OpType == op::GELU) {
                registerNode<GELUNode>(a, result_weak);
            } else if constexpr (OpType == op::Log) {
                registerNode<LogNode>(a, result_weak);
            } else if constexpr (OpType == op::Exp) {
                registerNode<ExpNode>(a, result_weak);
            } else if constexpr (OpType == op::Abs) {
                registerNode<AbsNode>(a, result_weak);
            } else if constexpr (OpType == op::SiLU) {  // PEL25 Stage 5.1: SiLU dispatch
                registerNode<SiLUNode>(a, result_weak);
            }
            if (result_ptr->getRelatedNode()) {
                result.setRelatedNode(result_ptr->getRelatedNode());
                result_ptr->getRelatedNode()->setResultOwner(result_ptr);
            }
        }

        return result;
    }

    inline Tensor dispatch_softmax(const Tensor& a, int dim = -1) {
        Tensor result = CtorchScheduler::getInstance().dispatch_softmax(a, dim);

        if (EnableGrad && a.requires_grad()) {
            result.requires_grad(true);
            auto result_ptr = std::make_shared<Tensor>(result);
            std::weak_ptr<Tensor> result_weak = result_ptr;

            Arena &arena = Arena::getInstance();
            std::vector<std::shared_ptr<Node>> upStreamNodes;

            if (a.getRelatedNode() == nullptr) {
                a.setRelatedNode(arena.invoke<GradAccumulator>(a.getWeakPtr()));
            }
            upStreamNodes.push_back(a.getRelatedNode());

            std::vector<Tensor> inputs = {a};
            const auto node = arena.invoke<SoftmaxNode>(upStreamNodes, inputs, result_weak, dim);
            if (result_weak.lock()) {
                result_weak.lock()->setRelatedNode(node);
                for (auto& upStream : node->getUpStreamNodes()) {
                    if (upStream != nullptr) {
                        upStream->increase();
                    }
                }
            }
            if (result_ptr->getRelatedNode()) {
                result.setRelatedNode(result_ptr->getRelatedNode());
                result_ptr->getRelatedNode()->setResultOwner(result_ptr);
            }
        }

        return result;
    }

};

#endif // CTORCH_AUTOGRAD_H
