/**
 * @file Node.h
 * @author Beapoe
 * @brief 自动微分计算图节点定义
 * @date 2026/2/17
 */

#ifndef CTORCH_NODE_H
#define CTORCH_NODE_H
#include <atomic>
#include <unordered_set>

#include "../Tensor.h"
class Node;
#include <vector>
#include <memory>

/**
 * @struct GradPack
 * @brief 梯度传播包，用于反向传播时存储梯度和目标节点信息
 * @details 反向传播过程中，每个节点计算完梯度后，将其打包成 GradPack
 *          发送给上游节点。
 */
struct GradPack{
    /** @brief 梯度要传播到的上游节点 */
    std::shared_ptr<Node> _targetNode;
    /** @brief 梯度张量列表 */
    std::vector<Tensor> _grad;
    /** @brief 当前梯度对应的输入索引 */
    int _idx{0};
};


/**
 * @class Node
 * @brief 自动微分计算图的抽象基类
 * @details 所有计算图节点（如加法、乘法、ReLU等）都继承自此类。
 *          负责管理上游节点、输入张量、输出张量引用以及依赖计数。
 */
class Node {
protected:
    /** @brief 上游节点列表 */
    std::vector<std::shared_ptr<Node>> _upStreamNodes;
    /** @brief 输入张量列表 */
    std::vector<Tensor> _inputs;
    /** @brief 输出张量的弱引用 */
    std::weak_ptr<Tensor> _result;
    /** @brief 输出张量的强引用，防止反向传播时result被释放 */
    std::shared_ptr<Tensor> _result_owner;
    /** @brief 输出张量的形状缓存 */
    std::vector<size_t> _resultShape;
    /** @brief 依赖节点总数 */
    size_t _dependencies{0};
    /** @brief 当前活跃的依赖计数 */
    std::atomic<size_t> _count{0};
    /** @brief 构建时被下游注册引用的次数(稳定 fanout, 不随 backward 变) — 供 MIMO 等单消费者守卫 */
    std::atomic<size_t> _downstreamCount{0};
    /** @brief 是否需要加速计算 */
    bool _requireAccelerate{false};
public:
    /** @brief 默认构造函数 */
    Node() = default;

    /**
     * @brief 构造函数
     * @param upStreamNodes 上游节点列表（const引用）
     * @param inputs 输入张量列表（const引用）
     */
    Node(const std::vector<std::shared_ptr<Node>>& upStreamNodes, const std::vector<Tensor>& inputs);
    
    /**
     * @brief 构造函数（移动语义）
     * @param upStreamNodes 上游节点列表（右值引用）
     * @param inputs 输入张量列表（右值引用）
     */
    Node(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs);

    /**
     * @brief 构造函数（带输出张量引用）
     * @param upStreamNodes 上游节点列表（const引用）
     * @param inputs 输入张量列表（const引用）
     * @param result 输出张量的弱引用
     */
    Node(const std::vector<std::shared_ptr<Node>>& upStreamNodes, const std::vector<Tensor>& inputs, const std::weak_ptr<Tensor>& result);
    
    /**
     * @brief 构造函数（带输出张量引用，移动语义）
     * @param upStreamNodes 上游节点列表（右值引用）
     * @param inputs 输入张量列表（右值引用）
     * @param result 输出张量的弱引用
     */
    Node(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs, const std::weak_ptr<Tensor>& result);

    /**
     * @brief 构造函数（仅输出张量）
     * @param result 输出张量的弱引用
     */
    explicit Node(const std::weak_ptr<Tensor> &result);

    /** @brief 虚析构函数 */
    virtual ~Node() = default;

    /** @brief 增加依赖计数 */
    void increase();

    /**
     * @brief 减少依赖计数
     * @return 当计数减到0时返回true，否则返回false
     */
    bool decrease();

    /** @brief 恢复依赖计数到初始值 */
    void restore();

    /** @brief 获取依赖节点总数 */
    [[nodiscard]] size_t getDependencies() const;

    /** @brief 设置依赖节点总数 */
    void setDependencies(size_t dependencies);

    /** @brief 设置当前活跃的依赖计数 */
    void setCount(size_t count);

    /** @brief 获取当前活跃依赖计数(下游完成倒计时) */
    [[nodiscard]] size_t getCount() const;

    /** @brief 构建时被下游注册引用的次数(稳定 fanout)。MIMO 单消费者守卫用。 */
    [[nodiscard]] size_t getDownstreamCount() const;

    /** @brief 获取上游节点列表（返回 const 引用，避免热路径图遍历每次拷贝） */
    [[nodiscard]] const std::vector<std::shared_ptr<Node>>& getUpStreamNodes() const;

    /** @brief 获取输入张量列表 */
    [[nodiscard]] const std::vector<Tensor>& getInputs() const { return _inputs; }

    /** @brief 检查是否需要加速计算 */
    [[nodiscard]] bool requireAccelerate() const;

    /** @brief 设置是否需要加速计算 */
    void set_requireAccelerate(bool requireAccelerate);

    /**
     * @brief 反向传播接口（纯虚函数）
     * @param downStreamGrads 下游传来的梯度列表
     * @return 梯度传播包列表，包含要发送给上游节点的梯度
     */
    virtual std::vector<GradPack> backward(const std::vector<Tensor>& downStreamGrads) = 0;

    /** @brief 递归恢复依赖计数
     * @param visited 已访问节点集合，防止重复访问
     */
    void restoreRecursive(std::unordered_set<Node*>& visited);

    /** @brief 递归清理节点引用，打破循环引用
     * @param visited 已访问节点集合，防止重复访问
     */
    void clearRecursive(std::unordered_set<Node*>& visited);

    /** @brief 获取该节点对应的输出张量（可能为nullptr如果已被释放） */
    [[nodiscard]] std::shared_ptr<Tensor> getResult() const {
        auto result = _result.lock();
        if (!result && _result_owner) {
            return _result_owner;
        }
        return result;
    }

    /**
     * @brief [Fix §4.95 P1-03 修正] move 后原地更新输出弱引用(不替换节点本身)
     * @details Tensor move 会用 this 重建 _self 控制块, 而本节点持有的弱引用仍绑旧块。
     *          替代方案(用 createGradAccumulator 替换 _node)会摧毁中间节点的真实类型
     *          (如 SumNode), 截断梯度链。rebind 只更新绑定, 保留节点结构与梯度数据。
     *          同时重置 _result_owner(其指向 moved-from 地址, 已失效)。
     */
    virtual void rebind(const std::weak_ptr<Tensor>& result) {
        _result = result;
        _result_owner.reset();
    }

    /** @brief 设置输出张量的强引用所有者，防止反向传播时result被释放 */
    void setResultOwner(std::shared_ptr<Tensor> owner) {
        _result_owner = std::move(owner);
    }

    /** @brief 清理输出张量引用，用于打破循环引用 */
    void clearResultOwner() {
        _result_owner.reset();
    }

    /** @brief 获取输出张量的形状 */
    [[nodiscard]] const std::vector<size_t>& getResultShape() const {
        return _resultShape;
    }
};

#endif // CTORCH_NODE_H
