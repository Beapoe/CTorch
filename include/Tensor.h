/**
 * @file Tensor.h
 * @brief 张量类和相关操作的定义
 * @author GhostFace, Beapoe
 * @date 2025/12/21
 * @version v3.1
 * @details 定义了张量类(Tensor)及其相关操作，包括存储管理、自动微分、各种数学运算等。
 */
#ifndef TENSOR_H
#define TENSOR_H

// includes
#include <algorithm>
#include <atomic>
#include <cstddef>
#include "AutogradMeta.h"
#include "CtorchError.h"
#include "Storage.h"
#ifdef __APPLE__
#include <Accelerate/Accelerate.h> // 使用Apple的BLAS实现
extern "C" void MPS_flush_wait(bool wait);
extern "C" void MPS_markBufferModified(void* ptr, size_t bytes);
#endif

#include <initializer_list>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <sstream>
#include <functional>
#include <cstring>
#include <iomanip>
#include <string>
#include <unordered_set>
#include <limits>
#include <map>
#include <mutex>
#include "Ctools.h"

// ======================= 前向声明 =======================
class Tensor;
class Node;

// ======================= 存储类 (Storage) =======================

/**
 * @class Storage
 * @brief Tensor 的存储底层
 * @authors GhostFace Beapoe
 */

// ======================= 张量类 (Tensor) =======================

/**
 * @struct ShapeTag
 * @brief 形状标签类型，用于区分构造函数重载
 * @details 在构造张量时使用此类作为第一个参数，明确表示后续参数为形状而非尺寸。
 *          例如：Tensor(ShapeTag{}, {3, 4}) 创建一个 3x4 的张量。
 */
struct ShapeTag {};

/// @brief 占位符标签，用于创建形状正确但无数据存储的占位张量
struct PlaceholderTag {};

// 前向声明：惰性物化器，完整定义见本文件末尾（class Tensor 之后）
class LazyMaterializer;

/**
 * @brief 矩阵乘法函数
 * @param a 第一个矩阵张量
 * @param b 第二个矩阵张量
 * @return 矩阵乘法结果
 */
Tensor matMul(const Tensor &a, const Tensor &b);

/**
 * @class Tensor
 * @brief 张量类，用于表示和操作多维数组
 * @details 张量是自动微分系统的核心数据结构，支持各种数学运算和自动微分
 */
class Tensor {
  private:
    /**
     * @var _autograd_meta 自动微分元数据，封装所有自动微分相关状态
     */
    AutogradMeta _autograd_meta;

    /**
     * @var global_tensor_id
     * @brief 全局张量ID计数器
     */
    static std::atomic<size_t> global_tensor_id;

    /**
     * @var tensor_id_
     * @brief 张量的唯一标识符
     */
    size_t tensor_id_;

    /**
     * @var _strides
     * @brief 每个维度的步幅
     */
    std::vector<size_t> _strides;

    /**
     * @var _storage_offset
     * @brief 存储中的起始偏移量
     */
    size_t _storage_offset;

    /**
     * @var _device
     * @brief 张量所在的设备
     */
    DeviceType _device;

    /**
     * @var _dtype
     * @brief 张量元素的数据类型
     */
    DType _dtype;

    /**
     * @var _storage
     * @brief 存储张量数据的对象
     */
    Storage _storage;

    /**
     * @var _lazy
     * @brief 惰性物化器（LazyBox）。
     * @details 当张量是区域融合预走的占位结果（空 storage）时，携带一个物化闭包。
     *          首次通过 data_read() 访问时触发物化，按需重算真实中间值，
     *          保证 autograd 后向与用户读取都能拿到正确数据。
     *          非占位张量该成员为空。
     */
    mutable std::shared_ptr<LazyMaterializer> _lazy;

    // ======================= 内部辅助函数 =======================

    /**
     * @brief 检查 in-place 操作是否安全
     * @param op_name 算子名称，用于错误信息
     * @throw CtorchError 若当前张量 requires_grad=true
     */
    void check_inplace_safe_(const char* op_name) const;

    /**
     * @brief 为当前张量创建 GradAccumulator 节点
     * @details 定义放在 Tensor.cpp，避免 Tensor.h 与 GradAccumulator.h 循环包含
     */
    static std::shared_ptr<Node> createGradAccumulator(const std::shared_ptr<Tensor>& self);

    /**
     * @brief [§4.95 P1-03] move 后原地 rebind 节点弱引用(实现见 Tensor.cpp;
     *        Node 在此处为不完整类型, 成员调用必须下沉到 .cpp)
     */
    static void rebindAutogradNode(const std::shared_ptr<Node>& node,
                                   const std::shared_ptr<Tensor>& self);

    /**
     * @brief 计算步幅 (基于行优先顺序)
     */
    void computeStrides();

    /**
     * @brief 计算存储中的索引
     * @param indices 多维索引
     * @return 存储中的一维索引
     */
    [[nodiscard]] size_t computeStorageIndex(std::initializer_list<size_t> indices) const;

    /**
     * @brief 触发惰性物化，返回真实中间值张量（定义在 Tensor.cpp）
     * @details 仅当 _lazy 非空且 _storage 为空时由 data_read() 调用。
     *          物化结果缓存在物化器内，返回引用保证 data_read() 返回的指针在
     *          物化器存活期间始终有效。
     */
    Tensor &materializeLazy_() const;

    /**
     * @brief 检查数据类型是否匹配
     * @tparam T 期望的数据类型
     * @throw std::runtime_error 如果数据类型不匹配
     */
    template <typename T> void checkDType() const;

    /**
     * @brief 通用逐元素操作
     * @tparam T 数据类型
     * @tparam Op 操作类型
     * @param result 结果张量
     * @param a 输入张量a
     * @param b 输入张量b
     * @param op 操作函数
     */
    template <typename T, typename Op>
    void elementwiseOp(Tensor &result, const Tensor &a, const Tensor &b, Op op) const;

    /**
     * @brief 支持广播的逐元素操作
     * @tparam T 数据类型
     * @tparam Op 操作类型
     * @param result 结果张量
     * @param a 输入张量a
     * @param b 输入张量b
     * @param bc 广播结果
     * @param op 操作函数
     */
    template <typename T, typename Op>
    void broadcast_elementwise_op(Tensor &result, const Tensor &a, const Tensor &b,
                                  const BroadCastResult &bc, Op op) const;

    /**
     * @brief 递归打印张量内容
     * @tparam T 数据类型
     * @param os 输出流
     * @param dim 当前维度
     * @param indices 当前索引
     */
    template <typename T>
    void printRecursive(std::ostream &os, size_t dim, std::vector<size_t> indices) const;

  protected:
    /**
     * @var _shape
     * @brief 张量的维度大小
     */
    std::vector<size_t> _shape;
    //

  private:
    // 统一初始化 autograd 弱引用锚点。
    // 值类型 Tensor 无法用 std::enable_shared_from_this（须真正由 shared_ptr 管理才可 shared_from_this），
    // 故用 noop-deleter shared_ptr 提供 getWeakPtr() 的弱引用定位，不持有其生存期。所有构造器统一走此入口。
    void initAutogradSelf() {
        _autograd_meta._self = std::shared_ptr<Tensor>(this, [](Tensor* /*noop del*/) {});
    }

  public:
    // ======================= 构造和析构 =======================

    /**
     * @brief 默认构造函数
     */
    Tensor();

    /**
     * @brief 标量构造函数
     * @param value 标量值
     */
    Tensor(float value)
        : tensor_id_(global_tensor_id++),
          _storage_offset(0), _device(DeviceType::kCPU), _dtype(DType::kFloat) {
        _shape = {};
        initAutogradSelf();
#ifdef CTORCH_DEBUG
        std::ostringstream oss;
        oss << ">>> Tensor标量构造, ID: " << tensor_id_ << ", 值: " << value;
        std::string msg = oss.str();
        CTORCH_TRACE(ErrorPlatform::kCPU, msg);
#endif
        computeStrides();
        _storage = Storage(1, _dtype, _device);
        _autograd_meta._node.reset();
        if (_storage.data<float>()) {
            *_storage.data<float>() = value;
#ifdef CTORCH_DEBUG
            std::ostringstream oss;
            oss << ">>> 标量Tensor设置完成, 存储值: " << *_storage.data<float>();
            std::string msg = oss.str();
            CTORCH_TRACE(ErrorPlatform::kCPU, msg);
#endif
        } else {
            CtorchError::log(ErrorLevel::ERROR, ErrorPlatform::kCPU, ErrorType::MEMORY,
                              "!!! 错误: 无法分配存储");
        }
    }

    /**
     * @brief 标量构造函数（带设备参数）
     * @param value 标量值
     * @param device 设备类型
     */
    Tensor(float value, DeviceType device)
        : tensor_id_(global_tensor_id++),
          _storage_offset(0), _device(device), _dtype(DType::kFloat) {
        _shape = {1};
        initAutogradSelf();
#ifdef CTORCH_DEBUG
        std::ostringstream oss;
        oss << ">>> Tensor标量构造, ID: " << tensor_id_ << ", 值: " << value << ", 设备: " << static_cast<int>(device);
        std::string msg = oss.str();
        CTORCH_TRACE(DeviceTypeToErrorPlatform(device), msg);
#endif
        computeStrides();
        _storage = Storage(1, _dtype, _device);
        _autograd_meta._node.reset();
        if (_storage.data<float>()) {
            *_storage.data<float>() = value;
#ifdef CTORCH_DEBUG
            std::ostringstream oss;
            oss << ">>> 标量Tensor设置完成, 存储值: " << *_storage.data<float>();
            std::string msg = oss.str();
            CTORCH_TRACE(DeviceTypeToErrorPlatform(device), msg);
#endif
        } else {
            CtorchError::log(ErrorLevel::ERROR, DeviceTypeToErrorPlatform(device), ErrorType::MEMORY,
                              "!!! 错误: 无法分配存储");
        }
    }

    /**
     * @brief 初始化列表构造函数
     * @param values 初始化列表
     */
    Tensor(std::initializer_list<float> values)
        : tensor_id_(global_tensor_id++),
          _storage_offset(0), _device(DeviceType::kCPU), _dtype(DType::kFloat) {
        initAutogradSelf();
        _autograd_meta._node.reset();
        _shape = {values.size()};
        computeStrides();
        _storage = Storage(values.begin(), values.size(), _dtype, _device);
    }

    Tensor(std::initializer_list<float> values, DeviceType device)
        : tensor_id_(global_tensor_id++),
          _storage_offset(0), _device(device), _dtype(DType::kFloat) {
        initAutogradSelf();
        _autograd_meta._node.reset();
        _shape = {values.size()};
        computeStrides();
        _storage = Storage(values.begin(), values.size(), _dtype, _device);
    }

    /**
     * @brief 形状标签构造函数
     * @param tag 形状标签
     * @param shape 张量形状
     * @param dtype 数据类型
     * @param device 设备类型
     * @param zero_init 是否零初始化
     */
    Tensor(ShapeTag /*tag*/, const std::vector<size_t> &shape, DType dtype = DType::kFloat,
           DeviceType device = DeviceType::kCPU, bool zero_init = true)
        : tensor_id_(global_tensor_id++),
          _storage_offset(0), _device(device), _dtype(dtype) {
        initAutogradSelf();
        _autograd_meta._node.reset();
        _shape = shape;
        computeStrides();
        _storage = Storage(numel(), _dtype, _device);
        if (zero_init)
            zero();
    }

    /// @brief 占位符构造函数：创建形状正确但无数据存储的占位张量
    /// @param tag 占位符标签
    /// @param shape 张量形状
    /// @param dtype 数据类型
    /// @param device 设备类型
    Tensor(PlaceholderTag /*tag*/, const std::vector<size_t> &shape,
           DType dtype = DType::kFloat, DeviceType device = DeviceType::kCPU)
        : tensor_id_(global_tensor_id++),
          _storage_offset(0), _device(device), _dtype(dtype) {
        initAutogradSelf();
        _autograd_meta._node.reset();
        _shape = shape;
        computeStrides();
        // _storage stays default (empty) - no data allocation
    }

    /**
     * @brief 1D张量构造函数
     * @param size 张量大小
     * @param dtype 数据类型
     * @param device 设备类型
     * @param zero_init 是否零初始化
     */
    Tensor(size_t size, DType dtype = DType::kFloat, DeviceType device = DeviceType::kCPU,
           bool zero_init = true)
        : tensor_id_(global_tensor_id++),
          _storage_offset(0), _device(device), _dtype(dtype) {
        initAutogradSelf();
        _autograd_meta._node.reset();
        _shape = {size};
        computeStrides();
        _storage = Storage(size, _dtype, _device);
        if (zero_init)
            zero();
    }

    /**
     * @brief 拷贝构造函数
     * @param other 被拷贝的张量
     * @details 新对象分配新的张量ID，但与原张量共享底层存储（浅拷贝）。
     *          需要深拷贝时请显式调用 clone()。
     */
    Tensor(const Tensor &other)
        : tensor_id_(global_tensor_id++),
          _strides(other._strides),
          _storage_offset(other._storage_offset), _device(other._device), _dtype(other._dtype),
          _storage(other._storage),
          _shape(other._shape),
          _lazy(other._lazy) {
        initAutogradSelf();
        _autograd_meta._requires_grad = other._autograd_meta._requires_grad;
        _autograd_meta._grad = nullptr;  // 现代语义：copy/assign 不深拷 grad（独立张量各自按需累积）
        if (_autograd_meta._requires_grad) {
            _autograd_meta._node = createGradAccumulator(_autograd_meta._self);
        }
#ifdef CTORCH_DEBUG
        std::ostringstream oss;
        oss << ">>> Tensor拷贝构造, 新ID: " << tensor_id_ << ", 原ID: " << other.tensor_id_;
        std::string msg = oss.str();
        CTORCH_TRACE(ErrorPlatform::kCPU, msg);
#endif
    }

    /**
     * @brief 赋值操作符
     * @param other 被赋值的张量
     * @return 引用当前对象
     * @details 与原张量共享底层存储（浅拷贝）。需要深拷贝时请显式调用 clone()。
     */
    Tensor &operator=(const Tensor &other) {
        if (this != &other) {
            tensor_id_        = global_tensor_id++;
            _shape            = other._shape;
            _strides          = other._strides;
            _storage_offset   = other._storage_offset;
            _device           = other._device;
            _dtype            = other._dtype;
            _storage          = other._storage;
            _lazy             = other._lazy;
            _autograd_meta._requires_grad    = other._autograd_meta._requires_grad;
            _autograd_meta._node.reset();
            _autograd_meta._grad = nullptr;  // 现代语义：copy/assign 不深拷 grad（独立张量各自按需累积）
            initAutogradSelf();
            if (_autograd_meta._requires_grad) {
                _autograd_meta._node = createGradAccumulator(_autograd_meta._self);
            }
        }
        return *this;
    }

    /**
     * @brief 移动构造函数
     * @param other 被移动的张量
     * @details 移动构造后，原对象的tensor_id变为0，避免冲突
     */
    Tensor(Tensor &&other) noexcept
        : _autograd_meta(std::move(other._autograd_meta)), tensor_id_(other.tensor_id_),
          _strides(std::move(other._strides)),
          _storage_offset(other._storage_offset), _device(other._device), _dtype(other._dtype),
          _storage(std::move(other._storage)), _shape(std::move(other._shape)),
          _lazy(std::move(other._lazy)) {
        initAutogradSelf();
        // [Fix 2026-09-10 §4.95 P1-03] move 后原地 rebind _node 的弱引用:
        // initAutogradSelf() 用 this 重建了 _self 控制块, 而 move 来的 _node 内
        // weak_ptr 仍绑旧块(旧块引用归零 → lock() 失败 → 梯度静默丢失)。
        // 注意不能替换节点: _node 可能是中间节点(SumNode 等), 替换成
        // GradAccumulator 会截断梯度链(§4.95 P1-03 修正)。
        if (_autograd_meta._node) {
            rebindAutogradNode(_autograd_meta._node, _autograd_meta._self);
        }
        other.tensor_id_ = 0;
        other._shape.clear();
        other._strides.clear();
        other._storage_offset = 0;
    }

    /**
     * @brief 移动赋值操作符
     * @param other 被移动的张量
     * @return 引用当前对象
     * @details 移动赋值后，原对象的tensor_id变为0，避免冲突
     */
    Tensor &operator=(Tensor &&other) noexcept {
        if (this != &other) {
            tensor_id_        = other.tensor_id_;
            _shape            = std::move(other._shape);
            _strides          = std::move(other._strides);
            _storage_offset   = other._storage_offset;
            _device           = other._device;
            _dtype            = other._dtype;
            _storage          = std::move(other._storage);
            _lazy             = std::move(other._lazy);
            _autograd_meta    = std::move(other._autograd_meta);
            initAutogradSelf();
            // [Fix 2026-09-10 §4.95 P1-03] 同 move 构造: rebind 而非替换(见上)
            if (_autograd_meta._node) {
                rebindAutogradNode(_autograd_meta._node, _autograd_meta._self);
            }

            other.tensor_id_ = 0;
            other._shape.clear();
            other._strides.clear();
            other._storage_offset = 0;
        }
        return *this;
    }

    /**
     * @brief 析构函数
     */
    ~Tensor();

    /**
     * @brief 设置梯度需求
     * @param key 是否需要梯度
     */
    void requires_grad(bool key);

    /**
     * @brief 获取是否需要梯度
     * @return 如果需要梯度返回true，否则返回false
     */
    bool requires_grad() const noexcept { return _autograd_meta._requires_grad; }

    /**
     * @brief 设置梯度需求
     * @param requires_grad 是否需要梯度
     */
    void set_requires_grad(bool requires_grad) { _autograd_meta._requires_grad = requires_grad; }

    // ======================= 访问器 =======================

    /**
     * @brief 获取张量ID
     * @return 张量的唯一标识符
     */
    [[nodiscard]] size_t id() const { return tensor_id_; }

    /**
     * @brief 获取张量的形状
     * @return 张量的形状向量
     */
    [[nodiscard]] const std::vector<size_t> &shape() const;

    /**
     * @brief 获取张量的大小（元素总数量）
     * @return 张量的元素总数量
     */
    [[nodiscard]] size_t numel() const;

    /**
     * @brief 获取张量的步幅
     * @return 张量的步幅向量
     */
    [[nodiscard]] const std::vector<size_t> &strides() const noexcept { return _strides; }

    /**
     * @brief 获取张量的数据类型
     * @return 张量的数据类型
     */
    [[nodiscard]] DType dtype() const noexcept { return _dtype; }

    /**
     * @brief 获取张量所在的设备
     * @return 张量所在的设备
     */
    [[nodiscard]] DeviceType device() const noexcept { return _device; }

    /**
     * @brief 将张量移动到指定设备
     * @param target_device 目标设备
     * @return 移动到目标设备后的新张量
     */
    Tensor to(DeviceType target_device) const;

    /**
     * @brief 获取张量的存储
     * @return 张量的存储对象
     */
    [[nodiscard]] Storage &storage() { return _storage; }

    /**
     * @brief 获取常量存储
     * @return 常量存储对象
     */
    [[nodiscard]] const Storage &storage() const { return _storage; }

    /**
     * @brief 判断是否为惰性占位张量（LazyBox）
     * @return true 表示该张量是区域融合预走产生的占位结果，数据未物化
     * @note 与 storage().empty() 不同的地方：懒占位张量在首次 data_read() 时会自动物化，
     *      因此不能用 storage().empty() 判断（物化后 storage 非空）。此方法不触发物化。
     */
    [[nodiscard]] bool isLazyBox() const noexcept { return _lazy != nullptr; }

    /**
     * @brief 为懒占位张量设置物化闭包（仅区域融合预走使用）
     * @param m 物化器，携带按需重算真实中间值的闭包
     */
    void setLazyMaterializer(std::shared_ptr<LazyMaterializer> m) { _lazy = std::move(m); }

    /**
     * @brief 获取物化器指针（region 融合 kernel 已算出中间值时注入，跳过 eager 重算）
     * @return 物化器，未设置时为空指针
     */
    [[nodiscard]] std::shared_ptr<LazyMaterializer> lazyMaterializer() const noexcept {
        return _lazy;
    }

    /**
     * @brief 获取张量的存储偏移量
     * @return 存储中的起始偏移量
     */
    [[nodiscard]] size_t storage_offset() const noexcept { return _storage_offset; }

    /**
     * @brief 设置存储偏移量（JIT 3.0 零拷贝输出分配）
     */
    void set_storage_offset(size_t offset) { _storage_offset = offset; }

    /**
     * @brief 检查存储偏移量是否有效
     * @return 如果存储偏移量有效返回true，否则返回false
     */
    [[nodiscard]] bool check_storage_offset() const;

    /**
     * @brief 检查索引是否在边界内
     * @param indices 多维索引
     * @return 如果索引在边界内返回true，否则返回false
     */
    [[nodiscard]] bool check_index_bounds(const std::vector<size_t> &indices) const;

    /**
     * @brief 获取张量的维度
     * @return 张量的维度数量
     */
    [[nodiscard]] int dim() const noexcept { return static_cast<int>(_shape.size()); }

    /**
     * @brief 获取张量的维度大小
     * @return 张量的维度大小向量
     */
    [[nodiscard]] const std::vector<size_t> &sizes() const noexcept { return _shape; }

    /**
     * @brief 获取张量指定维度的大小
     * @param dim 维度索引
     * @return 指定维度的大小
     */
    [[nodiscard]] size_t size(int dim) const;

    /**
     * @brief 获取张量指定维度的步幅
     * @param dim 维度索引
     * @return 指定维度的步幅
     */
    [[nodiscard]] size_t stride(int dim) const;

    // ======================= 数据访问 =======================

#ifdef CT_PROFILE_ACCESS
    // 数据访问次数统计（仅在 CT_PROFILE_ACCESS 定义时启用，用于反事实基线测量）。
    // 采用 relaxed 原子计数：不影响生产逻辑，仅用于统计 Eager 路径的读写分布。
    inline static std::atomic<uint64_t> g_data_read_count{0};
    inline static std::atomic<uint64_t> g_data_write_count{0};
#endif

    /**
     * @brief 获取只读原始数据指针（MPS 路径会显式同步）
     * @tparam T 数据类型
     * @return 常量数据指针
     * @throw std::runtime_error 如果数据类型不匹配或存储偏移量无效
     */
    template <typename T> [[nodiscard]] const T *data_read() const {
#ifdef CT_PROFILE_ACCESS
        g_data_read_count.fetch_add(1, std::memory_order_relaxed);
#endif
        // LazyBox 物化收敛点：占位张量首次被读取时，按需重算真实中间值。
        if (_storage.empty() && _lazy) {
            Tensor& real = materializeLazy_();
            if (real.storage().empty()) {
                return nullptr;
            }
            return real.data_read<T>();
        }
        if (_storage.empty()) {
            return nullptr;
        }
        // storage_offset 合法性与溢出已在 view 构造期（operator[]）校验，
        // 此处热路径不再重复检查，减少每次访问的固定开销。
#ifdef __APPLE__
        // MPS 路径异步执行；通过 data_read() 暴露主机指针前必须确保 GPU 写入已完成，
        // 否则调用者可能读到未初始化的值（如 0）。
        if (_device == DeviceType::kMPS) {
            MPS_flush_wait(true);
        }
#endif
        return _storage.data<T>() + _storage_offset;
    }

    /**
     * @brief 获取可写原始数据指针（MPS 路径会自动 mark modified）
     * @tparam T 数据类型
     * @return 可修改的数据指针
     * @throw std::runtime_error 如果数据类型不匹配或存储偏移量无效
     */
    template <typename T> T *data_write() {
#ifdef CT_PROFILE_ACCESS
        g_data_write_count.fetch_add(1, std::memory_order_relaxed);
#endif
        if (_storage.empty()) {
            return nullptr;
        }
        // storage_offset 已在 view 构造期校验，此处热路径不再重复检查。
        T* ptr = _storage.data<T>() + _storage_offset;
#ifdef __APPLE__
        // MPS 路径下 host 写入 shared buffer 后需通知 Metal，使后续 GPU kernel 能看到更新。
        // 注意：data_write() 调用即视为写入意图，即使调用者未实际写入也会产生一次 mark，
        // 这是保证正确性的保守做法。
        if (_device == DeviceType::kMPS) {
            size_t n = numel();
            if (sizeof(T) != 0 && n > SIZE_MAX / sizeof(T)) {
                CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::MEMORY,
                                            "data_write: byte count overflow");
            }
            MPS_markBufferModified(ptr, n * sizeof(T));
        }
#endif
        return ptr;
    }

    /**
     * @brief 获取常量原始数据指针（兼容别名，等价于 data_read()）
     * @deprecated 请优先使用 data_read() / data_write() 以明确读写语义
     * @tparam T 数据类型
     * @return 常量数据指针
     */
    template <typename T> [[nodiscard]] const T *data() const {
        return data_read<T>();
    }

    /**
     * @brief 获取可修改的原始数据指针（兼容别名，MPS 路径仍会同步）
     * @deprecated 请优先使用 data_read() / data_write() 以明确读写语义
     * @tparam T 数据类型
     * @return 可修改的数据指针
     */
    template <typename T> [[nodiscard]] T *data() {
        if (_storage.empty()) {
            return nullptr;
        }
#ifdef __APPLE__
        // 保持兼容：非 const data() 仍先同步再返回指针，避免迁移期间读到旧值。
        if (_device == DeviceType::kMPS) {
            MPS_flush_wait(true);
        }
#endif
        return data_write<T>();
    }

    /**
     * @brief 获取标量值
     * @tparam T 数据类型
     * @return 标量值
     * @throw std::runtime_error 如果张量不是标量或数据类型不匹配
     */
    template <typename T> [[nodiscard]] T item() const;

    /**
     * @brief 索引操作
     * @param index 索引值
     * @return 索引结果张量
     * @throw std::runtime_error 如果索引操作不支持
     */
    Tensor operator[](size_t index) const;

    // ======================= 操作 =======================

    /**
     * @brief 创建一个新的张量，形状相同，数据不同
     * @return 新的张量
     */
    Tensor clone() const;

    /**
     * @brief 将张量转换为指定数据类型
     * @param dtype 目标数据类型
     * @return 转换后的数据类型
     */
    Tensor to(DType dtype) const;

    /**
     * @brief 转置张量
     * @param dim0 第一个维度
     * @param dim1 第二个维度
     * @return 转置后的张量
     * @throw std::runtime_error 如果转置操作不支持
     */
    Tensor transpose(int dim0, int dim1) const;

    /**
     * @brief [§4.95 P1-10] 是否行优先连续(步长 == 紧凑布局)
     * @details 视图操作(transpose/slice)会产生非连续 strides; 若干内核
     *          (Softmax/CrossEntropy) 此前忽略 strides 线性寻址 → 错读。
     *          strides 信息缺失时视为连续(与历史行为一致)。
     */
    [[nodiscard]] bool is_contiguous() const;

    /**
     * @brief [§4.95 P1-10] 物化为连续张量; 已连续则返回自身(浅拷贝)
     */
    [[nodiscard]] Tensor contiguous() const;

    /**
     * @brief 转置张量（二维情况）
     * @return 转置后的张量
     */
    Tensor t() const;

    /**
     * @brief 重塑张量形状
     * @param new_shape 新的形状
     * @return 重塑后的张量
     */
    Tensor reshape(std::initializer_list<size_t> new_shape) const;

    /**
     * @brief 重塑张量形状
     * @param new_shape 新的形状
     * @return 重塑后的张量
     */
    Tensor reshape(const std::vector<size_t> &new_shape) const;

    /**
     * @brief 沿第0维切片（零拷贝视图）
     * @param start 起始索引
     * @param size 切片大小
     * @return 切片后的零拷贝张量
     */
    Tensor slice_dim0(size_t start, size_t size) const;

    /**
     * @brief 广播张量到指定形状
     * @param shape 目标形状
     * @return 广播后的张量
     */
    Tensor broadcast_to(const std::vector<size_t> &shape) const;

    /**
     * @brief 零初始化张量
     */
    void zero();

    /**
     * @brief 一初始化张量
     */
    void ones();

    /**
     * @brief 随机初始化张量
     */
    void rand();

    //  ======================= 运算 =======================

    /**
     * @brief 矩阵乘法
     * @param other 另一个矩阵张量
     * @return 矩阵乘法结果
     */
    Tensor matmul(const Tensor &other) const;

    /**
     * @brief 矩阵乘法操作符重载
     * @param other 另一个矩阵张量
     * @return 矩阵乘法结果
     */
    Tensor operator*(const Tensor &other) const;

    /**
     * @brief 加法操作符重载
     * @param other 另一个张量
     * @return 加法结果
     */
    Tensor operator+(const Tensor &other) const;

    /**
     * @brief 减法操作符重载
     * @param other 另一个张量
     * @return 减法结果
     */
    Tensor operator-(const Tensor &other) const;

    /**
     * @brief 除法操作符重载
     * @param other 另一个张量
     * @return 除法结果
     */
    Tensor operator/(const Tensor &other) const;

    /**
     * @brief 标量乘法操作符重载
     * @param scalar 标量值
     * @return 标量乘法结果
     */
    Tensor operator*(float scalar) const;

    /**
     * @brief 标量加法操作符重载
     * @param scalar 标量值
     * @return 标量加法结果
     */
    Tensor operator+(float scalar) const;

    /**
     * @brief 标量减法操作符重载
     * @param scalar 标量值
     * @return 标量减法结果
     */
    Tensor operator-(float scalar) const;

    /**
     * @brief 标量除法操作符重载
     * @param scalar 标量值
     * @return 标量除法结果
     */
    Tensor operator/(float scalar) const;

    /**
     * @brief 一元减操作符重载
     * @return 取反结果
     */
    Tensor operator-() const;

    /**
     * @brief 取反原地操作
     * @return 当前张量引用
     * @details 原地修改当前张量，不追踪梯度。若当前张量 requires_grad=true 则抛异常。
     */
    Tensor &neg_();

    //  ======================= 线性代数专用（2026-08-10，走 CtorchScheduler）=======================

    /**
     * @brief Givens 旋转：原地修改两个等长向量
     * @param other 第二个向量（被原地修改）
     * @param c cos 系数
     * @param s sin 系数
     * @details 数学定义：
     *            x[i] ← c * x[i] + s * other[i]
     *            other[i] ← c * other[i] - s * x[i]
     *          底层走 cblas_srot（Apple Accelerate）
     */
    void rot(Tensor& other, float c, float s);

    /**
     * @brief Householder 反射应用：原地更新 self
     * @param v Householder 反射向量（m 维，v[0..k_offset-1] 应为 0）
     * @param tau 反射系数
     * @param k_offset 起始行（从哪一行开始 apply，默认 0）
     * @details 操作：
     *            H = I - tau * v * v^T
     *            self[k_offset:, :] := H @ self[k_offset:, :]
     *          底层走 cblas_sgemv + cblas_sger（Apple Accelerate）
     */
    void applyHouseholder(const Tensor& v, float tau, std::size_t k_offset = 0);

    /**
     * @brief 比较操作符重载：大于标量
     * @param scalar 标量值
     * @return 比较结果张量，元素为布尔类型
     */
    Tensor operator>(float scalar) const;

    /**
     * @brief 比较操作符重载：小于标量
     * @param scalar 标量值
     * @return 比较结果张量，元素为布尔类型
     */
    Tensor operator<(float scalar) const;

    /**
     * @brief 比较操作符重载：等于标量
     * @param scalar 标量值
     * @return 比较结果张量，元素为布尔类型
     */
    Tensor operator==(float scalar) const;

    /**
     * @brief 比较操作符重载：大于等于标量
     * @param scalar 标量值
     * @return 比较结果张量，元素为布尔类型
     */
    Tensor operator>=(float scalar) const;

    /**
     * @brief 比较操作符重载：小于等于标量
     * @param scalar 标量值
     * @return 比较结果张量，元素为布尔类型
     */
    Tensor operator<=(float scalar) const;

    /**
     * @brief 比较操作符重载：不等于标量
     * @param scalar 标量值
     * @return 比较结果张量，元素为布尔类型
     */
    Tensor operator!=(float scalar) const;

    /**
     * @brief 比较操作符重载：大于另一个张量
     * @param other 另一个张量
     * @return 比较结果张量，元素为布尔类型
     */
    Tensor operator>(const Tensor &other) const;

    /**
     * @brief 比较操作符重载：小于另一个张量
     * @param other 另一个张量
     * @return 比较结果张量，元素为布尔类型
     */
    Tensor operator<(const Tensor &other) const;

    /**
     * @brief 比较操作符重载：等于另一个张量
     * @param other 另一个张量
     * @return 比较结果张量，元素为布尔类型
     */
    Tensor operator==(const Tensor &other) const;

    /**
     * @brief 比较操作符重载：大于等于另一个张量
     * @param other 另一个张量
     * @return 比较结果张量，元素为布尔类型
     */
    Tensor operator>=(const Tensor &other) const;

    /**
     * @brief 比较操作符重载：小于等于另一个张量
     * @param other 另一个张量
     * @return 比较结果张量，元素为布尔类型
     */
    Tensor operator<=(const Tensor &other) const;

    /**
     * @brief 比较操作符重载：不等于另一个张量
     * @param other 另一个张量
     * @return 比较结果张量，元素为布尔类型
     */
    Tensor operator!=(const Tensor &other) const;

    /**
     * @brief 求和操作
     * @param dims 要求和的维度列表
     * @param keepdim 是否保持原维度
     * @return 求和结果张量
     */
    Tensor sum(const std::vector<int> &dims, bool keepdim = false) const;

    /**
     * @brief 求和操作（单个维度）
     * @param dim 要求和的维度
     * @param keepdim 是否保持原维度
     * @return 求和结果张量
     */
    Tensor sum(int dim, bool keepdim = false) const;

    /**
     * @brief 求和操作（所有维度）
     * @return 求和结果张量（标量）
     */
    Tensor sum() const;

    /**
     * @brief 求平均值操作
     * @param dims 要求平均值的维度列表
     * @param keepdim 是否保持原维度
     * @return 平均值结果张量
     */
    Tensor mean(const std::vector<int> &dims, bool keepdim = false) const;

    /**
     * @brief 求平均值操作（单个维度）
     * @param dim 要求平均值的维度
     * @param keepdim 是否保持原维度
     * @return 平均值结果张量
     */
    Tensor mean(int dim, bool keepdim = false) const;

    /**
     * @brief 求平均值操作（所有维度）
     * @return 平均值结果张量（标量）
     */
    Tensor mean() const;

    /**
     * @brief 求最大值操作
     * @param dims 要求最大值的维度列表
     * @param keepdim 是否保持原维度
     * @return 最大值结果张量
     */
    Tensor max(const std::vector<int> &dims, bool keepdim = false) const;

    /**
     * @brief 求最大值操作（单个维度）
     * @param dim 要求最大值的维度
     * @param keepdim 是否保持原维度
     * @return 最大值结果张量
     */
    Tensor max(int dim, bool keepdim = false) const;

    /**
     * @brief 求最大值操作（所有维度）
     * @return 最大值结果张量（标量）
     */
    Tensor max() const;

    /**
     * @brief 逐元素取最大值
     * @param other 另一个张量
     * @return 逐元素最大值结果张量
     */
    Tensor max(const Tensor& other) const;

    /**
     * @brief 求最小值操作
     * @param dims 要求最小值的维度列表
     * @param keepdim 是否保持原维度
     * @return 最小值结果张量
     */
    Tensor min(const std::vector<int> &dims, bool keepdim = false) const;

    /**
     * @brief 求最小值操作（单个维度）
     * @param dim 要求最小值的维度
     * @param keepdim 是否保持原维度
     * @return 最小值结果张量
     */
    Tensor min(int dim, bool keepdim = false) const;

    /**
     * @brief 求最小值操作（所有维度）
     * @return 最小值结果张量（标量）
     */
    Tensor min() const;

    /**
     * @brief 逐元素取最小值
     * @param other 另一个张量
     * @return 逐元素最小值结果张量
     */
    Tensor min(const Tensor& other) const;

    /**
     * @brief 求标准差操作
     * @param dims 要求标准差的维度列表
     * @param keepdim 是否保持原维度
     * @return 标准差结果张量
     */
    Tensor std(const std::vector<int> &dims, bool keepdim = false) const;

    /**
     * @brief 求标准差操作（单个维度）
     * @param dim 要求标准差的维度
     * @param keepdim 是否保持原维度
     * @return 标准差结果张量
     */
    Tensor std(int dim, bool keepdim = false) const;

    /**
     * @brief 求标准差操作（所有维度）
     * @return 标准差结果张量（标量）
     */
    Tensor std() const;

    /**
     * @brief 求方差操作
     * @param dims 要求方差的维度列表
     * @param keepdim 是否保持原维度
     * @return 方差结果张量
     */
    Tensor var(const std::vector<int> &dims, bool keepdim = false) const;

    /**
     * @brief 求方差操作（单个维度）
     * @param dim 要求方差的维度
     * @param keepdim 是否保持原维度
     * @return 方差结果张量
     */
    Tensor var(int dim, bool keepdim = false) const;

    /**
     * @brief 求方差操作（所有维度）
     * @return 方差结果张量（标量）
     */
    Tensor var() const;

    /**
     * @brief 求绝对值操作
     * @return 绝对值结果张量
     */
    Tensor abs() const;

    /**
     * @brief 绝对值原地操作
     * @return 当前张量引用
     * @details 原地修改当前张量，不追踪梯度。若当前张量 requires_grad=true 则抛异常。
     */
    Tensor &abs_();

    /**
     * @brief 求指数操作
     * @return 指数结果张量
     */
    Tensor exp() const;

    /**
     * @brief 指数原地操作
     * @return 当前张量引用
     * @details 原地修改当前张量，不追踪梯度。若当前张量 requires_grad=true 则抛异常。
     */
    Tensor &exp_();

    /**
     * @brief 求对数操作
     * @return 对数结果张量
     */
    Tensor log() const;

    /**
     * @brief 对数原地操作
     * @return 当前张量引用
     * @details 原地修改当前张量，不追踪梯度。若当前张量 requires_grad=true 则抛异常。
     */
    Tensor &log_();

    /**
     * @brief 求平方根操作
     * @return 平方根结果张量
     */
    Tensor sqrt() const;

    /**
     * @brief 求平方操作
     * @return 平方结果张量
     */
    Tensor square() const;

    /**
     * @brief 求正弦操作
     * @return 正弦结果张量
     */
    Tensor sin() const;

    /**
     * @brief 正弦原地操作
     * @return 当前张量引用
     * @details 原地修改当前张量，不追踪梯度。若当前张量 requires_grad=true 则抛异常。
     */
    Tensor &sin_();

    /**
     * @brief 求余弦操作
     * @return 余弦结果张量
     */
    Tensor cos() const;

    /**
     * @brief 余弦原地操作
     * @return 当前张量引用
     * @details 原地修改当前张量，不追踪梯度。若当前张量 requires_grad=true 则抛异常。
     */
    Tensor &cos_();

    /**
     * @brief 求正切操作
     * @return 正切结果张量
     */
    Tensor tan() const;

    /**
     * @brief 求反正弦操作
     * @return 反正弦结果张量
     */
    Tensor asin() const;

    /**
     * @brief 求反余弦操作
     * @return 反余弦结果张量
     */
    Tensor acos() const;

    /**
     * @brief 求反正切操作
     * @return 反正切结果张量
     */
    Tensor atan() const;

    /**
     * @brief 求双曲正弦操作
     * @return 双曲正弦结果张量
     */
    Tensor sinh() const;

    /**
     * @brief 求双曲余弦操作
     * @return 双曲余弦结果张量
     */
    Tensor cosh() const;

    /**
     * @brief 求双曲正切操作
     * @return 双曲正切结果张量
     */
    Tensor tanh() const;

    /**
     * @brief 双曲正切原地操作
     * @return 当前张量引用
     * @details 原地修改当前张量，不追踪梯度。若当前张量 requires_grad=true 则抛异常。
     */
    Tensor &tanh_();

    /**
     * @brief 求ReLU操作
     * @return ReLU结果张量
     */
    Tensor relu() const;

    /**
     * @brief ReLU原地操作
     * @return 当前张量引用
     * @details 原地修改当前张量，不追踪梯度。若当前张量 requires_grad=true 则抛异常。
     */
    Tensor &relu_();

    /**
     * @brief 求点乘操作
     * @param other 另一个张量
     * @return 点乘结果张量
     */
    Tensor dot(const Tensor &other) const;

    /**
     * @brief 求Leaky ReLU操作
     * @param negative_slope 负斜率值
     * @return Leaky ReLU结果张量
     */
    Tensor leaky_relu(float negative_slope = 0.01f) const;

    /**
     * @brief Leaky ReLU原地操作
     * @param negative_slope 负斜率值（当前仅 0.01f 被 kernel 支持）
     * @return 当前张量引用
     * @details 原地修改当前张量，不追踪梯度。若当前张量 requires_grad=true 则抛异常。
     */
    Tensor &leaky_relu_(float negative_slope = 0.01f);

    /**
     * @brief 求Sigmoid操作
     * @return Sigmoid结果张量
     */
    Tensor sigmoid() const;

    /**
     * @brief Sigmoid原地操作
     * @return 当前张量引用
     * @details 原地修改当前张量，不追踪梯度。若当前张量 requires_grad=true 则抛异常。
     */
    Tensor &sigmoid_();

    /**
     * @brief 求GELU操作
     * @return GELU结果张量
     */
    Tensor gelu() const;

    /**
     * @brief GELU原地操作
     * @return 当前张量引用
     * @details 原地修改当前张量，不追踪梯度。若当前张量 requires_grad=true 则抛异常。
     */
    Tensor &gelu_();

    /**
     * @brief SiLU (Sigmoid Linear Unit) 激活函数
     * @details silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
     *          LLaMA / PaLM / SwiGLU 核心组件
     * @return SiLU结果张量
     * @note Stage 1: Eager CPU 路径, 不接 C3 dispatch (避免 op 枚举扩展)
     */
    Tensor silu() const;

    /**
     * @brief SwiGLU 激活函数 (双输入)
     * @details swiglu(x, gate) = silu(x) * gate
     *          LLaMA / PaLM FFN 核心组件
     * @param gate 门控信号张量, shape 必须跟 *this 一致
     * @return SwiGLU结果张量, shape 跟 *this 一致
     * @note Stage 1: Eager CPU 路径, 不接 C3 dispatch
     */
    Tensor swiglu(const Tensor& gate) const;

    /**
     * @brief 求Softmax操作
     * @param dim 进行Softmax的维度
     * @return Softmax结果张量
     */
    Tensor softmax(int dim = -1) const;
    


    /**
     * @brief 求LogSoftmax操作
     * @param dim 进行LogSoftmax的维度
     * @return LogSoftmax结果张量
     */
    Tensor log_softmax(int dim = -1) const;

    /**
     * @brief 逐元素裁剪操作
     * @param min_val 最小值
     * @param max_val 最大值
     * @return 裁剪后的张量
     */
    Tensor clamp(float min_val, float max_val) const;

    /**
     * @brief 从计算图分离
     * @return 分离后的张量，不参与梯度计算
     */
    Tensor detach() const;

    /**
     * @brief 浅拷贝：共享底层 storage（零数据拷贝），保留 requires_grad 状态，但**不深拷贝 grad**。
     * @note 普通拷贝构造会对 requires_grad 张量深拷其 grad（`_grad->clone()`，如权重 W 的 grad 3.2MB），
     *       在 prewalk 占位捕获这类仅需 data 重算的场景是纯浪费。lazy materialize 重算 op 时会重建 grad 链，
     *       故此处置空 grad + 重建 GradAccumulator，语义对齐 copy。
     * @date 2026-08-27 (RD start 22µs 归因优化)
     */
    Tensor shallow() const;

    /**
     * @brief 求交叉熵损失
     * @param target 目标张量
     * @return 交叉熵损失结果张量
     */
    Tensor cross_entropy(const Tensor &target) const;

    /**
     * @brief 求均方误差
     * @param target 目标张量
     * @return 均方误差结果张量
     */
    Tensor mse_loss(const Tensor &target) const;

    /**
     * @brief 求平均绝对误差
     * @param target 目标张量
     * @return 平均绝对误差结果张量
     */
    Tensor mae_loss(const Tensor &target) const;

    //  ======================= 统一矩阵乘法接口 =======================

    /**
     * @brief 统一矩阵乘法 - 自动选择算法并支持自动微分
     * @param other 另一个矩阵张量
     * @return 矩阵乘法结果
     */
    Tensor matmul_unified(const Tensor &other) const;

    // 矩阵乘法操作符重载 - 使用统一接口
    // 注意：这里不重载operator*，因为已经存在了，使用matmul_unified方法
    // 添加一个辅助函数来创建真正的空 Tensor
    // 保留设置方法

    [[nodiscard]] std::shared_ptr<Node> getRelatedNode() ;
    [[nodiscard]] std::shared_ptr<Node> getRelatedNode() const ;
    void setRelatedNode(std::shared_ptr<Node> ptr) const;

    [[nodiscard]] std::weak_ptr<Tensor> getWeakPtr() const { return _autograd_meta._self; }


    Tensor view(std::initializer_list<size_t> shape);

    void setGrad(std::shared_ptr<Tensor> grad);
    
    /**
     * @brief 获取张量的梯度
     * @return 梯度张量
     */
    Tensor grad() const {
        if (_autograd_meta._grad) {
            return *_autograd_meta._grad;
        }
        // 返回一个与当前张量形状相同的零张量
        return Tensor(ShapeTag{}, _shape, _dtype, _device, true);
    }

    void zero_grad() {
        if (_autograd_meta._grad) {
            _autograd_meta._grad->zero();
        }
    }

    float* grad_ptr() {
        if (_autograd_meta._grad) {
            return _autograd_meta._grad->data<float>();
        }
        return nullptr;
    }

    void detach_autograd() {
        _autograd_meta._node.reset();
    }
};

// ======================= 标量操作符重载（右操作数） =======================

/**
 * @brief 标量加法操作符重载（右操作数）
 * @param scalar 标量值
 * @param tensor 张量
 * @return 加法结果
 */
Tensor operator+(float scalar, const Tensor &tensor);

/**
 * @brief 标量减法操作符重载（右操作数）
 * @param scalar 标量值
 * @param tensor 张量
 * @return 减法结果
 */
Tensor operator-(float scalar, const Tensor &tensor);

/**
 * @brief 标量乘法操作符重载（右操作数）
 * @param scalar 标量值
 * @param tensor 张量
 * @return 乘法结果
 */
Tensor operator*(float scalar, const Tensor &tensor);

/**
 * @brief 标量除法操作符重载（右操作数）
 * @param scalar 标量值
 * @param tensor 张量
 * @return 除法结果
 */
Tensor operator/(float scalar, const Tensor &tensor);

// ======================= 比较操作符重载（右操作数） =======================

/**
 * @brief 比较操作符重载：标量大于张量
 * @param scalar 标量值
 * @param tensor 张量
 * @return 比较结果张量，元素为布尔类型
 */
Tensor operator>(float scalar, const Tensor &tensor);

/**
 * @brief 比较操作符重载：标量小于张量
 * @param scalar 标量值
 * @param tensor 张量
 * @return 比较结果张量，元素为布尔类型
 */
Tensor operator<(float scalar, const Tensor &tensor);

/**
 * @brief 比较操作符重载：标量等于张量
 * @param scalar 标量值
 * @param tensor 张量
 * @return 比较结果张量，元素为布尔类型
 */
Tensor operator==(float scalar, const Tensor &tensor);

/**
 * @brief 比较操作符重载：标量大于等于张量
 * @param scalar 标量值
 * @param tensor 张量
 * @return 比较结果张量，元素为布尔类型
 */
Tensor operator>=(float scalar, const Tensor &tensor);

/**
 * @brief 比较操作符重载：标量小于等于张量
 * @param scalar 标量值
 * @param tensor 张量
 * @return 比较结果张量，元素为布尔类型
 */
Tensor operator<=(float scalar, const Tensor &tensor);

/**
 * @brief 比较操作符重载：标量不等于张量
 * @param scalar 标量值
 * @param tensor 张量
 * @return 比较结果张量，元素为布尔类型
 */
Tensor operator!=(float scalar, const Tensor &tensor);

// ======================= 输出操作符 =======================

/**
 * @brief 输出张量信息
 * @param os 输出流
 * @param tensor 张量
 * @return 输出流
 */
std::ostream &operator<<(std::ostream &os, const Tensor &tensor);

/**
 * @brief 全局的matMul函数
 * @param a 第一个矩阵张量
 * @param b 第二个矩阵张量
 * @return 矩阵乘法结果
 */
Tensor matMul(const Tensor &a, const Tensor &b);

// ======================= 辅助函数 =======================

/**
 * @brief 计算两个张量的广播结果
 * @param a 第一个张量
 * @param tensor2 第二个张量
 * @return 广播结果
 */
BroadCastResult broadCast(const Tensor &a, const Tensor &tensor2);

// ======================= 惰性物化器（LazyBox） =======================

/**
 * @class LazyMaterializer
 * @brief 惰性物化器：封装区域融合预走占位张量的物化闭包。
 * @details 占位张量（空 storage）携带一个物化闭包，首次通过 data_read() 访问时，
 *          触发闭包按需重算真实中间值。物化结果缓存在本对象内（幂等），
 *          返回的引用保证 data_read() 拿到的指针在物化器存活期间始终有效。
 *          线程安全：物化闭包的触发与缓存受互斥锁保护。
 */
class LazyMaterializer {
public:
    /// 物化闭包：返回被物化张量的真实中间值
    using MaterializeFn = std::function<Tensor()>;

    explicit LazyMaterializer(MaterializeFn fn) : fn_(std::move(fn)) {}

    /**
     * @brief 触发物化（幂等），返回真实中间值张量的引用
     * @return 物化结果的引用，缓存在本对象内
     */
    Tensor &materialize() {
        std::lock_guard<std::mutex> lk(mutex_);
        if (!materialized_) {
            result_ = fn_();
            materialized_ = true;
        }
        return result_;
    }

    /**
     * @brief 预置真实中间值（region 融合 kernel 已算出时注入，跳过 eager 重算）
     * @details 幂等：仅在尚未物化时生效。data_read() 后续触发 materialize() 时直接返回预置值，
     *          使 backward 复用融合 kernel 的 pre-activation 值，避免 placeholder 首次读取触发 eager 重算。
     */
    void preload(const Tensor& value) {
        std::lock_guard<std::mutex> lk(mutex_);
        if (!materialized_) {
            result_ = value;
            materialized_ = true;
        }
    }

private:
    MaterializeFn fn_;          ///< 物化闭包
    mutable std::mutex mutex_;  ///< 保护物化状态
    bool materialized_ = false; ///< 是否已物化
    Tensor result_;             ///< 物化结果缓存
};

#endif // TENSOR_H