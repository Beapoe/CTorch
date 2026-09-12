/**
 *@file ComputeCore.cpp
 *@author Beapoe
 *@brief 计算核心
 *@date 2026/2/22
 **/

#include <utility>
#include <queue>
#include <algorithm>
#include <functional>
#include <unordered_map>
#include <unordered_set>

#include "../../include/AutoGrad/ComputeCore.h"
#include "../../include/AutoGrad/Node.h"
#include "../../include/CtorchError.h"
#include "../../include/CtorchScheduler.h"  // ct::detail::set_in_backward（DEBT-NEW-7 H2 fix）
#include "../../include/ThreadPool.h"
#include "../../include/Arena.h"
#include "../../include/AutoGrad.h"
#ifndef CT_DISABLE_C3
#include "C3/C3BackwardCapture.h"  // DEBT-NEW-7 v0.5.1+ 接通 C3 backward fusion
#endif
#ifdef __OBJC__
#include "../../src/kernels/kernels.h"
#endif

GradBucket &GradBucket::getInstance() {
    static GradBucket instance;
    return instance;
}

ssize_t GradBucket::find(const std::shared_ptr<Node> &target) {
    auto it = std::find_if(_packs.begin(), _packs.end(), [target](const GradPack &pack) {
        return pack._targetNode == target;
    });
    if (it != _packs.end())
        return std::distance(_packs.begin(), it);
    return -1;
}

ssize_t GradBucket::find(const std::shared_ptr<Node> &target, int idx) {
    auto it = std::find_if(_packs.begin(), _packs.end(), [target, idx](const GradPack &pack) {
        return pack._targetNode == target && pack._idx == idx;
    });
    if (it != _packs.end())
        return std::distance(_packs.begin(), it);
    return -1;
}

void GradBucket::add(std::vector<GradPack>&& newPacks) {
    std::scoped_lock lock(_mtx);
    for (auto&& pack : newPacks) {
        const ssize_t idx = find(pack._targetNode);
        if (idx != -1) {
            for (size_t i = 0; i < pack._grad.size(); ++i) {
                if (i < _packs[idx]._grad.size()) {
                    if (_packs[idx]._grad[i].numel() > 0) {
                        // 避免 MPS 异步写入完成前触发深拷贝导致梯度归零
                        Tensor add_result = _packs[idx]._grad[i] + pack._grad[i];
                        _packs[idx]._grad[i] = std::move(add_result);
                    } else {
                        _packs[idx]._grad[i] = std::move(pack._grad[i]);
                    }
                } else {
                    _packs[idx]._grad.push_back(std::move(pack._grad[i]));
                }
            }
        } else {
            pack._idx = -1;
            _packs.push_back(std::move(pack));
        }
    }
}

void GradBucket::remove(const std::shared_ptr<Node> &target) {
    std::scoped_lock lock(_mtx);
    const ssize_t idx = find(target);
    if (idx != -1)
        _packs.erase(_packs.begin() + idx);
}

bool GradBucket::tryGetGrad(const std::shared_ptr<Node> &target, std::vector<Tensor> &out_grads) {
    std::scoped_lock lock(_mtx);
    const ssize_t idx = find(target);
    if (idx != -1) {
        out_grads = std::move(_packs[idx]._grad);
        _packs.erase(_packs.begin() + idx);
        return true;
    }
    return false;
}

bool GradBucket::empty() {
    std::scoped_lock lock(_mtx);
    return _packs.empty();
}

void GradBucket::clear() {
    std::scoped_lock lock(_mtx);
    _packs.clear();
}

std::vector<Tensor> GradBucket::operator[](const std::shared_ptr<Node> &target) {
    std::scoped_lock lock(_mtx);
    ssize_t idx = find(target);
    if (idx != -1) {
        auto result = std::move(_packs[idx]._grad);
        _packs.erase(_packs.begin() + idx);
        return result;
    }
    CtorchError::error(ErrorPlatform::kAutoDiff, ErrorType::UNKNOWN,
                        "Trying to visit an nonexistential grad pack");
    return {};
}

std::shared_ptr<Node> ComputeCore::tryPopReadyNode() {
    std::scoped_lock lock(_mtx);
    if (_readyNodes.empty())
        return nullptr;
    auto node = std::move(_readyNodes.front());
    _readyNodes.pop();
    return node;
}

ComputeCore &ComputeCore::getInstance() {
    static ComputeCore instance;
    return instance;
}

void ComputeCore::addReadyNode(std::shared_ptr<Node> node) {
    std::scoped_lock lock(_mtx);
    _readyNodes.push(std::move(node));
    _cv.notify_one();
}

void ComputeCore::backward(std::shared_ptr<Node> root, bool retainGraph) {
    // DEBT-NEW-7 H2 修复:在 backward 期间标记 in_backward=true，
    // 调度器的 inAutogradScope guard 借此识别反向传播路径（其 matmul 输入
    // 通常 requires_grad=false,如 x.T @ grad），跳过 c3 单 kernel 注入。
    // RAII 模式：函数出口自动清除 flag，即使中途抛异常。
    // 注：ct::detail 命名空间仅在 C3 启用时可用(由 CtorchScheduler.h 提供)
#ifndef CT_DISABLE_C3
    bool prev_in_backward = ct::detail::g_in_backward();
    ct::detail::set_in_backward(true);
    struct FlagGuard {
        bool prev;
        ~FlagGuard() { ct::detail::set_in_backward(prev); }
    } guard{prev_in_backward};
#endif

    // [Fix §4.95 P2] EnableGrad 恢复改 RAII: 原异常路径下 thread_local 永久 false
    // (不再记录计算图); in_backward 已有 FlagGuard 同款模式
    struct EnableGradGuard {
        bool prev;
        ~EnableGradGuard() { AutoGrad::EnableGrad = prev; }
    } eg_guard{AutoGrad::EnableGrad};
    AutoGrad::EnableGrad = false;

    // [BW-SEG 2026-08-27] env C3_BW_SEG=1：量化 backward 编排各段，定位非 MIMO 的 ~36ms/epoch
    struct BwSeg {
        enum { POP=0, GET=1, NBWD=2, MIMO=3, DEC=4, ADD=5, PUSH=6, CLEAR=7, NUM=8 };
        static std::atomic<uint64_t>& N(int i){ static std::atomic<uint64_t> v[NUM]; return v[i]; }
        static std::atomic<uint64_t>& C(int i){ static std::atomic<uint64_t> v[NUM]; return v[i]; }
        static bool on(){ static bool e=[](){ auto* p = std::getenv("C3_BW_SEG"); return p && *p=='1'; }(); return e; }
    };
    struct BwSegGuard {
        int s; std::chrono::steady_clock::time_point t0; bool on_;
        BwSegGuard(int i): s(i), on_(BwSeg::on()) { if (on_) t0 = std::chrono::steady_clock::now(); }
        ~BwSegGuard(){
            if (!on_) return;
            uint64_t d = (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - t0).count();
            BwSeg::N(s).fetch_add(d, std::memory_order_relaxed);
            BwSeg::C(s).fetch_add(1, std::memory_order_relaxed);
            static thread_local size_t acc = 0;
            if ((++acc) % 20 == 0) {
                const char* nm[8] = {"pop","get","nbwd","mimo","dec","add","push","clear"};
                fprintf(stderr, "[BW-SEG]");
                for (int i = 0; i < 8; ++i) {
                    uint64_t n = BwSeg::N(i).load(), c = BwSeg::C(i).load();
                    fprintf(stderr, " %s=%.2fms/%llu", nm[i], n*1e-3, (unsigned long long)c);
                }
                fprintf(stderr, "\n");
            }
        }
    };

    GradBucket &bucket = GradBucket::getInstance();
    bucket.clear();

    const std::vector<size_t>& root_shape = root->getResultShape();
    auto root_result = root->getResult();
    DeviceType root_device = root_result ? root_result->device() : DeviceType::kCPU;
    
    if (root_shape.empty() || (root_shape.size() == 1 && root_shape[0] == 1)) {
        Tensor grad_tensor(1.0f, root_device);
        GradPack primary = {root, std::vector({grad_tensor}), -1};
        bucket.add(std::vector({std::move(primary)}));
    } else {
        Tensor cpu_tensor(ShapeTag{}, root_shape, DType::kFloat, DeviceType::kCPU, false);
        float* p = cpu_tensor.data_write<float>();
        std::fill(p, p + cpu_tensor.numel(), 1.0f);
        Tensor grad_tensor = cpu_tensor.to(root_device);
        GradPack primary = {root, std::vector({grad_tensor}), -1};
        CTORCH_TRACE(ErrorPlatform::kAutoDiff, "ComputeCore::backward - grad_tensor dim: " + std::to_string(grad_tensor.dim()));
        CTORCH_TRACE(ErrorPlatform::kAutoDiff, "ComputeCore::backward - Adding primary grad pack");
        bucket.add(std::vector({std::move(primary)}));
    }

    CTORCH_TRACE(ErrorPlatform::kAutoDiff, "ComputeCore::backward - Adding root to ready nodes");
    // [Fix 2026-09-06 苏璃珞] 依赖计数只计从 root 反向可达的活跃子图。
    // 根因: 图里存在"死分支"时(如 sum-loss 训练里仍构造了无关的 CE 头 MatMul),
    // 死节点注册时对上游 increase() 使活跃节点的 _count(下游完成倒计时)永远多 1,
    // 死节点永不处理 → 活跃节点永远不 ready → 所有叶子梯度静默为 null。
    // 修复: 从 root 反向 DFS 得活跃子图, 按"活跃下游数"重算每个节点的 _count,
    // 排除死分支的 increase。无死分支时重算结果与注册时一致, 行为不变。
    {
        // [perf 诊断] env CT_DISABLE_ACTIVE_RESET=1 关闭活跃子图重算, 用于 A/B 测 DFS 开销
        static const bool active_reset_enabled = [] {
            const char* e = std::getenv("CT_DISABLE_ACTIVE_RESET");
            return !(e && std::string(e) == "1");
        }();
        if (active_reset_enabled) {
        std::unordered_set<Node*> active;
        std::function<void(Node*)> dfs = [&](Node* n) {
            if (!n || active.count(n)) return;
            active.insert(n);
            for (const auto& up : n->getUpStreamNodes()) {
                if (up) dfs(up.get());
            }
        };
        dfs(root.get());
        for (Node* n : active) n->setCount(0);
        for (Node* n : active) {
            for (const auto& up : n->getUpStreamNodes()) {
                if (up && active.count(up.get())) {
                    up->setCount(up->getCount() + 1);
                }
            }
        }
        }
    }
    addReadyNode(root);

    CTORCH_TRACE(ErrorPlatform::kAutoDiff, "ComputeCore::backward - Processing ready nodes");
    while (true) {
        std::shared_ptr<Node> node;
        { BwSegGuard _g(BwSeg::POP); node = tryPopReadyNode(); }
        if (!node) {
            if (!bucket.empty()) {
                CTORCH_TRACE(ErrorPlatform::kAutoDiff, "ComputeCore::backward - Grad bucket not empty but no ready nodes");
            }
            break;
        }

        CTORCH_TRACE(ErrorPlatform::kAutoDiff, "ComputeCore::backward - Processing node");

        std::vector<Tensor> grads;
        { BwSegGuard _g(BwSeg::GET); bool ok = bucket.tryGetGrad(node, grads); if (!ok) { continue; } }

        CTORCH_TRACE(ErrorPlatform::kAutoDiff, "ComputeCore::backward - Processing node backward");

        // ===== C3 backward fusion 接线 (5d follow-up 完成, 2026-08-11) =====
        // 尝试用 C3 JIT 编译的反向 kernel 计算梯度;未命中/形状不匹配/禁用时回退 eager。
        // 安全护栏: 仅当 ① grad 恰好 1 个(聚合梯度) ② 有输入 ③ 上游节点数==输入数
        // ④ C3 返回梯度数==输入数 且 每个梯度 shape 与对应输入 shape 一致, 才接受。
        // 任一不满足 → result 保持空 → 走 eager node->backward(grads)。
        std::vector<GradPack> result;
#ifndef CT_DISABLE_C3
        if (grads.size() == 1 && !node->getInputs().empty()) {
            const auto& fwd_inputs = node->getInputs();
            auto upstream = node->getUpStreamNodes();
            if (upstream.size() == fwd_inputs.size()) {
                // 记录反向节点序列（累积频次触发反向融合异步编译）。
                // 内部仅登记 supportsNodeType 的 element-wise 节点；多输入节点安全跳过。
                // 必须在 tryExecuteBackward 之前调用，使序列频次在 execute 前已累计。
                ct::c3::C3BackwardCapture::getInstance().recordBackwardNode(
                    typeid(*node).name(),
                    grads[0].sizes(),
                    fwd_inputs[0].sizes(),
                    fwd_inputs);
                std::optional<std::vector<Tensor>> c3_result_box;
            { BwSegGuard _g(BwSeg::MIMO);
              c3_result_box = ct::c3::C3BackwardCapture::getInstance().tryExecuteBackward(
                    node.get(), grads[0], fwd_inputs);
            }
            if (c3_result_box.has_value() && c3_result_box->size() == fwd_inputs.size()) {
                    bool shape_ok = true;
                    for (size_t i = 0; i < c3_result_box->size(); ++i) {
                        if ((*c3_result_box)[i].sizes() != fwd_inputs[i].sizes()) {
                            shape_ok = false;
                            break;
                        }
                    }
                    if (shape_ok) {
                        result.reserve(c3_result_box->size());
                        for (size_t i = 0; i < c3_result_box->size(); ++i) {
                            result.push_back(GradPack{
                                upstream[i], {(*c3_result_box)[i]}, static_cast<int>(i)});
                        }
#ifdef CT_DEBUG
                        std::cerr << "[DBG-C3BW-WIRE] node=" << typeid(*node).name()
                                  << " C3-HIT n_grads=" << c3_result_box->size() << std::endl;
                        std::cerr.flush();
#endif
                    }
#ifdef CT_DEBUG
                    else {
                        std::cerr << "[DBG-C3BW-WIRE] node=" << typeid(*node).name()
                                  << " C3-SHAPE-MISMATCH fallback eager" << std::endl;
                        std::cerr.flush();
                    }
#endif
                }
            }
        }
#endif
        if (result.empty()) {
            { BwSegGuard _g(BwSeg::NBWD);
              // [BW-NBWD 2026-08-27] env C3_BW_SEG=1：按 node 类型分桶 eager backward 耗时
              if (BwSeg::on()) {
                  auto tm0 = std::chrono::steady_clock::now();
                  result = node->backward(grads);
                  uint64_t d = (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
                      std::chrono::steady_clock::now() - tm0).count();
                  static thread_local std::unordered_map<std::string,uint64_t> nm, nc;
                  std::string key = typeid(*node).name();
                  nm[key] += d; nc[key]++;
                  static thread_local size_t acc = 0;
                  if ((++acc) % 20 == 0) {
                      fprintf(stderr, "[BW-NBWD]");
                      for (auto& kv : nm)
                          fprintf(stderr, " %s=%.2fms/%llu", kv.first.c_str(),
                                  kv.second*1e-3, (unsigned long long)nc[kv.first]);
                      fprintf(stderr, "\n");
                  }
              } else {
                  result = node->backward(grads);
              }
            }
        }

        if (!result.empty()) {
            CTORCH_TRACE(ErrorPlatform::kAutoDiff, "ComputeCore::backward - Adding " + std::to_string(result.size()) + " grad packs");

            std::vector<std::shared_ptr<Node>> ready_nodes;
            { BwSegGuard _g(BwSeg::DEC);
              for (const auto &pack : result) {
                if (pack._targetNode && pack._targetNode->decrease()) {
                    ready_nodes.push_back(pack._targetNode);
                }
              }
            }

            { BwSegGuard _g(BwSeg::ADD); bucket.add(std::move(result)); }

            for (const auto &node : ready_nodes) {
                CTORCH_TRACE(ErrorPlatform::kAutoDiff, "ComputeCore::backward - Adding upstream node to ready queue");
                { BwSegGuard _g(BwSeg::PUSH); addReadyNode(node); }
            }
        }

        if (!retainGraph) {
            bucket.remove(node);
        }
    }

    CTORCH_TRACE(ErrorPlatform::kAutoDiff, "ComputeCore::backward - Done processing ready nodes");
    
    if (retainGraph) {
        std::unordered_set<Node *> restored;
        { BwSegGuard _g(BwSeg::CLEAR); root->restoreRecursive(restored); }
    } else {
        std::unordered_set<Node *> cleared;
        { BwSegGuard _g(BwSeg::CLEAR); root->clearRecursive(cleared); }
        root.reset();
    }

#ifndef CT_DISABLE_C3
    // [优化 2026-08-16] 每次反向传播结束时，清空未消费的反向融合结果与 miss 标记。
    // 防止随着训练批次（Batch）演进，节点地址重用时误匹配到 Stale Tensor 或悬垂指针导致 SIGSEGV/UAF 崩溃。
    ct::c3::C3BackwardCapture::getInstance().clearCallScopedState();
#endif

    // EnableGrad 由 eg_guard 析构恢复(RAII, §4.95 P2)

    // [Fix 2026-09-10 §4.95 P1-13] 非保留图模式下, 本轮计算图已被 clearRecursive 释放,
    // 此时 reset Arena 安全(运行节点析构器 + block offset 归零复用, 内存有界)。
    // 保留图模式下不 reset(图仍被引用, reset 会复用其节点内存 → UAF)。
    // 逃生开关 CT_ARENA_NO_RESET=1 可关闭(特殊场景: 外部持有 Arena 分配对象跨轮使用)。
    if (!retainGraph && std::getenv("CT_ARENA_NO_RESET") == nullptr) {
        Arena::getInstance().reset();
    }
}