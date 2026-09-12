# ADR-012: 手写 MIMO pattern 退场采用通用链式识别器（阶段二设计）

> **状态**：Accepted（阶段二 v1 已实施并影子验证；默认切换待 v1 浸泡后）
> **日期**：2026-09-12
> **作者**：CTorch Agent（苏璃珞）
> **决策者**：CTorch Agent + 用户（寒假前清单 ④ 授权）
> **关联**：STATUS §4.103（阶段一影子对照）、c3 `c50c796`、`c3/C3BackwardCapture.cpp`
> **替代方案**：A 直接退场（删除手写路径）、C 识别器保留+仅泛化图构建

## 1. 背景

手写 MIMO pattern（FC：Activation→Add→MatMul；FFN：无 bias SwiGLU）承担两件事：

1. **前置识别器**：typeid 名称匹配识别反向图结构，决定进入 MIMO 编译路径；
2. **手写图构建**：按已知拓扑手工构建反向子图 + `MergeSpec` 缝合 + 手写执行段记账。

G3 通用接管（FusionPlanner + partitionGraph + OrchestratedKernel，§4.84-4.88 已默认开启）目前**只能被手写识别器触发**——它挂在 MIMO 编译路径内部。阶段一影子对照（§4.103）证明：

- `C3_MIMO_LEGACY=0`（手写执行段整体短路）时 MNIST 逐位一致、FFN step0 逐位一致；
- 性能代价 ≈1%（噪声级）。

但**直接退场会把 planner/G3 整条通用融合线打成死代码**（G3 无其他 on-path 触发点），与「通用图融合取代手写 pattern」的架构方向冲突。故阶段二必须提供通用识别器作为 G3 的 on-path 触发点。

## 2. 决策目标

在不改变默认行为、不牺牲正确性的前提下，用**通用链式识别器**替换手写识别器 + 手写图构建，使手写 pattern 可整体退场；最终默认切换 `C3_MIMO_LEGACY=0`。

## 3. 候选方案

| 方案 | 描述 | 改动量 | 风险 |
|------|------|--------|------|
| A 直接退场 | 默认 legacy=0，删除手写代码；反向全走逐节点路径 | 小 | planner/G3 变死代码；融合投资废弃；跨硬件（高 launch 税）无退路 |
| **B 通用链式识别器（选定）** | 线性链捕获：真实拓扑走链 → 通用逐节点反向构建器 + 拓扑缝合 → planner/G3 接管 → 通用 pending 记账 | 中（~500 行 + 影子验证） | 恒等子图雷区（历史 worker bug）；G3 分区与 legacy 图结构差异的数值等价需实证 |
| C 识别器保留+仅泛化构建 | typeid 识别保留，仅替换手写缝合 | 中 | 手写 pattern 未真正退场；两套识别逻辑长期并存 |

## 4. 评估矩阵

| 维度 | 权重 | A | B | C |
|------|------|---|---|---|
| 正确性（默认路径零变化） | 高 | ✓ | ✓（off-by-default 开关） | ✓ |
| 架构方向一致性（G3 保持可达） | 高 | ✗ | ✓ | △ |
| 可维护性（手写代码消除） | 中 | ✓ | ✓ | ✗ |
| 实现成本 | 中 | 低 | 中 | 中 |
| 回滚成本 | 低 | 低（删除可 revert） | 低（env 开关） | 低 |
| 综合 | - | ✗ | **✓** | △ |

## 5. 最终决策：方案 B，分两步

**B-v1（2026-09-12 完成, c3 9401c6f）**：线性链通用捕获，范围 = `{ReLU, Add, MatMul}` 白名单、firing 节点单输入、链必含 MatMul（层边界）、链长 ≤5。`C3_MIMO_GENERIC=1` 启用，位于手写识别器**之前**（miss 即透传），默认关闭 → 默认行为零变化。MNIST generic=1 与 generic=1+legacy=0 均与基线逐位一致 → FC 手写路径可退场。

**B-v2（2026-09-12 完成, c3 e5f07d5）**：树拓扑泛化，firing 放宽为 ReLU/MatMul，白名单加 SiLU/Mul（FFN SwiGLU），BFS 走树（除 firing 外 MatMul 为叶子=层边界），树节点数 ≤8。FFN generic=1 与 generic=1+legacy=0 的 step0 loss 1390.0156 逐位一致 → FFN 手写路径可退场。**④ 完整收口：FC+FFN 手写 pattern 均已被通用树式识别器等价覆盖。**

**B-v3（后续）**：Tanh/Sigmoid 入白名单（依赖 tanh 反向图执行层专项）→ 浸泡（多轮 shadow + 跨测试矩阵）→ 默认切换 legacy=0。

### 关键设计决策（已实施）

1. **恒等子图 → 别名槽**：Add 输入形状 == grad 形状时其梯度是恒等透传。历史 worker 对「无算力节点」图有跳过 bug（`C3BackwardCapture.cpp:243`），故**不建恒等子图**，改为别名共享父边 grad 输出槽 —— 与 legacy FC「grad_z 直连 mm_w/mm_x/add_b」结构一致。
2. **编译线程只传值**：树结构（类型序列 + grad desc + 各输入 desc + 父子关系）同步提取、按值传编译线程，不引用 Node*（沿用 MIMO pending 生命周期模式）。
3. **fwd 张量计划缓存**：子图 fwd_input_map 由构建器确定性给出；编译线程算出「外部张量顺序计划」并缓存（key→计划），install 前写入 → 执行侧命中时计划必在，无竞态。
4. **G3 复用**：merged 图直接走 `computeMimoPartition` + `tryG3TakeoverKernel`，与 legacy 同一口径。
5. **firing 多输入单图双输出**（v2 实测教训）：MatMul 双输入若拆两个子图，各声明一个 grad 外部输入，GraphMerger 不去重 → 外部输入数 +1，与 registry「[grad]+fwd」喂入约定冲突 → 改为单图双输出子图（与 legacy buildMMDual 同构），槽位模型泛化为全局输出槽（每子图可多输出）。

## 6. 反事实分析 (CFC)

- **若半年前实施**：可避免 MIMO/FFN 两套手写编译路径的持续膨胀（约 1200 行），但当时 planner/G3 尚未落地，通用识别器无下游 → 时机不成立，现在正是 G3 成熟后的正确窗口。
- **若撤回**：`C3_MIMO_GENERIC=0`（默认）即完全回到现状，零残留。
- **新隐藏假设**：`getUpStreamNodes()` 与 `getInputs()` 索引对齐（DataCore::registerNode 按输入序 push，已验证成立）；Add 恒等别名不改变 pending 语义（pending 槽位仍按「每节点每输入」拆分）。
- **时序风险**：编译线程与执行线程共享 plan 缓存 → 用「先写计划、后 install kernel」的偏序消除竞态；pending 表沿用 shared_mutex 模式。

## 7. 实施步骤

1. `C3Config.h`：`mimoGenericEnabled()`（默认 false）。
2. `C3BackwardCapture.h/.cpp`：`GenericChainSpec` + `tryExecuteGenericChainMIMO` + `compileGenericChainMIMOAsync` + plan 缓存成员。
3. 钩子：`tryExecuteUnifiedMIMOBackward` 在 legacy 短路**之前**试通用路径。
4. 影子验证：MNIST `C3_MIMO_GENERIC=1` vs 基线逐位一致；FFN step0 不变量；全量回归。
5. 浸泡后（v2 就绪）默认切换。

## 8. 验证结果

- **B-v1（9401c6f）**：MNIST generic=1 轨迹与基线全等（loss 0.5436/0.2167/0.1552/0.1210/0.0985, acc 85.12/93.69/95.51/96.50/97.14）；generic=1+legacy=0 纯通用路径同结果；FFN step0 不变量；回归 test_c3_graph 118 / test_c3_backward max_diff=0 / test_sum_mean_grad ALL PASS。
- **B-v2（e5f07d5）**：FFN generic=1 step0 loss 1390.0156 逐位一致；generic=1+legacy=0 纯通用树路径同结果（G3 切 2 子图编排接管）；MNIST 两种模式均 0.0985/97.1421%；回归 118/0/ALL PASS。详见 STATUS §4.104/§4.105。

## 9. 对抗思考 (ADV)

- **更简单方案**：方案 A 直接退场确实更简单，但杀死 G3 可达性，收益 80% 时留下架构负资产 → 否。
- **过度设计**：v1 只做线性链、白名单 3 类节点；v2 只扩到 FFN 树拓扑 + 2 类节点，不碰任意 DAG → 否。
- **同步调整**：tanh 执行层专项（修复后 Tanh/Sigmoid 才能进白名单）→ 已列入待办；默认切换 legacy=0 待浸泡。
- **失败先例**：DEBT-2（旧 fused backward）死于 intercepted 节点生命周期；本方案用 MIMO 已验证的 pending_mimo_intercepted_ + 单消费者守卫规避同根因。

## 10. 状态

Accepted（v1+v2 已实施并影子验证；默认切换待浸泡 + B-v3 白名单扩展）。
