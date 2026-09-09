# C3 通用图融合设计 (Universal Graph Fusion)

> **状态**: 设计草案 / scope (2026-09-07, 洛锦/苏璃珞)
> **性质**: 设计文档, 暂不实现。先批 scope + 决策门, 通过后再逐段落地。
> **目标**: 把 C3 现在"为每种图结构手写一个融合 pass"的能力, 收敛成
> "编译器在统一 IR 上**自动判定可融合子图**并**统一 codegen**" 的通用机制,
> 兑现「跨平台可用 / 能力泛化 / 不针对测试写 pass / 优化极致」的硬标准。
> **引用**: 与 `C3_DEPLOY_AUTOTUNE_DESIGN.md`(硬件决策自适应)同源互补;
> 本文解决"融合什么/怎么判", autotune 解决"该不该(按机器)融合"。

---

## 1. 现状: 三套并存的融合, 都还不是"通用"

C3 目前有**三套**融合机制, 每套都各自生效、各自有手写白名单/线性假设:

| 机制 | 触发面 | 判定方式 | 代码位置 | 泛化短板 |
|---|---|---|---|---|
| **A. Forward 运行时 region fusion** | 热路径 dispatch 流 | **手写 if-else** 匹配最近 3~2 个 `DispatchRecord` 的线性序列(MatMul+Add+{Sig/ReLU/SiLU}, MatMul+{Sig/ReLU/SiLU}); 还带"在窗口里找 bias Add 升级模式"的专项逻辑 | `C3HotPathManager.h:536-690` (`checkPattern`) + `RegionFusionRegistry`(RollingHash 按 `op_seq`) | 只认写过的线性链; 用**执行顺序**近似拓扑,**无真实 DAG**; 扇出/分支/重排必然漏; 结构一换就不认 |
| **B. 反向逐节点 + MIMO 目录** | backward autograd 节点 | 每节点类型有反向子图映射表; 另有**手写 MIMO 目录**: `compileFFNMIMOBackwardAsync` / unified MIMO (FC/FFN 整段反向 → 单内核多输出) | `C3BackwardCapture.*` | 新结构要**手写新 MIMO pass**; 输入角色按 operand 顺序推断 |
| **C. Graph 层融合 + GraphMerger** | 编译期整图 | `compile()` 先 canonicalize→eliminateDeadCode→fuse, 再走 `generateFromGraphMLIR`(单节点/多节点/FusedNode); `GraphMerger` 把子图**按外部给的顺序链接**并成一个图 | `C3Engine.cpp` / `GraphMerger.*` / `LinalgFusedGen.*` / `LinalgOneShotGen.*` | 最接近 DAG 语义, 但 `GraphMerger` 只做"装配"——**拓扑链接是外面喂的**, 它自己**不去发现**该不该融合 |

关键观察: **`Graph` 已是三套共同的中间 IR**(forward 经 `buildFusedGraph` 构造, backward 经 capture 构造, GraphMerger/OneShot 直接吃 Graph)。差异不在"用什么表示图", 而在"**谁来、按什么判据、把哪些节点并成一个 kernel**"。这三段判据各写各的 → 三份白名单漂移, 才叫"不通用"。

### 为什么"再补一个 pattern"不是出路

我们上一轮修的恰好是两个"判据漂移"实例:
- `buildGt` 只读 `rhs[0]` → 因为假定 Gt 的 rhs 恒标量;
- Linalg OneShot 的 `isBroadcastableTo` 放行短输入 → 因为假定 ABI 能处理广播而实际是 identity 1D。

这两处都证明:**每多写一个特判, 就多一个与真实 shape/numel/lowering 能力脱节的点**。
"通用"的定义应当是——**融合判定只由 (a) lowering 能力、(b) shape/numel 传播、(c) 数据依赖 割、(d) 代价 决定, 而非某个 op 名或某种结构**。

---

## 2. 核心概念: 把"手写 pattern"换成"融合判定函数"

新增一个决策层 **FusionPlanner**(编译期纯函数), 输入一个 `Graph`,
输出一组**可独立编译/执行的融合单元**(每个 = 一个连通子 DAG + 边界张量集),
同一份输出被 forward / backward / 整图三段共用。

```
                    ┌─────────────────────────────────────────────┐
  Graph(统一 IR) ──▶ │  FusionPlanner                              │
                    │  (1) 传播 shape/numel/live-rank (一次性)      │
                    │  (2) 找"必须物化"的割点(不能并的边界)          │
                    │  (3) 对每个连通候选: lowering 能力 ⊇ 算子集 ?  │
                    │  (4) 代价门(调 autotune 指纹)决定并 or 不并    │
                    └───────────────┬─────────────────────────────┘
                                    ▼
              融合单元[0..k] ──每个─▶ 统一 codegen (现有 MLIR 多节点 /
                                    FusedNode / MIMO 内核), 不再按结构特判
```

**"能并" = 三条件同时成立:**
1. **Lowering 兼容**: 单元内每个算子都存在到目标 kernel 的统一降级(逐元素→vector/arith,
   GEMM→cblas/SIMD+epilogue 内联, MIMO 反向→多输出内核)。兼容性是一张 **op→能力** 表,
   不是"某三段链成立";
2. **Shape/numel 安全**: 边界与中间张量在目标 ABI 下无越界(这正是我们刚修的 Gt / Linalg
   两处判据的**通用化**——在 planner 里做 shape 传播与广播/标量归类, 而不是每个 codegen 自己猜);
3. **物化边界正确**: 只有 **多消费者 / 参数边界 / 不可降级算子 / 需要独立调度的点** 才割开;
   单消费者、共享 live-range 的中间张量一律内联(这是 MIMO 单消费者守卫、buffer-reuse 守卫
   等的统一来源, 不再散落各 codegen)。

**"该不该并" = 交给代价门**(而非写死): 单位读写次数节省 / launch 税, 复用以
autotune 机器指纹校准后的阈值(§关联)。收益不足就不并——与现有 `FusionCostModel` 对齐。

---

## 3. 取代对象与取舍(逐条)

> 目标不是"推倒重写", 而是让三段共用一个 planner 的输出。MVP 先让 planner 的判据
> 能**复现**现有命中, 再反客为主。

| # | 取代 | 现在 | 之后 | 兼容策略 |
|---|---|---|---|---|
| 3.1 | Forward `checkPattern` if-else | 手写 6 段线性链 | planner 在**真实 DAG**上按 §2 判; 同一判据自然覆盖线性链, 也覆盖扇出/重排 | MVP 用 planner 复现现有命中数(`fused_hit`)做**回归门槛**, 相等才替换 |
| 3.2 | 逐节点反向白名单 | 每节点类型写 backward 图 | 反向本就是 Graph, planner 对每个 backward 段判融合; MIMO 变成"多输出 codegen 能力"而非"FFN/FC 专有 pass" | 保留单输出 fallback, 保证逐输入梯度正确性不回退 |
| 3.3 | `GraphMerger` 被动装配 | 外部喂 `MergeSpec` 顺序链 | `mergeSequential`/`merge` 可改由 planner 产出的单元直接构成(发现即装配) | `GraphMerger::validate` 的 shape/环校验**保留复用**作 planner 产物的后置校验 |
| 3.4 | 各 codegen 自带的广播/标量特判 | `buildGt` 猜 rhs、OneShot 猜 numel | shape/numel 传播集中在 planner 一次完成, 附到融合单元元数据; codegen 只消费**已判好的** `scalar?/broadcast?/numel` | 逐步把散落的 `numel==1` 归类挪进传播层 |

**明确不改/不动的红线**(保持 AGENTS.md):
- `C3HotPathManager.h:236-240` `in_autograd` 短路(训练一致性核心)。
- op 枚举 `Ctools.h` / `CtorchScheduler.h:229` static_assert——通用化**不新增 op**, 只统一判据。
- 不针对任何 benchmark/网络写"长这样才并"的规则; 验收正是"去掉手写仍覆盖 MNIST+FFN"。

---

## 4. FusionPlanner 的判据细节(scope 内的设计要点)

**(1) shape/numel/live-rank 传播(一次性 DFS)**
- 对每个节点算: out_shape、out_numel、in_numel、是否 size-1 维(真广播)、live 消费者数。
- 产出一张 `FusionMeta[node]` 表, codegen 阶段**只读不重判**。
- 直接吸收我们刚修的判据: Gt 的 `rhs_scalar = (rhs_numel==1)`; OneShot 门 = `所有输入 numel == 输出 numel` 才算 identity-1D 可并; 真广播(有 size-1 维且 numel 不等)走**专用广播 codegen** 或**不并入**该单元。

**(2) 依赖割(找必须物化的点)**
- 扫描 DAG, 割点 = (i) 图输入/参数边界; (ii) **出度>1** 的中间(多消费者必须落 buffer, 除非 MIMO 能一次产出——MIMO 单消费者守卫在此统一为"该中间是单元内单消费则可内联, 否则割"); (iii) 不在 lowering 能力内的算子; (iv) shape 语义断层(如 reshape/transpose 折返未验证前视为边界)。
- 割出来的连通区是**候选融合单元**; 单元间不再隐式共享 live-range。

**(3) 代价门(该不该)**
- 对每个候选: 内联省掉的中间写读 + launch vs 引入的寄存器/局部压力。
- 阈值来源 = autotune 机器指纹(GEMM/向量决策同理), 不写死常数。
- 关闭开关保留: env `C3_FUSION_PLANNER=0` 回退旧路径(与 `C3_DISABLE_*` 系一致), 保证任何回归可隔离。

**codegen 统一**: 融合单元 → 现有 `generateFromGraphMLIR` 的 FusedNode/多节点/MIMO 三条下放路径。MVP 不新写 codegen, 只把"并成什么单元"交给 planner; 单元结构仍是现有 codegen 认得的子图, 确保数值零改动。

---

## 5. 分期 (每期独立可交付、可回归、可回滚)

### L1 · 判据层 + Forward DAG 复现(MVP)
- 实现 `FusionPlanner` 判据(§4 的 shape 传播 + 割 + 能力表)。
- 对 forward:**先用 planner 在真实捕获 DAG 上"事后判定"**, 与 `checkPattern` 现有命中对比。
  - **验收门槛**: MNIST + LLaMA FFN 的 `fused_hit` 数**不小于**现状; 数值 diff=0。
  - 通过后, 才让 planner 的产出接管 forward 提交路径, 删除 `checkPattern` 的 if-else 段。
- 新增判据型回归(非测试特化): 任意随机构造的连通逐元素链 + 单层 GEMM+激活,
  planner 应自动并(证明**不是**认名字)。

### L2 · 反向 MIMO 目录统一
- 把 FC-MIMO / FFN-MIMO 从"专有入口"改成"planner 判定出的多输出单元 + 现有多输出 codegen"。
- MIMO 命中不再依赖"恰好识别 FFN", 而是依赖"这是可多输出的连通反向单元"。
- **验收**: FFN `mimo_hit` 不降; backward `max_diff=0`。

### L3 · 整图分区(接 GraphMerger)
- 把 planner 输出的单元集喂 `GraphMerger` 做整网络一次装配/一次编译(替代逐层)。
- 全图编译命中、热替换语义沿用现路径。

> L1 独立成里程碑; L2/L3 视 L1 结论再排, 避免一次性推倒。

---

## 6. 判定通过的输入(给洛锦批 scope 用)

1. 通用化**边界**: 只统一"判据", 不重写 codegen; MIMO/多输出 kernel 仍是现有多输出 codegen。
2. L1 MVP 用 **forward + 事后判定复现** 收敛风险, 命中回归不降才替换 `checkPattern`。
3. **删除手写 if-else / 专有 MIMO 目录**是硬性验收(否则没兑现"泛化")。
4. 保留完整回退 env + 逐输入正确性 fallback(backward 单输出梯度不回退)。

---

## 7. 风险 / 待核实 / 红线

- **性能**: planner 判据若比现有 if-else 慢, 会伤热路径税(C3 fwd 单 kernel tax 已 ~0.7-2µs/op)。
  → L1 判据只跑在"热路径判定为候选"之后, 不进每次 dispatch 的 O(1) 短路; 位掩码短路保留。
- **数值**: planner 只改"并什么", 不碰 codegen 算术 → 数值应与现状一致; 仍以 max_diff=0 把关。
- **待核实**(写文档时未逐行确认, 落地前核对): `compile().fuse()` 在 Graph 层是否已把相邻
  region 并成 FusedNode; `generateFromGraphMLIR` 对 FusedNode 是否全内联展开; backward 图
  每节点是否都"天然可用 planner"(与 MIMO 共存时谁优先)。
- **红线**: 不新增 op; 不动 `in_autograd` 短路; 不针对 benchmark 结构写规则。

---

## 8. 关联文档

- `docs/C3_DEPLOY_AUTOTUNE_DESIGN.md` — 硬件决策自适应(本文代价门阈值来源)。
- `docs/C3_REGION_FUSION_SILU.md` / `docs/C3_FUSED_BACKWARD_DEBT2_*` — 现有融合演进史。
- 上一轮修复记录: `STATUS_CONTEXT.md` §4.60(buildGt + Linalg OneShot 广播, c3 0e8cdf1)
  —— 两处特判正是本设计要"集中到传播层"的判据实例。

---

## 9. 实现进度与实测 (2026-09-07)

### L1-1 已落地: FusionPlanner 判据层 (c3 83300d2 / main 0947e4b)
`FusionPlanner::planUnits(Graph)` → 融合单元集, 纯函数只读, 判据数据驱动(非按名特判):
逐元素等 numel 共内核; MatMul 吸收单消费者尾链 → GEMM_EPILOGUE; 多消费者 / 双 GEMM /
numel 不一致一律割。5 单测全绿(逐元素链 / GEMM epilogue / 扇出割 / numel 割 / 共享 GEMM 不并)。
未接管热路径。配套诊断 env `C3_PLANNER_DIAG=1`(真实 FFN fused_graph 上量化 planner 输出)。

### L2 关键实测: MIMO ≠ planner 默认模型 (诊断证据)
在 `compileFFNMIMOBackwardAsync` 真实 fused_graph(34 节点, 9 输出)上跑默认 planner 得到:

```
[PLANNER-DIAG] FFN-MIMO graph nodes=34 compute_units=9:
  [ELEM n=4 Neg Exp Add Div] [GEMM] [GEMM] [ELEM n=6 Mul Sub Mul Mul Add Mul]
  [ELEM n=1 Mul] [GEMM] [GEMM] [GEMM] [GEMM]
```

**结论**: MIMO 是「单内核多输出 region」——一次算共享中间量 grad_h/grad_g/grad_u/grad_gate_pre,
喂给 6 个 GEMM 分支 + 3 段逐元素链, 写 9 个 grad; 靠**共享中间量不落内存** + 省 8 次 launch 取胜。
而 planner 默认「多消费者中间量必物化割开 / 双 GEMM 不并」正好把该 region 切成 9 个单元——
两模型目标相反。故前向单 GEMM 单元策略**不能**直接搬到 backward。

### 下一步定义: region-kernel 策略 (backward 方向)
planner 需增一种单元策略 `REGION_KERNEL`: 对**连通** backward 区段(共享中间量仅在本区段内复用),
允许整个连通计算分量作**一个多输出内核**, 是否采纳由**代价门**决定:
- 收益 = 省掉的中间量落内存(Σ 各共享中间 numel) + 省掉的 (GEMM数+链数-1) 次 launch;
- 代价 = 单内核寄存器/调度压力; 阈值走 autotune 指纹, 不写死。
- 与现 MIMO 运行时的关系: 该策略是 MIMO 目录的**通用替代**; 因触碰训练正确性核心,
  落地需分两步: (1) 先在 planner 加 `REGION_KERNEL` 判据 + 单测(用真实 fused_graph 拓扑,
  非按名特判, 复现"连通→1 单元"), (2) 验证后经决策门再考虑接管 `compileFFNMIMOBackwardAsync`。

### RegionKernel 判据已落地 + 真实 FFN 实测 (c3 HEAD, STATUS 4.63)
`FusionPlanner::planUnits(Graph, FusionStrategy::RegionKernel)` 新增单内核多输出 region 判据:
对**连通 regionable 分量**(逐元素/MatMul/Transpose, 排除 SumReduce/Softmax/CrossEntropy/Fused/Const 硬边界)
并成 `REGION_KERNEL` 单元; 共享中间量内联、Transpose 折叠、允许多 GEMM(单内核顺序执行+多输出)。
配套 3 单测(共享中间量→1 region / transpose 并入 / 不相连→多 region), 默认策略 5 测仍绿。

**真实 FFN fused_graph 实测(诊断 C3_PLANNER_DIAG)**: `nodes=34 default_units=9 region_units=2`,
region 拆为 n=21 + n=2 两个连通分量。
**关键发现**: 纯 C3 **节点连通**判据在真实 FFN 上给出 2 个 region(而非 MIMO 的单内核)。
原因: FFN 两部分计算在节点图里只通过**共享外部输入**(grad/各 activation buffer)间接相连——
节点图无内边把它们联通。故要把跨分量再并成一个内核, 需额外判据「同一次 backward 调用共享外部
输入(尤其 grad 与同组 activation)」+ **代价门**决定, 而非纯节点连通定律。该判据设计留给下一步,
**不强凑 1 region**(否则即针对 MIMO 特判, 违背泛化)。

### 代价门雏形 + 真实 FFN 决策证据 (STATUS 4.64)
RegionKernel 增跨分量合并代价门: 度量 = 共享外部输入重读节省(reload) vs 工作集(ws)代理;
`merged = reload > min_benefit_ratio * ws`(默认 0.25, 系数终态由 autotune 指纹给出; launch 省税
不在原型中计入)。真实 FFN 实测: `comp=2 reload=524288 ws=290947072 merged=0` → 保守默认**保持
2 regions**。结论: MIMO 单内核的收益主体是 **launch 省税 + 内核内向量化**, 不是重读节省——
把 launch 项补进代价门(带机器指纹)是让 planner 泛化复现 MIMO 决策的最后一块, 而非再加结构特判。

### launch 项接入 + live 工作集修正 (STATUS 4.65)
代价门补全: `merged = has_shared_ext && (reload + launch) > ratio * ws`。
- launch 项 `saved_launch = (k-1)*launch_unit_bytes`, 仅在共享外部输入的分量间计入(同一 backward 调用分支);
  纯不相连分支不并(保连通性结论)。`launch_unit_bytes` 默认 400KB(约 2µs@200GB/s), 终态由 autotune 校准。
- working_set 修正为 live 中间量(graph 输出除外)。真实 FFN: `reload=512KB launch=400KB ws=142MB merged=0`。
  结论: 大 batch FFN 的中间 activation(如 128×11008 的 grad_g/u)在 GEMM 间本就要落地、融合也塞不进
  缓存, 故纯字节代价下保守不并——与 M3「大 batch GEMM 合并负收益」一致。
- **剩余代理误差**: ws 现为「所有中间量求和」而非「峰值 live」, 会高估; 下一步用峰值 live 精化,
  或直接交给 autotune 在目标机实测 launch 税与最优系数, 而非继续手调代理。

### ws 峰值 live 精化 (STATUS 4.66)
working_set 由「求和」精化为「峰值 live」(任意时刻同时存活中间量的最大 numel, graph 输出除外,
拓扑序用节点 id, 中间量 live 于 [m, last_use(m)])。真实 FFN: ws 142MB → 134MB(几乎不降),
因 grad_g/grad_u 等大中间量本就同时存活。**结论坐实**: BS=128 下单内核的收益不在中间量复用
(中间量巨大且同时 live), 而在 launch 省税; 字节代价模型(launch=400KB)低估了真实 launch 税,
这正是 autotune 机器指纹该实测的量。判据层至此完整且保守, 不再手调代理。

### forward 整图捕获层已落地(STATUS 4.67) —— 前置缺口补齐
`ForwardCapture::capture(rootTensor)` 从根输出沿 autograd 上游遍历, 翻译成 c3::Graph:
纯只读快照, 不碰 dispatch / in_autograd。语义: requires_grad 叶(GradAccumulator)=去重外部叶,
非 grad 常量=每次新叶, 计算中间量递归。支持 MatMul/Add/Sub/Mul/Div/Neg/ReLU/Sigmoid/Tanh/
Exp/Log/Softmax/CrossEntropy, 不支持类型明确报错。
集成测试: 真实 eager 前向 MatMul->ReLU 捕获成 4 节点/2 叶 Graph, planner 自动判 GEMM_EPILOGUE。
**意义**: 通用图融合现在有了 forward 真实整图输入——planner 可从根输出看到完整前向结构,
补上「forward 无整图捕获」这一架构前置。下一步即可让 planner 在这类 forward Graph 上做
region 判定, 与 backward/MIMO 同口径。
