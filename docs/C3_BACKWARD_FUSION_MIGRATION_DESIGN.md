# C3 Backward 运行时融合迁移：决策门设计 (Backward Fusion Migration Gate)

> **状态**: 设计草案 (2026-09-07, 洛锦/苏璃珞)
> **性质**: 设计文档, 先批决策门, **暂不改运行时**。文档配套 G1 reconcile 诊断(off-path, 默认关)。
> **引用**: 承接 `C3_UNIVERSAL_FUSION_DESIGN.md`(判据/代价门/捕获层已就绪);
>         红线见 `AGENTS.md`(尤其 `in_autograd` 短路)。

## 1. 为什么需要一个"决策门"而不是直接替换

backward 的 MIMO 目录(FFN/FC)正在生产路径上稳定生效(FFN bwd MIMO 命中, ~8% 快)。
"通用化"的目标是**用 planner 判定取代手写结构匹配**, 但这一步会动训练正确性核心:
一旦 planner 判错"该把哪几个上游节点并入单内核", 梯度回填/多消费者守卫就会错,
直接破 parity(曾出过 `C3-BUG-20260905-01` 同根因)。所以**替换必须经过一个可量化的决策门**,
而不是"设计看着对就切"。决策门 = 让 planner 的判定先与现状 MIMO 逐例对拍, 一致率达标才放行。

## 2. 拆解: backward 融合到底由哪几件事组成

对一个被 ComputeCore 触发 backward 的 autograd 节点 N:

| 部件 | 做什么 | 现状 | 通用化程度 |
|---|---|---|---|
| A | 为 N 生成其 backward 的 C3 子图 | `buildBackwardGraph`(逐节点, 已存在) | 已通用 |
| B | **决定"向上游吞多少"**(把哪些上游节点的 backward 并入一次算) | MIMO 目录**手写**: FFN/FC 各一条 typeid 结构匹配 | **不通用**(研究级难点) |
| C | 判定并出来的子图是否值得当单内核 | 目录直接编译(不判代价) | FusionPlanner(RegionKernel+代价门)已就绪 |
| D | codegen + 执行 + 梯度回填 | `compile` + `tryExecuteBackward`(已存在, 含单消费者守卫) | 已通用 |

**关键认知**: 通用化最难的是 **B**(上游扩展范围), 不是 C 或 D。
MIMO 目录的存在价值恰恰在 B——它用结构匹配编码了"FFN 这种 SwiGLU 结构, backward 该一次吞 4 个上游节点"。
纯粹靠 planner 在 C3 Graph 上判连通/代价, **并不能自发决定"吞多少上游"**——
因为 planner 看到的是一个已扩展好的 `fused_graph`(34 节点 9 输出), 它只决定"这些并成几个内核"。

## 3. 决策门分层 (G0 → G3), 逐级放行

```
G0 现状        MIMO 手写目录 + 单节点 backward 共存(生产)
  │  每次 MIMO 命中, planner 对真实 fused_graph 跑 RegionKernel, 记 reconciled?/一致率
G1 校验(本次)  planner 判定 vs MIMO 实际范围 逐例对拍 → 输出一致率/差异原因(只读, off-path)
  │  一致率 & 差异可解释后
G2 影子        不一致时仅告警/记录, 仍走 MIMO(不改行为); 积累"planner 会错/不会错"的证据
  │  planner 判定在覆盖结构上 100% 复现 MIMO 后
G3 接管        以 planner+代价门为唯一判定; MIMO 目录降为 fallback(结构兜底)
```

**放行条件(硬)**
- G1: MNIST + FFN 每次 MIMO 命中, `[BW-RECONCILE]` 一致率 ≥ 阈值(建议初始 100%, 即 planner 判定
  恒与 MIMO 范围一致; 不一致必须可解释——例如 FFN 当前 mismatch 是因 launch 税低估)。
- G3: 覆盖结构全部 reconciled, 且单测/端到端 max_diff=0, 无 fallback 次数上升。

## 4. 已知的不一致: 为什么 planner 现在 ≠ MIMO(FFN)

FFN 真实 fused_graph 对拍(`C3_PLANNER_DIAG`):
```
[BW-RECONCILE] mimo_kernels=1 planner_wants=2 reconciled=0
region_metric[comp=2 reload=512KB launch=400KB ws=134MB merged=0]
```
MIMO 发 1 个内核; planner 保守代价门判 2 个 region。差异**可解释**:
launch 税(现 400KB 等价)低估了真实多输出 dispatch 开销, 且 ws=134MB 的峰值 live 偏大,
使代价门不敢并。**这不是 planner 判据错, 是代价门参数没校准**——校准即需 autotune 实测
launch 税(设计 `C3_DEPLOY_AUTOTUNE_DESIGN.md` 已列该校准域)。所以 G1→G3 的前置之一是
**先做 autotune launch 探针**, 否则 G1 一致率恒不达标。

## 5. 风险 / 边界 / 红线

- **B(吞上游)不通用**: 本方案不假装解决"任意结构自动吞上游"。它把 B 仍交由目录(启发式),
  但用 planner(RegionKernel)+代价门 **校验**目录的"该不该并成一个内核"这一步(C/D)。
  真·让 B 也通用是需要"每节点逐级扩展+代价择优"的研究级重写, 单独立项。
- **红线**: 不碰 `C3HotPathManager.h:236-240` `in_autograd` 短路; 不新增 op;
  不改单节点 backward 的正确性路径。
- **回退**: G2/G3 全程保留 env 回退(如 `C3_PLANNER_DIAG` 关闭即原行为);
  任何不一致 → 记日志并走 MIMO/单节点, 不阻塞训练。

## 6. 落地物(本次已交付)
- `docs/C3_UNIVERSAL_FUSION_DESIGN.md` §9 记录判据/代价门/捕获层实现进度与实测。
- 诊断 reconcile 行 `[BW-RECONCILE]`(FFN MIMO 编译处, `C3_PLANNER_DIAG=1`, off-path):
  planner 打算发几个 region kernel vs MIMO 现发 1 个, reconciled?/差异原因。
- 本文件: G0-G3 决策门 + 放行条件。

## 7. 建议的下一步(需洛锦排)
1. **autotune launch 探针**(最优先): 让代价门 launch 项用机器实测, FFN reconcile 才有望转 1,
   否则 G1 一致率恒不达标。
2. **G1 覆盖率扩到 FC-MIMO + 稳态统计**: 在 compileUnifiedMIMOBackwardAsync 也挂 reconcile,
   跑 MNIST/FFN 采集一致率矩阵, 给洛锦看数据再决定是否进 G2。
