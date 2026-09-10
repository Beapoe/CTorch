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
  恒与 MIMO 范围一致; 不一致必须可解释)。
  > 口径澄清(2026-09-10): G1 校验的是**结构等价性**(planner 的 region 划分范围 == MIMO 单内核范围),
  > 故应在**强制合并模式**(`C3_FORCE_REGION_MERGE=1`)下取一致率——代价门判"划不划算"是另一层,
  > 见 §4.2。
- G3: 覆盖结构全部 reconciled, 且单测/端到端 max_diff=0, 无 fallback 次数上升。

## 4. 已知的不一致: 为什么 planner 现在 ≠ MIMO(FFN)

FFN 真实 fused_graph 对拍(`C3_PLANNER_DIAG`):
```
[BW-RECONCILE] mimo_kernels=1 planner_wants=2 reconciled=0
region_metric[comp=2 reload=512KB launch=400KB ws=134MB merged=0]   # LLaMA-1B 真实维度
```

### 4.1 根因修正(2026-09-10, 实测证伪原归因)

> **原文归因("launch 税低估 → 校准后即可转 1")经实测证伪**, 详见
> `work/reports/2026-09-10/bw-reconcile-root-cause-diagnosis.md`。

实测三维度(`C3_PLANNER_DIAG=1`, off-path):

| BS×HID×INT | saved_reload | saved_launch | ws | 门槛 0.25×ws | merged |
|---|---|---|---|---|---|
| 8×16×32 | 128 | 409600 | 2560 | 640 | 1 |
| 64×256×512 | 16384 | 409600 | 524288 | 131072 | 1 |
| 128×1024×2048 | 131072 | 409600 | 7340032 | 1835008 | 0 |

**真正主因**: `ws`(峰值 live 中间量)随维度线性增长且绝对值远大于 `saved_reload`
(128×1024×2048 下差 56x), 而 `saved_launch` 是与维度无关的常数 →
维度越大 `0.25×ws` 门槛越高, `reload + launch` 越追不上 → **必然 merged=0**。
把 launch 税"校准"到机器实测 12KB 反而让差距**更大**(528KB→140KB vs 门槛 1792KB)。

结论: **launch 税校准对 FFN reconcile 无正贡献**, 不应作为 G1→G2 前置。

### 4.2 解耦方案: 强制合并(已落地 2026-09-10)

既然瓶颈在"收益模型"而非"结构划分", 把两个正交问题解耦:

- **结构正确性**(planner 划分 vs MIMO 范围是否等价) → 用**强制合并**验证, 跳过代价门
- **代价判定**(该不该并是否划算) → 作为**独立优化层后补**

实现: `RegionFusionPolicy::force_merge`(默认 false) + env `C3_FORCE_REGION_MERGE=1`。
强制模式下只要结构可并(`component_count > 1 && has_shared_ext`)就并, 不做收益判定;
`reload/launch/ws` 度量仍填充供观测。

实测(强制模式, 4 维度): **reconciled 全部 = 1**(planner_wants=1 == MIMO 单内核)。

**待补(登记的后续项)**: 代价判定需重新设计收益模型——当前
`saved_reload`(省外部输入重读) 低估了 MIMO 的真实收益(应为"省中间量物化");
但直接改成内联中间量会让 `saved ≥ ws` 恒真、判据退化为"总是合并",
需配独立第二约束(缓冲压力/region 节点数上限)。

### 4.2.1 代价门重设计: ADR-0002 方案 C (已落地 2026-09-10)

§4.2 的后续项已由 ADR-0002 决策并落地(见 `work/reports/2026-09-10/adr-0002-*.md`)。

**关键实验(EXP-1)**: 跨分量合并的收益 = 省重读 128KB(5.33µs) + 省 1 次 launch(0.49µs)
= **5.81µs**, 相对 FFN bwd 单步 4900µs 仅 **0.119%**。
→ 该层判别力价值不成比例; 无论判并或不并, 性能差异 < 0.2%。

**决策(方案 C)**: 跨分量**默认合并**, 判别力**下沉到"规模保护"**(防单内核代码膨胀/寄存器压力)。
- 替代方案 A(修正收益模型) 会因 `saved ≥ ws` 恒真而退化; 方案 B(ws 语义重分配) 缺"容量上限"依据。
- 实现: `RegionMergeStrategy{Strict, Allow}` + `RegionFusionPolicy::merge_strategy`(默认 Strict)
  + `max_region_nodes`(默认 64) + env `C3_REGION_MERGE_ALLOW=1`。
- 判定优先级: `force_merge` > `Allow`(结构 + 规模保护) > `Strict`(现行门槛)。

**实测**: Strict(默认) `reconciled=0`(与改动前逐位一致); Allow `reconciled=1`(与 MIMO 一致)。

**仍待做(ADR 步 3-5)**: ① G3 集成点(planner 判定参与执行, 带开关) ② A/B 实测(1 内核 vs 2 内核,
方案 C 最终依据) ③ `max_region_nodes` 实测标定。

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
1. ~~autotune launch 探针~~ → **证伪, 已移出前置**(见 §4.1): launch 税校准对 FFN reconcile 无正贡献。
   已改为**强制合并**(§4.2, 已落地)解耦结构验证与代价判定。
2. **G1 覆盖率扩到 FC-MIMO + 稳态统计** → ✅ **已完成(2026-09-10, STATUS §4.76/§4.77)**:
   抽取共用方法 `C3BackwardCapture::diagnosePlannerReconcile(fused_graph, label, mimo_kernels)`,
   在 `compileUnifiedMIMOBackwardAsync` 与 `compileFFNMIMOBackwardAsync` 两处挂载(纯 off-path)。
   实测一致率矩阵: **FC-MIMO 与 FFN-MIMO 的结构等价性均 100%**(FFN 大维度需强制模式绕过代价门)。
   稳态统计(§4.77): `reconcile_total/matched` 跨结构聚合 + `[G1-RATIO]`/`[*-G1-STAT]` 输出;
   默认路径零开销。
3. **G2 影子观测** → ✅ **已落地(2026-09-10, STATUS §4.78)**:
   `C3_PLANNER_SHADOW=1` 开启常态化影子对拍 —— **静默一致、仅不一致时告警**
   `[G2-SHADOW-MISMATCH]`、**绝不改行为**(真实执行仍走 MIMO 手写目录)。
   与 `C3_PLANNER_DIAG`(详细诊断) 独立, 二者可共存且均默认关闭(零开销)。
   用途: 常态化运行累积"planner 会错/不会错"的证据, 为 G3(真接管) 提供依据。
4. **代价判定重设计(新立项)**: 修正收益模型(省中间量物化 vs 现行省外部输入重读),
   并补独立第二约束避免判据退化为"总是合并"。**这是 G2→G3 的前置**(影子阶段不需要它)。
