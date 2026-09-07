# CTorch Agent Context

> AI agent onboarding doc for **CTorch** — 笙歌/ShengFlow 团队的轻量级 C++ 深度学习框架。
> Last updated: 2026-09-06 (session: sum/mean 断链 + FFN 反向 MIMO + 性能画像)

## 项目一句话

轻量级 C++ 深度学习框架, 类 PyTorch 接口, 核心是 **C3 JIT 编译器** (MLIR → LLVM IR → ExecutionEngine) + 区域融合 (region fusion) + MIMO 反向融合 + 多后端 kernel (CPU-BASIC / CPU-SIMD / AMX / MPS)。

## 当前状态 (2026-09-06 晚)

| 领域 | 状态 | 关键交付 |
|------|------|----------|
| PEL25 Stage 1-5 (SwiGLU/SiLU + region fusion) | ✅ DONE | 30 op; SiLU/SwiGLU; MatMul+SiLU region fusion |
| **MatMul epilogue 向量化** | ✅ DONE | 移除 vector.broadcast → arith-on-vector + undef/insertelement splat (c3 4b1d459) |
| **DEBT-2 (fused backward)** | 🔴 superseded | 被 MIMO 取代, 不复活 (c3 43d9fbe, STATUS §4.54) |
| **sum()/mean() 家族反向断链** | ✅ DONE | DotNode 缺失 bug; SumNode/MeanNode/DimReduceNode + NEON SIMD (主仓 99e1fae/dbe6e92/b57a52d) |
| **sum-loss 死分支断链** | ✅ DONE | ComputeCore 活跃子图依赖重算 (主仓 3085a6b) |
| **LLaMA FFN 反向 MIMO** | ✅ DONE | 无 bias SwiGLU FFN 整段反向→单内核 9 输出 (c3 12ac4c6, STATUS §4.59) |
| LLaMA-1B FFN bench | ✅ 新增 | `bench_llama_ffn_train` (c3 vs eager ~5% 快, bwd ~8%) |
| 论文 | ✅ 更新 | 中英 MIMO 节加"无 bias SwiGLU FFN"扩展 (本地 paper/, gitignored) |

**最近变更速览** (详细日志见 `STATUS_CONTEXT.md` §4.53-4.59 + git log):
- §4.53 MatMul epilogue 向量化; §4.54 DEBT-2 降级 + MNIST 画像
- §4.55 全量回归矩阵; §4.56/4.57 偷工减料审查+修复
- §4.58 sum/mean 断链 + LLaMA-FFN bench; §4.59 FFN MIMO + sum-loss 断链遗留→已修(3085a6b)

**当前性能基线** (M3 Pro, 需干净机器, 数值受热降频 ±15% 波动):
- MNIST 训练稳态 epoch ~138-160ms, acc 97.1421%, loss 0.0985
- LLaMA FFN(128×4096×11008): C3 ~180ms/step vs eager ~190ms (bwd MIMO ~8% 快)
- MIMO 命中: MNIST mimo_hit 4678/epoch; FFN mimo_hit 命中, bw_hit 66→16

## 🔧 下一步待办 (2026-09-06)

0. **部署时自适应校准(新设计)**: GEMM 合并/线程/opt-level/向量宽等**硬件相关决策**不应拍板或特化。
   探针(M3)证实 GEMM 合并在 M=64..2048 全程不占优, 但跨硬件会分化。
   → 设计 `docs/C3_DEPLOY_AUTOTUNE_DESIGN.md`(机器性能指纹, 首跑校准持久化)。实现前先批 scope。
1. ~~batched GEMM 合并~~ → 砍: 特化 + M3 实测合并负收益(-1~10%)。GEMM 决策改走第 0 条自适应校准。
2. **修 pre-existing standalone 失败**: test_c3_pgo_deopt/compile_error 已修绿; 仍红 = test_relu_backward
   (MPS 设备类型崩溃, 不经 C3)、test_region_fusion(性能退化, bench 波动类)。
3. DCU 节点验证 + x86 AVX-512 实测 (曙光智算, 机时充足; 正好验证自适应校准的跨机分化)。
4. forward 优化 + RC2 进程级异步 (c3d, docs/C3_PROCESS_ASYNC_*)。

## 设计蓝图 (docs/, 多未实现)

- `docs/C3_DEPLOY_AUTOTUNE_DESIGN.md` — 部署时自适应校准(机器指纹, 跨硬件决策) 【新增 2026-09-07】
- `docs/C3_PROCESS_ASYNC_BLUEPRINT.md` — RC2 进程级异步 c3d
- `docs/C3_SIMD_CROSSARCH_BLUEPRINT.md` — x86 AVX-512/NEON 跨架构向量化
- 其余 `docs/C3_*.md` 为 bug 报告/论文素材(DEBT2/first-call/paper/perf 等)

## 关键路径速查

| 关注点 | 路径 |
|--------|------|
| op 枚举 (30 个) | `include/Ctools.h:178-221` |
| op 静态断言 | `include/CtorchScheduler.h:229-230` (`kCount==30`) |
| Region candidate 白名单 | `include/CtorchScheduler.h:34-36` (5 pattern: MatMul/Add/ReLU/Sigmoid/SiLU) |
| dispatch 表 | `src/CtorchScheduler.cpp:99, 112, 134, 158, 985` |
| Eager API 入口 | `include/Tensor.h` (~1380 行) |
| AutoGrad dispatch 模板 | `include/AutoGrad.h:113-229` (单/双输入 if constexpr 派发) |
| C3 Engine (MLIR→LLVM) | `c3/src/C3/C3Engine.cpp` |
| C3 region fusion registry | `c3/src/C3/RegionFusionRegistry.cpp` (237 行) |
| C3 region pattern 触发 | `c3/include/C3/C3HotPathManager.h:529-660` (`tryFuseRecentDispatches`) |
| Linalg fused IR gen | `c3/src/C3/LinalgFusedGen.cpp` (SiLU/ReLU/Sigmoid 等 fused body) |
| SIMD 真向量化 | `include/kernels/SIMDMath.h` + `src/kernels/CPU-SIMD/SIMDMath.cpp` |
| Backward graph 捕获 | `c3/src/C3/C3BackwardCapture.cpp` |
| 新算子协议 | `PEL25 §6` + 文档沉淀 → `/Users/ghostface/skills/prompts/new-module-prompt.md` |

## 构建 & 测试

> ⚠️ 本会话(sum/mean/FFN 修复)在 **`build-release/`**(Release + ninja)开发/验证; `build/` 与 `build_eager/` 是另两套(可能旧)。
> - `build-release/`  = C3 + autograd 完整 Release (跑所有 test_c3_* / bench_*)
> - `build_eager/` / `build-eager/` = `CT_DISABLE_C3`(纯 eager 对照, 测 C3 vs eager 用)
> - mnist 数据在仓库根(`train-images-idx3-ubyte` 等), 跑 mnist test 须从根目录执行

```bash
# 构建 (本会话主用 build-release)
cd /Users/ghostface/CTorch-optimize-AutoDiff/build-release
ninja test_c3_graph test_c3_backward test_sum_mean_grad bench_llama_ffn_train  # 按需编目标

# 跑测试(从仓库根, mnist 数据)
cd /Users/ghostface/CTorch-optimize-AutoDiff
./build-release/test_c3_graph      # 115 断言(含 Benchmark)
./build-release/test_sum_mean_grad # sum/mean/dim/dims 梯度回归(18 断言)
./build-release/test_c3_backward   # 反向正确性(max_diff=0)
./build-release/test_c3_mnist_train  # MNIST 端到端训练(acc 97.1421%)
./build-eager/test_c3_mnist_train    # 纯 eager 对照

# LLaMA FFN 训练基准 (c3 vs eager)
./build-release/bench_llama_ffn_train 128 4096 11008 8   # C3(MIMO)
./build-eager/bench_llama_ffn_train 128 4096 11008 8     # 纯 eager 对照
#   env: FFN_CBLAS_PROBE=1(cblas GEMM 分桶) / C3_FFN_DUMP=1(MIMO 中间梯度)
#        FFN_LOSS_SUM=1(sum loss) / FFN_DUMP_GRAD=1(打印 W 梯度)
```

**关键开关** (env):
- `C3_DISABLE_HOTPATH=1` 关闭 C3 hotpath 检测
- `C3_DISABLE_REGION_FUSION=1` 关闭 region fusion
- `C3_DISABLE_SINGLE_KERNEL=1` 关闭单 kernel 编译触发
- `C3_ENABLE_BACKWARD=0` 关 C3 backward(走 eager; 注意 forward 仍可能走 C3 单 kernel, 非纯 eager 对照)

## PEL25 §6 新算子开发协议 (Stage 1-4 沉淀)

**任何新算子必须按以下 7 步走** (PEL25 §6 协议):
1. **接口契约**: `include/Tensor.h` 加 `Tensor::xxx()` 声明 + `include/ops/Xxx.h` 加 Eager API
2. **Eager CPU (BASIC + SIMD)**: `src/ops/Xxx.cpp` + `src/kernels/CPU-{BASIC,SIMD}/Xxx_*.cpp`
3. **Autograd Node**: `include/AutoGrad/Nodes/XxxNode.h` + `.cpp` (4 构造 + 1 backward 虚函数)
4. **op 枚举扩展**: `include/Ctools.h` + `CtorchScheduler.h:229-230` 静态断言更新
5. **C3 Kernel Registry**: 3 个后端 (kCPU/kSIMD/kAMX) dispatch 表注册
6. **MLIR TableGen**: `c3/include/C3/C3Ops.td` (新 op 定义)
7. **Region fusion pattern** (可选): LinalgFusedGen.cpp 加白名单 + C3HotPathManager.h 加 checkPattern

**Stage 5 简化的进阶** (5.1 协议):
- `Tensor::xxx()` 走 `AutoGrad::dispatch<op::Xxx>(...)` 模板, 跟 gelu() 模式一致
- 避免手写 registerNode 逻辑, dispatch 模板 if constexpr 自动派发

## 🔴 绝对不要碰的红线 (洛锦 2026-08-13 警告)

| 路径 | 风险 | 备注 |
|------|------|------|
| `c3/include/C3/C3HotPathManager.h:236-240` (`in_autograd` 短路) | 触及训练一致性核心, 改错破 parity 97.18% | **2026-08-13 revert 警告**, 改前必须跟洛锦确认 |
| `include/CtorchScheduler.h:229-230` 静态断言 | op 枚举跟 binary 不一致会 segfault | 改 op 枚举必须同步 static_assert |
| `include/Ctools.h:178-221` op 枚举顺序 | C3 dispatch 表按 op 索引, 顺序错了 runtime 行为乱 | 新 op 永远加在末尾 (GELU 后) |

## 已知未解决问题

| 级别 | 问题 | 触发/现状 | 建议 |
|------|------|----------|------|
| **P0** | 无 | - | - |
| **P1** | 训练期 region fusion 命中因结构而异 | MNIST(FC 带 bias) fused_hit 高; **LLaMA FFN(无 bias) fused_hit=0**(编译了不执行)。但 C3 default 仍最快(~5-10% vs hotpath-off) | 结论: 不是"C3 浪费"; forward 单 kernel + MIMO 已覆盖。训练期 forward fusion 命中是大 forward 结构(FFN)的可选增益 |
| **P1** | sum-loss(非 CE 头)场景若图含无关死分支 | 已修: ComputeCore 活跃子图依赖重算(3085a6b); 正常 CE loss 训练不受影响 | 保留回归 test_sum_mean_grad(18 断言) |
| **P1** | Stage 5.2 ARM NEON fused 0.77x (反直觉) | x86 AVX-512 + DCU 预期显著加速 | Stage 5.4 DCU 验证 |
| **P1** | x86 AVX-512 实测未做 | 曙光智算机时充足 | Stage 5.4 |
| **P2** | 非核心 standalone 红(pre-existing) | test_relu_backward(MPS 设备崩溃, 不经 C3)、test_region_fusion(性能退化类) | 独立立项; 与主线无交集 |
| **P2** | Stage 1 伪 SIMD (8-wide + 标量 exp) | ops/SiLU.cpp 仍保留 | 可降级 fallback |

## Cross-Project Memory (Agent lessons, 跨项目适用)

append 到 `/Users/ghostface/.minimax/agents/mavis/memory/MEMORY.md` 的 CTorch lessons:
- **2026-08-13**: "C3 region fusion 训练期修复走 multi_node 代码层, 不碰 in_autograd 短路"
- **2026-08-13**: "C3 8.3x forward 退步根因 (误判修正) — 训练期走 Eager bypass, MLIR pipeline 不影响"
- **2026-08-13**: "MiniMax Code 必须通过 launchd plist 拉起, 否则 CDP 9341 没人 listen"
- **2026-09-05**: "PEL 候选 prompt 生成必须 cross-check user/agent memory 硬约束"
- **2026-09-05**: "PEL 启动前必须先 cross-check 种子 prompt 本身"

## 报告路径 (PEL25 阶段产物)

```
/Users/ghostface/skills/work/reports/2026-09-05/
  prompt-evolution-summary-PEL23-25.md    # 3 轮 PEL 总结
  pel{23,24,25}-candidate-Seed.md        # Seed 评测
  pel{23,24,25}-candidate-MUT-A/B/C.md   # MUT 候选评测

/Users/ghostface/skills/work/reports/2026-09-06/
  swiglu-stage4-report.md      # Stage 4 真 SIMD (1.52-1.56x)
  swiglu-stage5-report.md      # Stage 5.1+5.2 dispatch + region fusion

/Users/ghostface/skills/memories/2026-09-05/
  prompt-evolution-failures-pel{23,24,25}.md

/Users/ghostface/skills/prompts/
  performance-optimization-prompt.md  # PEL23 沉淀 + §13
  compiler-flags-prompt.md            # PEL24 沉淀 + §5.8/§12
  new-module-prompt.md                # PEL25 沉淀 + §6+§7
```

## Quick reference: 给 agent 的一条精简 workflow

```bash
# 新会话开头:
1. cat ~/skills/main.md  # 洛锦的 AGI 总纲
2. cd /Users/ghostface/CTorch-optimize-AutoDiff
3. cat AGENTS.md          # 本文件: 当前状态/已知问题/下一步/红线
4. git log --oneline -20  # 看最新 commit; 详细日志 tail STATUS_CONTEXT.md
5. tail -120 STATUS_CONTEXT.md  # 最近几条工作记录(§4.5x)
6. 跟洛锦确认 scope + 决策门

# 跑测试 sanity (主用 build-release, 从仓库根跑 mnist 需数据在根)
cd /Users/ghostface/CTorch-optimize-AutoDiff
./build-release/test_c3_graph && ./build-release/test_sum_mean_grad && ./build-release/test_c3_backward
```

## Test 矩阵 (跑这些保平安)

| 关注点 | 测试 target | 备注 |
|--------|-------------|------|
| C3 graph + Benchmark 全量 | `test_c3_graph`(build-release) | 115 断言含 MLP/MLIR, 必过 |
| **sum/mean 梯度回归** | `test_sum_mean_grad`(build-release) | 18 断言(sum/mean/dim/dims/DotNode 断链回归) |
| 反向正确性 | `test_c3_backward` | max_diff=0 |
| MNIST 端到端训练 | `test_c3_mnist_train`(根目录) | acc 97.1421% 基线 |
| LLaMA FFN MIMO | `bench_llama_ffn_train`(128 4096 11008) | build-release vs build-eager 对照 |
| SwiGLU/SiLU (Stage 5) | `test_swiglu` | 3208 断言 |
| GELU (dispatch 模式) | `test_gelu` | if constexpr 改动必跑 |
| Autograd 通用 | `test_autograd_issues` `test_autograd_v2` | dispatch 模板改动必跑 |
| C3 region fusion | `test_graph_merger` | 改动 LinalgFusedGen/checkPattern 必跑 |
| C3 compile pipeline | `test_c3_compile_merged` `test_c3_compile_merged_pgo` | 10/11 断言 |
| 反向 fusion/DEBT | `test_fused_bw_debt2` | fused BW 默认 off, sanity |
| pgo/错误路径(已修绿) | `test_c3_pgo_deopt` `test_c3_compile_error` | bad_weak_ptr 已修 |
