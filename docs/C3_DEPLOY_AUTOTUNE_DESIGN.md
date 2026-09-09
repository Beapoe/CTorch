# C3 部署时自适应校准设计 (Deploy-time Auto-calibration)

> **状态**: 设计草案 (2026-09-07, 洛锦/苏璃珞)
> **性质**: 设计文档, 暂不实现。实现前需按项目红线 + 跟洛锦确认 scope。
> **引用**: 本文件是 C3 跨硬件"该用哪种优化"问题的统一答案, 接入点见 AGENTS.md。

## 1. 为什么需要

C3 的许多优化决策是**硬件相关**的, 不能拍脑袋定, 也无法跨机器打包票:

| 决策 | 例子 | 为什么不能写死 |
|---|---|---|
| GEMM 合并/批处理 | 共享操作数的两个 `x@W` 合成一个大 GEMM | M3+FP32 上实测合并反而慢 1-10%(见下); DCU/GPU/大核 x86 可能反超 |
| 线程数 | 并行切片线程上限 | 取决于核数与 cache |
| 向量宽度 | NEON 4-wide / AVX2 8 / AVX-512 16 | 已有编译期检测(61bec0c), 但运行时可再验 |
| 优化级别 | Tier O2 / Ofast(fast-math 多项式逼近) | fast-math 精度-速度权衡机器相关 |
| 后端选择 | cblas / AMX / 手写 | 后端实现质量随硬件变 |
| batch 分块 | 巨型张量切片阈值 | cache/内存带宽相关 |

**触发点**: 2026-09-06 实测 `M=64..2048` 全程合并不占优(M3), 但结论只覆盖本机。
要普适, 必须由编译器在**目标机器**上实测并记忆最优——即本设计。

## 2. 核心概念

**Deploy-time Auto-calibration(部署时一次自动校准)**: C3 在某台机器/某组形状首次投入前,
自动运行一组**受限微基准**, 探测各优化决策在该硬件的取舍, 把结果持久化为
**"机器性能指纹"配置文件**; 运行时编译器据此做决策。它是"运行时学习(PGO/hotpath)"
的**机器级、持久化前置层**。

与 PGO 的区别:
- PGO/三层编译: 学习"这个程序的热路径/哪些 kernel 值得编"(程序相关)
- 本设计: 学习"这台机器对每种优化怎么选"(硬件相关, 程序无关, 可跨程序复用)

## 3. 可校准的决策域(初版收敛范围)

初版**只校准能自动收益、测量稳定**的几项, 避免校准成本失控:

1. **GEMM 策略**: 对若干代表形状(小/中/大 batch × 宽/窄 K) 实测:
   `两次 vs 合并(共享操作数)`、`cblas vs 自研`、`是否 transpose folding`, 存 shape→最优。
2. **并行线程数**: 实测并行切片最优线程数(1..P 核)。
3. **优化级别默认**: 实测 Tier O2 vs Ofast(fast-math) 的 精度损失+速度, 存是否默认开 Ofast。
4. **巨型张量切片阈值**: 实测多核并行 vs 单核的盈亏点(元素数阈值)。
5. **向量宽度**(若编译期宏与实际核不符时可验)。

其余(region fusion pattern / MIMO 结构)是结构性的、机器无关, 不在此校准。

## 4. 触发时机

- **首次部署**: 进程第一次跑 C3 且发现无指纹文件时, 在后台/空闲线程执行校准(不阻塞主训练)。
- **指纹过期/硬件变更**: 指纹带机器特征(CPU 型号/核数/内存)哈希; 变更则重校准。
- **手动**: env `C3_RECALIBRATE=1` 强制重跑。
- 校准期间主路径用保守默认(不因校准等待), 完成后热替换。

## 5. 配置 Schema(草案, JSON)

```jsonc
{
  "machine_fingerprint": "Apple-M3-Pro-5P6E-18GB-arm64",   // CPU 型号+核数+内存哈希
  "calibrated_at": "2026-09-07T...",
  "gemm": {
    // shape 签名 -> 决策
    "M:64-256,N:11008,K:4096": { "strategy": "separate" },  // 合并在此硬件不占优
    "M:2048+,N:11008,K:4096": { "strategy": "separate" }
  },
  "threads": { "best": 5, "slice_threshold_elements": 262144 },
  "opt_level_default": { "o2_only": true, "fast_math": false },
  "vector_lanes": 4
}
```

运行时提供 `calibration::get("gemm", shapeKey)` 查询, 未命中走保守默认 + 提示可校准。

## 6. 校准方法学(测才准)

- **复用 bench 设施**: bench_llama_ffn_train / cblas probe 已是注入式微基准, 校准器复用同款
  (强符号劫持计时 / 分 shape 探测)。
- **干净环境**: 无并发构建/后台; 多次取最优(非均值, 抗调度噪声)。
- **有限配置空间**: 每个决策只测 2-3 个候选(不穷举), 控制总校准时间(< 数秒/形状组)。
- 结果可复现, 记录在指纹里带测速方法版本。

## 7. 与现有设施融合

| 现有 | 角色 |
|---|---|
| `C3Config.h`(env 开关) | 保留为**手动覆盖最高优先**; 指纹只填"未显式指定"的默认 |
| `PGOManager`/三层编译 | 程序级热路径学习, 与指纹互补(指纹在更早层) |
| `HotPathManager` | 运行时命中统计, 不改 |
| `C3_MATMUL_NO_CBLAS` 等 | 可纳入指纹可调项 |

## 8. 演进路径

1. **Phase 0(本设计落地)**: 定 schema + 查询 API + 指纹读写 + env 覆盖。
2. **Phase 1(最小可用)**: 只校准 GEMM 合并决策 + 线程数; 用 M3 已测数据做 seed 指纹。
3. **Phase 2**: 扩到 opt-level/fast-math/切片阈值; 后台校准 + 热替换。
4. **Phase 3(跨机验证)**: DCU / x86 AVX-512 / 大内存机上重跑, 验证决策确实跨机器分化。

## 9. 边界 / 成本 / 风险

- **成本**: 首次校准数秒~数十秒(后台), 可接受; 需防校准污染生产。
- **过拟合风险**: 校准形状组须覆盖程序真实形状, 否则决策不适用 → 允许运行时按真实 shape 增量补充。
- **与编译期宏冲突**: 编译期硬关的(如 CT_C3_DISABLE_*)不能被校准覆盖(最高优先级原则)。
- 不替代"用户显式 env 配置": 显式 > 校准 > 默认。

## 10. 关联文档

- `AGENTS.md` — 项目入口/当前状态/红线
- `STATUS_CONTEXT.md` — 决策日志
- `docs/C3_PERF_UNIFIED_MATRIX.md` — 性能口径
- `c3/include/C3/C3Config.h` — 现有配置开关(校准写入的默认层之上)
- bench: `bench_llama_ffn_train`(FFN/MIMO 对照), `test_c3_mnist_train`(cblas probe 注入式计时)

## 11. 落地进度(2026-09-07): c3ctl + MachineFingerprint + 代价门桥接
- 生命周期已定型: **c3ctl 首次部署校准 → 写 c3.fingerprint → 运行时 Engine 启动 loadDefault()
  一次, 之后 O(1) 读**。env C3_FINGERPRINT 覆盖路径。
- `c3ctl calibrate`: 测有效内存带宽(memcpy)+ C3 单次 launch 税(极小 kernel 反复 execute) →
  写 launch_unit_bytes; `c3ctl show` 用 O(1) 读取路径打印验证。
- `MachineFingerprint`(c3 单例, 线程安全): load 一次后 getter O(1); 缺失回退保守默认, 不抛异常。
- `RegionFusionPolicy::fromMachineDefaults()`: 代价门 launch 项取自指纹。
- 实测 M3: bandwidth≈24.6GB/s, launch_us≈0.486 → launch_unit_bytes≈12KB。
- 下一步: 把其余决策域(GEMM 分 shape 策略/线程/opt_level)逐个并入 c3ctl 校准 + 指纹 schema 扩 JSON;
  并把 fromMachineDefaults 接进运行时真正消费 planner 代价门的位置(待运行时接管后)。
