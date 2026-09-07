# 任务：在超算/高性能节点上跑 C3 (CTorch JIT 编译器) 跨机器 + DCU 性能/功能实证

> 本 prompt 提交给超算节点的调度/执行 Agent。目标是为 C3 论文收集**非 M3 机器**的可复现数据，并验证
> **海光 DCU(GPU)** 支持。你不需要理解 C3 全部原理，按下面步骤执行；遇到环境/依赖不满足，如实报告并
> 返回实际输出，不要臆测或跳过。

## 背景（为何测）

C3 是 CTorch 框架的 JIT 编译器(MLIR/LLVM 22.1.8→ExecutionEngine + 可选 DCU AMDGPU)。
在 Apple M3 Pro 测得 MNIST 稳态 epoch ~138-160ms(acc 97.1421%)、LLaMA-style FFN C3 比 Eager 快 ~5%。
为证明不限于单台 M3、并验证跨硬件(尤其 x86 CPU 与 DCU GPU)，请你在目标节点上复跑同一组基准。

## ⛔ 明确禁止
- **不要安装/运行 PyTorch 或任何 Python 深度学习框架**(pytorch 无 DCU 版本, 且本任务只测 C3)。
- 不要修改 C3/CTorch 源码去"适配"环境；不确定就问。

## 你的职责
1. 检测节点硬件/OS/编译器/LLVM22/DCU(ROCm+DTK+hipBLAS+GCVM)。
2. clone 代码(见下, 需用户授权凭据/网络)。
3. 构建：纯 CPU(x86) 必须能跑；若有 DCU 再叠加 DCU 分支。
4. 跑通能跑的基准，收集数字。
5. 按指定 Markdown 返回。

## 获取代码(clone)

```bash
git clone -b feature-DCU --recurse-submodules https://github.com/ShengFlow/CTorch.git ct
cd ct
# 子模块 c3 若未拉全: git submodule update --init --recursive
```
(若网络/凭据受限，用用户提供的代码副本代替。)

## 环境自检(先做, 写进结果)
- `uname -a` / CPU 型号/核数 / 内存 / OS / C++ 编译器
- LLVM-MLIR 22.x? (`llvm-config --version`、找 `libMLIR*.a`/`libLLVM*.a`)
- DCU/ROCm/DTK? (`rocm-smi`、`ls /public/software/compiler/rocm/`、`ls /opt/dtk*`、hipBLAS、GCVM)
- CMake≥3.16 / ninja

## 构建策略

### 1. 纯 CPU(x86) — 必跑(跨机性能对照就够)
- 需要 MLIR-LLVM 22.1.8(主仓/子模块都依赖)。若机器已有直接配; 没有而装不了则报告(不强行)。
- CMake Release + ninja, 产出 `test_c3_mnist_train`/`bench_llama_ffn_train`/`test_c3_graph`/`test_sum_mean_grad`。
- MNIST 数据文件 `train-images-idx3-ubyte`/`train-labels-idx3-ubyte` 放运行目录。

### 2. DCU(GPU) — 若检测到 DTK/hipBLAS/GCVM 则叠加
- 仓库已适配海光 DCU: CMake 找 `CT_ENABLE_DCU`(默认 OFF); DTK 常见路径
  `/public/software/compiler/rocm/dtk-26.04` 或 `/opt/dtk`(仓库脚本 `scripts/probe-dcu-dtk24.sh` 可探)。
- 构建示例: `cmake -DCT_ENABLE_DCU=ON -DCMAKE_BUILD_TYPE=Release ..`
- DCU 若构建/运行不稳: **仍把纯 CPU 的 MNIST/FFN 跑完**(CPU 数字是跨机对照主证据), 另报告 DCU 到哪一步失败。

## 必跑基准

### A. 机器基线
- 单次大 GEMM 吞吐: 用可用 BLAS(openblas/MKL/自研)测 `sgemm [128,4096]x[4096,11008]` 多次取最优, 报 GFLOP/s; 无 BLAS 写 N/A。
- 报 CPU 核数/超线程。

### B. MNIST 训练稳态(核心, 复现 M3)
命令(仓库根): `./build-release/test_c3_mnist_train`
收集:
- 稳态 epoch ms(忽略前 1-2 个含编译的 epoch, 取最后稳态, 跑 2-3 次)
- 最终 acc / loss(应 ~97.1% / ~0.098)
- HOTSPOT 构成: `Forward (JIT)` / `Backward (Grad)` / `Loss` / `Optimizer (SGD)` ms 与占比
- `[C3-STAT]`/`[C3-BW-STAT]` 关键字段(mimo_hit/compile/命中)
- 纯 Eager 对照: 若有 `CT_DISABLE_C3` 版(`build_eager/`), 跑一次给 Forward/Backward 段对比。

### C. LLaMA-style FFN(若可构建 bench)
`./build-release/bench_llama_ffn_train 128 4096 11008 8` → `avg: fwd/bwd/upd/total` ms/step;
若可再跑 `./build-eager/bench_llama_ffn_train 128 4096 11008 8` 做 C3 vs Eager。

### D. 数值 sanity(可选但推荐)
`./build-release/test_c3_graph`(PASSED 计数)、`./build-release/test_sum_mean_grad`。

### E. DCU(GPU) — 仅在检测到 DCU 且构建成功时
- `./build-release/test_c3_dcu_hello`、`./build-release/test_dcu_hello_v2`(若产出) — 报告 GPU kernel 是否跑通/输出。
- 若 C3 有 DCU matmul/编译路径的示例命令, 也跑; 没有就报告"仓库 DCU 测试仅上述 hello 级"。
- 明确记录: DCU 上什么成功、什么失败/不支持(尤其 MLIR→AMDGPU 路径)。

## 结果返回格式(Markdown, 如实填)
```markdown
## HPC 节点实证结果
- 节点: <CPU/核数/内存/OS/编译器>
- LLVM/MLIR: <版本或缺失>
- DCU/ROCm: <有无/DTK 路径/是否构建成功>
- 构建: <成功/失败/策略>
### A 机器基线: sgemm GFLOP/s <值或 N/A+原因>
### B MNIST
- 稳态 epoch ms / acc / loss
- Forward/Backward/Loss/SGD ms(逐段)
- C3-STAT / C3-BW-STAT 摘要
- 纯 Eager 对照(若有): Forward/Backward ms
### C FFN: C3 avg ms/step <fwd/bwd/upd/total>; Eager avg(若有)
### D sanity: graph passed N/N; sum_mean_grad pass/na
### E DCU(GPU): hello 结果 / GPU 支持状况 / 失败点
### 差异/备注: 与 M3(~138-160ms 稳态, FFN~180ms)差异; 无法完成项+原因
```

## 铁律
- 只报告真实数字; 失败写原因, 不伪造/臆测。
- 测量前避免并行构建/后台污染(先构建完再单独跑测)。
- 每项耗时测 2-3 次取稳态, 注明负载影响。
- 不确定就问, 不擅自改 C3/CTorch 代码。
- 严禁装/跑 pytorch 等 Python ML 栈。
