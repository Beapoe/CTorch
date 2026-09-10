/**
 * @file bench_llama_ffn_train.cpp
 * @brief LLaMA-1B 尺寸 SwiGLU FFN 训练基准
 * @details FFN = silu(x @ W_gate) * (x @ W_up), 再 @ W_down。默认 hidden=4096,
 *          intermediate=11008 (LLaMA-1B 真实维度), batch*seq=128, FP32。
 *          训练 loop(forward + backward + SGD), 逐段计时, 用于回答:
 *          训练期 forward region fusion 的真实命中率与收益(fwd/bwd/upd 占比)。
 *          三环境对照(同 binary): C3 default / C3_DISABLE_HOTPATH=1 /
 *          C3_DISABLE_REGION_FUSION=1。
 * @date 2026-09-06
 */
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstdlib>
#include <vector>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <execinfo.h>
#include <unistd.h>
#ifdef __APPLE__
#include <dlfcn.h>
#endif
#include <atomic>
#include <mutex>
#include <unordered_map>
#include "Tensor.h"
#include "AutoGrad.h"
#include "CtorchScheduler.h"
#include "C3/C3HotPathManager.h"
#include "C3/C3KernelRegistry.h"
#include "C3/C3BackwardCapture.h"
#include "C3/C3Config.h"
#include "C3/C3Cleanup.h"
#include "C3/ForwardCapture.h"       // [HOOK] forward 整图捕获(旁路采集)
#include "C3/FusionPlanner.h"       // [HOOK] 通用融合判据(旁路采集)

extern "C" void cblas_saxpy(const int N, const float alpha, const float *X, const int incX,
                            float *Y, const int incY);

// ===== cblas_sgemm 强符号劫持 probe(与 test_c3_mnist_train 同款) =====
namespace ctprobe {
    struct Bucket { uint64_t ns = 0; uint64_t cnt = 0; };
    std::mutex& mu() { static std::mutex m; return m; }
    std::unordered_map<uint64_t, Bucket>& map() { static std::unordered_map<uint64_t, Bucket> m; return m; }
    std::atomic<bool>& on() { static std::atomic<bool> e{false}; return e; }
}
extern "C" {
void record_cblas_probe(int M, int N, int tA, int tB, unsigned long long ns) {
    if (!ctprobe::on().load(std::memory_order_relaxed)) return;
    uint64_t key = ((uint64_t)(uint32_t)M << 40) | ((uint64_t)(uint32_t)N << 16)
                 | ((uint64_t)(uint32_t)tA << 8) | (uint32_t)tB;
    std::lock_guard<std::mutex> lk(ctprobe::mu());
    auto& b = ctprobe::map()[key];
    b.ns += ns; b.cnt++;
}
void report_cblas_probe() {
    std::lock_guard<std::mutex> lk(ctprobe::mu());
    fprintf(stderr, "[FFN-CBLAS] bucket(M,N,tA,tB):\n");
    for (auto& kv : ctprobe::map()) {
        uint64_t M = (kv.first >> 40) & 0xFFFFFFFF, N = (kv.first >> 16) & 0xFFFFFF;
        int tA = (kv.first >> 8) & 0xFF, tB = kv.first & 0xFF;
        fprintf(stderr, "[FFN-CBLAS] M=%llu N=%llu tA=%d tB=%d total_us=%llu count=%llu avg_us=%.3f\n",
                (unsigned long long)M, (unsigned long long)N, tA, tB,
                (unsigned long long)(kv.second.ns / 1000), (unsigned long long)kv.second.cnt,
                kv.second.cnt ? (double)kv.second.ns / kv.second.cnt / 1000.0 : 0.0);
    }
}
#ifdef __APPLE__
// Apple-only: 强符号劫持 cblas_sgemm 统计 GEMM(GEMM 探针)。
// 非 Apple(x86 Linux/DCU)平台不劫持: CBLAS_ORDER/CBLAS_TRANSPOSE 类型不存在, matmul 走项目 SIMD 后端。
void cblas_sgemm(const enum CBLAS_ORDER __Order, const enum CBLAS_TRANSPOSE __TransA,
                 const enum CBLAS_TRANSPOSE __TransB, const int __M, const int __N, const int __K,
                 const float __alpha, const float* __A, const int __lda, const float* __B,
                 const int __ldb, const float __beta, float* __C, const int __ldc) {
    static auto real = (void (*)(enum CBLAS_ORDER, enum CBLAS_TRANSPOSE, enum CBLAS_TRANSPOSE,
                                 int, int, int, float, const float*, int, const float*, int,
                                 float, float*, int))dlsym(RTLD_NEXT, "cblas_sgemm");
    if (!real) return;
    auto t0 = std::chrono::steady_clock::now();
    real(__Order, __TransA, __TransB, __M, __N, __K, __alpha, __A, __lda, __B, __ldb, __beta, __C, __ldc);
    auto ns = (unsigned long long)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - t0).count();
    record_cblas_probe(__M, __N, (int)__TransA, (int)__TransB, ns);
}
#endif  // __APPLE__
}  // extern "C"

static void crashHandler(int sig) {
    void* arr[64];
    int n = backtrace(arr, 64);
    dprintf(2, "\n=== CRASH sig=%d ===\n", sig);
    backtrace_symbols_fd(arr, n, 2);
    _exit(128 + sig);
}

using namespace ct;
using namespace ct::c3;
using hires = std::chrono::high_resolution_clock;
using msd = std::chrono::duration<double, std::milli>;

static size_t BS = 128, HID = 4096, INT = 11008, STEPS = 12;
static const float LR = 0.001f;

static void fill(Tensor& t, float seed) {
    float* p = t.data_write<float>();
    for (size_t i = 0; i < t.numel(); ++i) p[i] = seed + (float)(i % 31) * 0.001f;
}

int main(int argc, char** argv) {
    signal(SIGSEGV, crashHandler);
    signal(SIGABRT, crashHandler);
    ctprobe::on().store(std::getenv("FFN_CBLAS_PROBE") != nullptr);
    if (argc >= 5) {
        BS = (size_t)std::atoll(argv[1]); HID = (size_t)std::atoll(argv[2]);
        INT = (size_t)std::atoll(argv[3]); STEPS = (size_t)std::atoll(argv[4]);
    }
    const char* env_rf = std::getenv("C3_DISABLE_REGION_FUSION");
    const char* env_hp = std::getenv("C3_DISABLE_HOTPATH");
    std::cout << "=== LLaMA-FFN train (BS=" << BS << " HID=" << HID << " INT=" << INT
              << " steps=" << STEPS << ") ===\n";
#ifndef CT_DISABLE_C3
    std::cout << "  C3 ON | hotpath=" << (env_hp && std::string(env_hp) == "1" ? "OFF" : "ON")
              << " region_fusion=" << (env_rf && std::string(env_rf) == "1" ? "OFF" : "ON") << "\n";
#else
    std::cout << "  Eager (C3 OFF)\n";
#endif
    std::cout.flush();

    Tensor W_g(ShapeTag{}, {HID, INT}), W_u(ShapeTag{}, {HID, INT}), W_d(ShapeTag{}, {INT, HID});
    fill(W_g, 0.01f); fill(W_u, 0.02f); fill(W_d, 0.03f);
    W_g.requires_grad(true); W_u.requires_grad(true); W_d.requires_grad(true);
    Tensor x(ShapeTag{}, {BS, HID});
    fill(x, 0.1f);  // 输入不 requires_grad(与 mnist_train 训练语义一致)
    // 分类头 + one-hot labels(用 cross_entropy 作 loss: sum()/mean() 反向链已知断链,
    // DotNode 缺失, 见 STATUS 4.58)
    Tensor W_cls(ShapeTag{}, {HID, 10});
    fill(W_cls, 0.05f);
    W_cls.requires_grad(true);
    Tensor one_hot(ShapeTag{}, {BS, 10});
    {
        float* p = one_hot.data_write<float>();
        std::memset(p, 0, BS * 10 * sizeof(float));
        for (size_t i = 0; i < BS; ++i) p[i * 10 + (i % 10)] = 1.0f;
    }

#ifndef CT_DISABLE_C3
    HotPathConfig hp_cfg;
    hp_cfg.hot_threshold = 2;  // 快速触发编译, 与 bench_wide_mlp_e2e 一致
    C3HotPathManager::instance().configure(hp_cfg);
#endif

    auto step_fn = [&]() {
        Tensor g = x.matmul(W_g).silu();   // MatMul + SiLU (region pattern 候选)
        Tensor u = x.matmul(W_u);          // MatMul
        Tensor h = g * u;                  // Mul (SwiGLU 逐元素门控)
        Tensor out = h.matmul(W_d);        // MatMul
        Tensor logits = out.matmul(W_cls); // 分类头
        static const bool use_sum = std::getenv("FFN_LOSS_SUM") != nullptr;
        Tensor loss = use_sum ? out.sum() : logits.cross_entropy(one_hot);
        return loss;
    };

    // warmup(让 C3 编译完成, 不计时)
    for (size_t s = 0; s < 3; ++s) {
        Tensor loss = step_fn();
        std::cerr << "[DBG] node=" << (loss.getRelatedNode() ? "ok" : "NULL")
                  << " requires_grad=" << (loss.requires_grad() ? 1 : 0) << "\n";
        AutoGrad::backward(loss.getRelatedNode(), false);
        std::cerr << "[DBG] after bwd: W_g=" << (W_g.grad_ptr() ? "ok" : "NULL")
                  << " W_u=" << (W_u.grad_ptr() ? "ok" : "NULL")
                  << " W_d=" << (W_d.grad_ptr() ? "ok" : "NULL") << "\n";
        W_g.zero_grad(); W_u.zero_grad(); W_d.zero_grad(); W_cls.zero_grad();
    }

    // [HOOK 旁路采集] C3_HOOK_CAPTURE=1: 真实 FFN forward 整图捕获 + planner 分区。
    // (默认关, 不改训练行为/结果; 作进 G2 的 forward 一致率决策数据。与 MNIST 同款)
    {
        static const bool hook = [] { const char* e = std::getenv("C3_HOOK_CAPTURE"); return e && *e == '1'; }();
        if (hook) {
            Tensor loss = step_fn();
            auto cap = ct::c3::ForwardCapture::capture(loss);
            if (!cap.ok) {
                fprintf(stderr, "[HOOK] FFN capture_FAIL: %s\n", cap.error.c_str());
            } else {
                auto plan = ct::c3::FusionPlanner::planUnits(cap.graph);
                fprintf(stderr, "[HOOK] FFN fwd graph nodes=%zu inputs=%zu compute_units=%zu\n",
                        cap.graph.nodeCount(), cap.graph.inputCount(), plan.compute_unit_count);
                for (const auto& u : plan.units) {
                    if (!u.isCompute()) continue;
                    fprintf(stderr, "[HOOK]   unit kind=%d n=%zu\n", (int)u.kind, u.node_ids.size());
                }
            }
        }
    }

    double fwd_ms = 0, bwd_ms = 0, upd_ms = 0;
    for (size_t s = 0; s < STEPS; ++s) {
        auto t0 = hires::now();
        Tensor loss = step_fn();
        auto t1 = hires::now();
        AutoGrad::backward(loss.getRelatedNode(), false);
        auto t2 = hires::now();
        auto upd = [&](Tensor& p) {
            float* gp = p.grad_ptr();
            if (!gp) {
                // 死分支参数(如 sum-loss 下未参与的 CE 头)无梯度, 跳过更新
                std::cerr << "[UPD-SKIP] no grad for param numel=" << p.numel() << "\n";
                return;
            }
            cblas_saxpy((int)p.numel(), -LR, gp, 1, p.data_write<float>(), 1);
        };
        if (std::getenv("FFN_DUMP_GRAD") && s < 2) {
            float* ggd = W_g.grad_ptr();
            float* gud = W_u.grad_ptr();
            float* gdd = W_d.grad_ptr();
            fprintf(stderr, "[GRAD-DUMP] step=%zu loss=%.6f Wg[0:2]=%.6f,%.6f Wu[0:2]=%.6f,%.6f Wd[0:2]=%.6f,%.6f\n",
                    s, loss.item<float>(),
                    ggd ? ggd[0] : -99.0f, ggd ? ggd[1] : -99.0f,
                    gud ? gud[0] : -99.0f, gud ? gud[1] : -99.0f,
                    gdd ? gdd[0] : -99.0f, gdd ? gdd[1] : -99.0f);
        }
        upd(W_g); upd(W_u); upd(W_d); upd(W_cls);
        auto t3 = hires::now();
        double f = std::chrono::duration_cast<msd>(t1 - t0).count();
        double b = std::chrono::duration_cast<msd>(t2 - t1).count();
        double u2 = std::chrono::duration_cast<msd>(t3 - t2).count();
        fwd_ms += f; bwd_ms += b; upd_ms += u2;
        if (s >= STEPS - 3 || s < 2) {
            std::cout << std::fixed << std::setprecision(1) << "  step " << s
                      << ": fwd=" << f << "ms bwd=" << b << "ms upd=" << u2
                      << "ms (loss=" << std::setprecision(4) << loss.item<float>() << ")\n";
        }
        W_g.zero_grad(); W_u.zero_grad(); W_d.zero_grad(); W_cls.zero_grad();
    }

    std::cout << std::fixed << std::setprecision(1)
              << "  avg: fwd=" << fwd_ms / STEPS << "ms  bwd=" << bwd_ms / STEPS
              << "ms  upd=" << upd_ms / STEPS << "ms  total=" << (fwd_ms + bwd_ms + upd_ms) / STEPS
              << "ms/step\n";
    std::cout.flush();

#ifndef CT_DISABLE_C3
    {
        auto s = C3KernelRegistry::getInstance().getStats();
        auto hp = C3HotPathManager::instance().getStats();
        fprintf(stderr, "[FFN-C3-STAT] single_active=%zu fused_entries=%zu hit=%zu miss=%zu bypass=%zu fused_hit=%zu compiles=%zu tracked=%zu\n",
                s.active_entries, s.fused_entries, s.hit_count, s.miss_count, s.bypass_count,
                s.fused_hit_count, hp.compilations_triggered, hp.calls_tracked);
        auto bw = C3BackwardCapture::getInstance().getStats();
        fprintf(stderr, "[FFN-BW-STAT] bw_hit=%zu bw_miss=%zu fusion_hit=%zu mimo_hit=%zu mimo_miss=%zu mimo_exec_us=%llu\n",
                bw.cache_hit_count, bw.cache_miss_count, bw.fusion_hit_count, bw.mimo_hit_count,
                bw.mimo_miss_count, (unsigned long long)bw.mimo_exec_us);
    }
#endif
    report_cblas_probe();
    C3Engine::getInstance().shutdown();
    return 0;
}
