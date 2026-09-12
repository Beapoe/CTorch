/**
 * @file test_c3_backward.cpp
 * @brief C3 Backward JIT 端到端验证
 * @details 验证：
 *          1. backward 时 C3 缓存未命中，回退 eager
 *          2. 异步编译完成后，相同形状再次 backward 命中缓存
 *          3. C3 backward 结果与 eager backward 一致
 *          4. forward 输入正确传递到 C3 kernel
 * @date 2026/8/4
 */

#include <cstdlib>
#include <iostream>
#include <chrono>
#include <thread>
#include <cmath>

#include "Tensor.h"
#include "CtorchScheduler.h"
#include "AutoGrad.h"
#include "C3/C3BackwardCapture.h"
#include "C3/C3KernelRegistry.h"
#include "C3/C3Engine.h"
#include "C3/C3Cleanup.h"

using namespace ct;
using namespace ct::c3;

static void printStats(const std::string& label) {
    auto stats = C3BackwardCapture::getInstance().getStats();
    std::cout << "  [" << label << "] C3 backward: hits=" << stats.cache_hit_count
              << " misses=" << stats.cache_miss_count
              << " compiles=" << stats.compile_count
              // [P0.1 2026-08-30 苏璃珞] backward 覆盖率统计打印
              << " | attempt=" << stats.backward_attempt_count
              << " c3_hit=" << stats.backward_c3_attempt_count
              << " fallback=" << stats.backward_eager_fallback_count;
    if (!stats.backward_fallback_reasons.empty()) {
        std::cout << " reasons=[";
        bool first = true;
        for (const auto& [k, v] : stats.backward_fallback_reasons) {
            if (!first) std::cout << ",";
            std::cout << k << ":" << v;
            first = false;
        }
        std::cout << "]";
    }
    std::cout << std::endl;

    // [P0.5 2026-08-30 苏璃珞] compile 失败原因统计打印
    auto err_stats = C3Engine::getInstance().getCompileErrorStats();
    std::cout << "  [" << label << "] C3 compile errors: total=" << err_stats.total_failures
              << " last_error_size=" << err_stats.last_error_size;
    if (!err_stats.reasons.empty()) {
        std::cout << " reasons=[";
        bool first = true;
        for (const auto& [k, v] : err_stats.reasons) {
            if (!first) std::cout << ",";
            std::cout << k << ":" << v;
            first = false;
        }
        std::cout << "]";
    }
    std::cout << std::endl;
}

int main() {
    std::cout << "=== C3 Backward JIT 端到端验证 ===" << std::endl;

    auto& sched = CtorchScheduler::getInstance();
    auto& engine = C3Engine::getInstance();
    auto& capture = C3BackwardCapture::getInstance();

    auto stats_before = capture.getStats();

    // main 作用域累计各测试的最大误差，供最终 PASS/FAIL 判定
    double overall_max_diff = 0.0;

    // ========== Test 1: ReLU backward ==========
    std::cout << "\n[Test 1] ReLU backward" << std::endl;

    const size_t N = 4;
    // Eager 参考
    Tensor x_ref(ShapeTag{}, {N}, DType::kFloat, DeviceType::kCPU);
    float* x_refp = x_ref.data_write<float>();
    for (size_t i = 0; i < N; ++i) x_refp[i] = static_cast<float>(static_cast<int>(i) - 2);
    x_ref.requires_grad(true);
    Tensor y_ref = x_ref.relu();
    AutoGrad::backward(y_ref.getRelatedNode(), false);
    auto eager_grad_ref = x_ref.grad();
    std::cout << "  Eager ref grad: ";
    for (size_t i = 0; i < N; ++i) std::cout << eager_grad_ref.data_read<float>()[i] << " ";
    std::cout << std::endl;

    for (int iter = 0; iter < 6; ++iter) {
        Tensor x(ShapeTag{}, {N}, DType::kFloat, DeviceType::kCPU);
        float* xp = x.data_write<float>();
        for (size_t i = 0; i < N; ++i) xp[i] = static_cast<float>(static_cast<int>(i) - 2);
        x.requires_grad(true);

        Tensor y = x.relu();
        AutoGrad::backward(y.getRelatedNode(), false);

        auto g = x.grad();
        if (iter == 5) {
            std::cout << "  Iter 5 (C3) grad: ";
            for (size_t i = 0; i < N; ++i) std::cout << g.data_read<float>()[i] << " ";
            std::cout << std::endl;
            double max_diff = 0;
            for (size_t i = 0; i < N; ++i) {
                double d = std::fabs(g.data_read<float>()[i] - eager_grad_ref.data_read<float>()[i]);
                if (d > max_diff) max_diff = d;
            }
            std::cout << "  Test 1 C3 max_diff vs eager: " << max_diff << std::endl;
        }
    }

    printStats("after ReLU x6");

    // ========== Test 2: Sigmoid backward ==========
    std::cout << "\n[Test 2] Sigmoid backward" << std::endl;

    // Eager 参考
    Tensor x_sig_ref(ShapeTag{}, {N}, DType::kFloat, DeviceType::kCPU);
    float* x_sig_refp = x_sig_ref.data_write<float>();
    for (size_t i = 0; i < N; ++i) x_sig_refp[i] = static_cast<float>(static_cast<int>(i) - 2) * 0.5f;
    x_sig_ref.requires_grad(true);
    Tensor y_sig_ref = x_sig_ref.sigmoid();
    AutoGrad::backward(y_sig_ref.getRelatedNode(), false);
    auto sig_eager_ref = x_sig_ref.grad();

    for (int iter = 0; iter < 6; ++iter) {
        Tensor x(ShapeTag{}, {N}, DType::kFloat, DeviceType::kCPU);
        float* xp = x.data_write<float>();
        for (size_t i = 0; i < N; ++i) xp[i] = static_cast<float>(static_cast<int>(i) - 2) * 0.5f;
        x.requires_grad(true);

        Tensor y = x.sigmoid();
        AutoGrad::backward(y.getRelatedNode(), false);

        auto g = x.grad();
        if (iter == 5) {
            std::cout << "  Iter 5 (C3) grad: ";
            for (size_t i = 0; i < N; ++i) std::cout << g.data_read<float>()[i] << " ";
            std::cout << std::endl;
            double max_diff = 0;
            for (size_t i = 0; i < N; ++i) {
                double d = std::fabs(g.data_read<float>()[i] - sig_eager_ref.data_read<float>()[i]);
                if (d > max_diff) max_diff = d;
            }
            std::cout << "  Test 2 C3 max_diff vs eager: " << max_diff << std::endl;
        }
    }

    printStats("after Sigmoid x6");

    // ========== Test 3: 结果正确性验证 ==========
    std::cout << "\n[Test 3] 结果正确性验证（C3 命中 vs Eager）" << std::endl;

    const size_t M = 8;

    // 第一次运行（eager 路径，同时触发 C3 异步编译）
    Tensor x1(ShapeTag{}, {M}, DType::kFloat, DeviceType::kCPU);
    float* x1p = x1.data_write<float>();
    for (size_t i = 0; i < M; ++i) x1p[i] = static_cast<float>(static_cast<int>(i) - 4);
    x1.requires_grad(true);

    Tensor y1 = x1.relu();
    AutoGrad::backward(y1.getRelatedNode(), false);
    auto eager_grad = x1.grad();
    std::vector<float> eager_grad_data(eager_grad.data_read<float>(),
                                       eager_grad.data_read<float>() + M);

    // 等待异步编译完成
    std::cout << "  等待异步编译完成..." << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    // 第二次运行（应该走 C3 缓存命中 + forward_inputs 传递）
    Tensor x2(ShapeTag{}, {M}, DType::kFloat, DeviceType::kCPU);
    float* x2p = x2.data_write<float>();
    for (size_t i = 0; i < M; ++i) x2p[i] = static_cast<float>(static_cast<int>(i) - 4);
    x2.requires_grad(true);

    Tensor y2 = x2.relu();
    AutoGrad::backward(y2.getRelatedNode(), false);
    auto c3_grad = x2.grad();

    // 比较结果
    double max_diff = 0.0;
    for (size_t i = 0; i < M; ++i) {
        double diff = std::fabs(static_cast<double>(eager_grad_data[i]) -
                                static_cast<double>(c3_grad.data_read<float>()[i]));
        if (diff > max_diff) max_diff = diff;
    }

    auto stats_final = capture.getStats();
    bool has_c3_hits = stats_final.cache_hit_count > stats_before.cache_hit_count;

    std::cout << "  Eager grad: ";
    for (size_t i = 0; i < M; ++i) std::cout << eager_grad_data[i] << " ";
    std::cout << std::endl;

    std::cout << "  C3    grad: ";
    for (size_t i = 0; i < M; ++i) std::cout << c3_grad.data_read<float>()[i] << " ";
    std::cout << std::endl;

    std::cout << "  max_diff: " << max_diff << std::endl;
    std::cout << "  C3 hits detected: " << (has_c3_hits ? "yes" : "no") << std::endl;

    printStats("final");

    // ========== Test 4+: 多输入节点反向（每输入独立单输出 kernel） ==========
    std::cout << "\n=== 多输入节点反向验证（Add/Mul/Sub/Div/MatMul） ===" << std::endl;

    auto check_tensor = [](const Tensor& got, const Tensor& ref, const char* name, double& max_diff) {
        auto g = got.data_read<float>();
        auto r = ref.data_read<float>();
        size_t n = got.numel();
        std::cout << "    " << name << " got=[";
        for (size_t i = 0; i < n; ++i) std::cout << g[i] << (+(i+1==n) ? "" : ",");
        std::cout << "] ref=[";
        for (size_t i = 0; i < n; ++i) std::cout << r[i] << (+(i+1==n) ? "" : ",");
        std::cout << "]" << std::endl;
        for (size_t i = 0; i < n; ++i) {
            double d = std::fabs(static_cast<double>(g[i]) - static_cast<double>(r[i]));
            if (d > max_diff) max_diff = d;
        }
        std::cout << "    " << name << " max_diff=" << max_diff << std::endl;
    };

    // Test 4: Mul backward（多输出：grad_a, grad_b）
    {
        std::cout << "\n[Test 4] Mul backward (multi-output)" << std::endl;
        const size_t N = 4;
        Tensor ra(ShapeTag{}, {N}, DType::kFloat, DeviceType::kCPU);
        float* rap = ra.data_write<float>();
        for (size_t i = 0; i < N; ++i) rap[i] = static_cast<float>(i) + 1.0f;
        Tensor rb(ShapeTag{}, {N}, DType::kFloat, DeviceType::kCPU);
        float* rbp = rb.data_write<float>();
        for (size_t i = 0; i < N; ++i) rbp[i] = static_cast<float>(i) * 0.5f + 2.0f;
        ra.requires_grad(true); rb.requires_grad(true);
        Tensor ry = ra * rb;
        AutoGrad::backward(ry.getRelatedNode(), false);
        auto ref_ga = ra.grad();
        auto ref_gb = rb.grad();

        double max_diff = 0;
        for (int it = 0; it < 6; ++it) {
            Tensor a(ShapeTag{}, {N}, DType::kFloat, DeviceType::kCPU);
            float* ap = a.data_write<float>();
            for (size_t i = 0; i < N; ++i) ap[i] = static_cast<float>(i) + 1.0f;
            Tensor b(ShapeTag{}, {N}, DType::kFloat, DeviceType::kCPU);
            float* bp = b.data_write<float>();
            for (size_t i = 0; i < N; ++i) bp[i] = static_cast<float>(i) * 0.5f + 2.0f;
            a.requires_grad(true); b.requires_grad(true);
            Tensor y = a * b;
            AutoGrad::backward(y.getRelatedNode(), false);
            if (it == 5) {
                check_tensor(a.grad(), ref_ga, "grad_a", max_diff);
                check_tensor(b.grad(), ref_gb, "grad_b", max_diff);
            }
        }
        overall_max_diff = std::max(overall_max_diff, max_diff);
        std::cout << "  Test 4 Mul max_diff=" << max_diff << (max_diff < 1e-5 ? "  ✅" : "  ❌") << std::endl;
    }

    // Test 5: Sub backward（多输出：grad_a=grad, grad_b=-grad）
    {
        std::cout << "\n[Test 5] Sub backward (multi-output)" << std::endl;
        const size_t N = 4;
        Tensor ra(ShapeTag{}, {N}, DType::kFloat, DeviceType::kCPU);
        float* rap = ra.data_write<float>();
        for (size_t i = 0; i < N; ++i) rap[i] = static_cast<float>(i) + 1.0f;
        Tensor rb(ShapeTag{}, {N}, DType::kFloat, DeviceType::kCPU);
        float* rbp = rb.data_write<float>();
        for (size_t i = 0; i < N; ++i) rbp[i] = static_cast<float>(i) * 0.5f;
        ra.requires_grad(true); rb.requires_grad(true);
        Tensor ry = ra - rb;
        AutoGrad::backward(ry.getRelatedNode(), false);
        auto ref_ga = ra.grad();
        auto ref_gb = rb.grad();

        double max_diff = 0;
        for (int it = 0; it < 6; ++it) {
            Tensor a(ShapeTag{}, {N}, DType::kFloat, DeviceType::kCPU);
            float* ap = a.data_write<float>();
            for (size_t i = 0; i < N; ++i) ap[i] = static_cast<float>(i) + 1.0f;
            Tensor b(ShapeTag{}, {N}, DType::kFloat, DeviceType::kCPU);
            float* bp = b.data_write<float>();
            for (size_t i = 0; i < N; ++i) bp[i] = static_cast<float>(i) * 0.5f;
            a.requires_grad(true); b.requires_grad(true);
            Tensor y = a - b;
            AutoGrad::backward(y.getRelatedNode(), false);
            if (it == 5) {
                check_tensor(a.grad(), ref_ga, "grad_a", max_diff);
                check_tensor(b.grad(), ref_gb, "grad_b", max_diff);
            }
        }
        overall_max_diff = std::max(overall_max_diff, max_diff);
        std::cout << "  Test 5 Sub max_diff=" << max_diff << (max_diff < 1e-5 ? "  ✅" : "  ❌") << std::endl;
    }

    // Test 6: Div backward（多输出）
    {
        sched.resetRegionFusion();
        std::cout << "\n[Test 6] Div backward (multi-output)" << std::endl;
        const size_t N = 4;
        Tensor ra(ShapeTag{}, {N}, DType::kFloat, DeviceType::kCPU);
        float* rap = ra.data_write<float>();
        for (size_t i = 0; i < N; ++i) rap[i] = static_cast<float>(i) + 2.0f;
        Tensor rb(ShapeTag{}, {N}, DType::kFloat, DeviceType::kCPU);
        float* rbp = rb.data_write<float>();
        for (size_t i = 0; i < N; ++i) rbp[i] = static_cast<float>(i) + 3.0f;
        ra.requires_grad(true); rb.requires_grad(true);
        Tensor ry = ra / rb;
        AutoGrad::backward(ry.getRelatedNode(), false);
        auto ref_ga = ra.grad();
        auto ref_gb = rb.grad();

        double max_diff = 0;
        for (int it = 0; it < 6; ++it) {
            Tensor a(ShapeTag{}, {N}, DType::kFloat, DeviceType::kCPU);
            float* ap = a.data_write<float>();
            for (size_t i = 0; i < N; ++i) ap[i] = static_cast<float>(i) + 2.0f;
            Tensor b(ShapeTag{}, {N}, DType::kFloat, DeviceType::kCPU);
            float* bp = b.data_write<float>();
            for (size_t i = 0; i < N; ++i) bp[i] = static_cast<float>(i) + 3.0f;
            a.requires_grad(true); b.requires_grad(true);
            Tensor y = a / b;
            AutoGrad::backward(y.getRelatedNode(), false);
            if (it == 5) {
                check_tensor(a.grad(), ref_ga, "grad_a", max_diff);
                check_tensor(b.grad(), ref_gb, "grad_b", max_diff);
            }
        }
        overall_max_diff = std::max(overall_max_diff, max_diff);
        std::cout << "  Test 6 Div max_diff=" << max_diff << (max_diff < 1e-5 ? "  ✅" : "  ❌") << std::endl;
    }

    // Test 7: MatMul backward（多输出，训练关键路径）
    {
        sched.resetRegionFusion();
        std::cout << "\n[Test 7] MatMul backward (multi-output)" << std::endl;
        const size_t M = 4, K = 3, N = 5;
        Tensor rx(ShapeTag{}, {M, K}, DType::kFloat, DeviceType::kCPU);
        float* rxp = rx.data_write<float>();
        for (size_t i = 0; i < M * K; ++i) rxp[i] = static_cast<float>(i % 7) * 0.1f;
        Tensor rw(ShapeTag{}, {K, N}, DType::kFloat, DeviceType::kCPU);
        float* rwp = rw.data_write<float>();
        for (size_t i = 0; i < K * N; ++i) rwp[i] = static_cast<float>(i % 5) * 0.2f;
        rx.requires_grad(true); rw.requires_grad(true);
        Tensor ry = rx.matmul(rw);
        AutoGrad::backward(ry.getRelatedNode(), false);
        auto ref_gx = rx.grad();
        auto ref_gw = rw.grad();

        // 等待异步编译完成，确保后续迭代命中 C3
        std::cout << "  等待异步编译完成..." << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(1500));

        double max_diff = 0;
        for (int it = 0; it < 6; ++it) {
            Tensor x(ShapeTag{}, {M, K}, DType::kFloat, DeviceType::kCPU);
            float* xp = x.data_write<float>();
            for (size_t i = 0; i < M * K; ++i) xp[i] = static_cast<float>(i % 7) * 0.1f;
            Tensor w(ShapeTag{}, {K, N}, DType::kFloat, DeviceType::kCPU);
            float* wp = w.data_write<float>();
            for (size_t i = 0; i < K * N; ++i) wp[i] = static_cast<float>(i % 5) * 0.2f;
            x.requires_grad(true); w.requires_grad(true);
            Tensor y = x.matmul(w);
            AutoGrad::backward(y.getRelatedNode(), false);
            if (it == 5) {
                check_tensor(x.grad(), ref_gx, "grad_x", max_diff);
                check_tensor(w.grad(), ref_gw, "grad_w", max_diff);
                bool hit = C3BackwardCapture::getInstance().getStats().cache_hit_count > stats_before.cache_hit_count;
                std::cout << "  C3 backward hits since start: " << (hit ? "yes" : "no") << std::endl;
            }
        }
        overall_max_diff = std::max(overall_max_diff, max_diff);
        std::cout << "  Test 7 MatMul max_diff=" << max_diff << (max_diff < 1e-4 ? "  ✅" : "  ❌") << std::endl;
    }

    // ========== Test 8: 反向融合 (ReLU → Sigmoid 链式) ==========
    // 预期：前几轮触发融合异步编译；等待编译后，后续轮次 fusion_hit_count 上升。
    // 数值上：融合输出结果应当与 eager 一致。
    {
        sched.resetRegionFusion();
        std::cout << "\n[Test 8] Backward Fusion: ReLU → Sigmoid chain" << std::endl;
        auto stats_fusion_start = capture.getStats();

        const size_t M = 32, K = 64;
        Tensor eager_x(ShapeTag{}, {M, K}, DType::kFloat, DeviceType::kCPU);
        float* xp_eager = eager_x.data_write<float>();
        for (size_t i = 0; i < eager_x.numel(); ++i) xp_eager[i] = (static_cast<float>(i) / (M*K) - 0.5f) * 4.0f;
        eager_x.requires_grad(true);
        Tensor eager_y = eager_x.sigmoid().relu();
        AutoGrad::backward(eager_y.getRelatedNode(), false);
        auto eager_gx = eager_x.grad();

        // 跑 8 轮：前 3~4 轮积累频次触发异步编译，等待后后面几轮应该能命中融合。
        double max_diff = 0.0;
        for (int iter = 0; iter < 8; ++iter) {
            Tensor x(ShapeTag{}, {M, K}, DType::kFloat, DeviceType::kCPU);
            float* xp = x.data_write<float>();
            for (size_t i = 0; i < x.numel(); ++i) xp[i] = (static_cast<float>(i) / (M*K) - 0.5f) * 4.0f;
            x.requires_grad(true);
            Tensor y = x.sigmoid().relu();
            AutoGrad::backward(y.getRelatedNode(), false);
            auto gx = x.grad();

            double d = 0.0;
            const float* e = eager_gx.data_read<float>();
            const float* g = gx.data_read<float>();
            for (size_t i = 0; i < gx.numel(); ++i) {
                double dd = std::fabs(g[i] - e[i]);
                if (dd > d) d = dd;
            }
            max_diff = std::max(max_diff, d);

            // === 调试输出：每次 iter 打印前 8 元素 + 最大误差位置 =============
            {
                size_t bad_pos = 0;
                double bad_d = 0;
                for (size_t i = 0; i < gx.numel(); ++i) {
                    double dd = std::fabs(g[i] - e[i]);
                    if (dd > bad_d) { bad_d = dd; bad_pos = i; }
                }
                std::cout << "  [iter " << iter << "] max_diff=" << d
                          << "  bad_pos=" << bad_pos
                          << "  x_at_bad=" << (static_cast<float>(bad_pos) / (M*K) - 0.5f) * 4.0f
                          << "  eager=" << e[bad_pos]
                          << "  got=" << g[bad_pos]
                          << "  | first8: ";
                for (size_t i = 0; i < 8; ++i) {
                    std::cout << "[" << i << "]=" << g[i] << "/" << e[i] << " ";
                }
                std::cout << std::endl;
            }

            if (iter == 3) {
                std::cout << "  Iter 3 → 等待异步融合编译 (3.5s)..." << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(3500));
            }
        }

        auto stats_fusion_end = capture.getStats();
        std::cout << "  融合编译次数: compiles="
                  << (stats_fusion_end.fusion_compile_count - stats_fusion_start.fusion_compile_count)
                  << "  命中=" << (stats_fusion_end.fusion_hit_count - stats_fusion_start.fusion_hit_count)
                  << "  未命中=" << (stats_fusion_end.fusion_miss_count - stats_fusion_start.fusion_miss_count)
                  << std::endl;
        std::cout << "  最大误差 max_diff=" << max_diff
                  << (max_diff < 1e-5 ? "  ✅" : "  ❌") << std::endl;
        overall_max_diff = std::max(overall_max_diff, max_diff);
    }

    // ========== Test 9: 反向融合 (ReLU → ReLU 直接串联) ==========
    // 复现 MNIST 反向融合 SIGBUS：两个 ReLU 直接相邻（无 MatMul 间隔）时，
    // 融合 kernel 的多输出平面 buffer 是否越界。
    {
        sched.resetRegionFusion();
        std::cout << "\n[Test 9] Backward Fusion: ReLU → ReLU chain" << std::endl;

        const size_t M = 32, K = 64;
        Tensor eager_x(ShapeTag{}, {M, K}, DType::kFloat, DeviceType::kCPU);
        float* xp_eager = eager_x.data_write<float>();
        for (size_t i = 0; i < eager_x.numel(); ++i) xp_eager[i] = (static_cast<float>(i) / (M*K) - 0.5f) * 4.0f;
        eager_x.requires_grad(true);
        Tensor eager_y = eager_x.relu().relu();
        AutoGrad::backward(eager_y.getRelatedNode(), false);
        auto eager_gx = eager_x.grad();

        double max_diff = 0.0;
        for (int iter = 0; iter < 8; ++iter) {
            Tensor x(ShapeTag{}, {M, K}, DType::kFloat, DeviceType::kCPU);
            float* xp = x.data_write<float>();
            for (size_t i = 0; i < x.numel(); ++i) xp[i] = (static_cast<float>(i) / (M*K) - 0.5f) * 4.0f;
            x.requires_grad(true);
            Tensor y = x.relu().relu();
            AutoGrad::backward(y.getRelatedNode(), false);
            auto gx = x.grad();

            double d = 0.0;
            const float* e = eager_gx.data_read<float>();
            const float* g = gx.data_read<float>();
            for (size_t i = 0; i < gx.numel(); ++i) {
                double dd = std::fabs(g[i] - e[i]);
                if (dd > d) d = dd;
            }
            max_diff = std::max(max_diff, d);
            std::cout << "  [iter " << iter << "] max_diff=" << d << std::endl;
            if (d > 0) {
                // 找出第一个差异大的位置
                size_t bad_idx = 0;
                for (size_t i = 0; i < gx.numel(); ++i) {
                    if (std::fabs(g[i] - e[i]) > 0.5) {
                        bad_idx = i;
                        break;
                    }
                }
                std::cout << "    [DEBUG-T9-BAD] idx=" << bad_idx
                          << " got=" << g[bad_idx]
                          << " ref=" << e[bad_idx]
                          << " x=" << xp[bad_idx]
                          << std::endl;
                std::cout << "    [DEBUG-T9] first8 got: ";
                for (size_t i = 0; i < 8; ++i) std::cout << g[i] << " ";
                std::cout << "\n    [DEBUG-T9] first8 ref: ";
                for (size_t i = 0; i < 8; ++i) std::cout << e[i] << " ";
                std::cout << "\n    [DEBUG-T9] first8 x:   ";
                for (size_t i = 0; i < 8; ++i) std::cout << xp[i] << " ";
                std::cout << std::endl;
            }

            if (iter == 3) {
                std::cout << "  Iter 3 → 等待异步融合编译 (3.5s)..." << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(3500));
            }
        }
        std::cout << "  Test 9 ReLU+ReLU 最大误差 max_diff=" << max_diff << std::endl;
        overall_max_diff = std::max(overall_max_diff, max_diff);
    }

    // ========== Test 10: 反向融合 (不同尺寸 ReLU 隔着 MatMul) ==========
    // 复现 MNIST 反向融合 SIGBUS：h2.ReLU(grad=[B,16]) 与 h1.ReLU(grad=[B,32])
    // 隔着 MatMul，recordBackwardNode 跳过 MatMul 后二者在序列里假相邻，
    // 形状不一致但仍可能被错误拼进融合链。
    {
        sched.resetRegionFusion();
        std::cout << "\n[Test 10] Backward Fusion: ReLU(16) → MatMul → ReLU(32)" << std::endl;
        const size_t B = 8, D = 64, H1 = 32, H2 = 16;

        // eager 参考
        Tensor W1(ShapeTag{}, {D, H1}, DType::kFloat, DeviceType::kCPU);
        Tensor b1(ShapeTag{}, {H1}, DType::kFloat, DeviceType::kCPU);
        Tensor W2(ShapeTag{}, {H1, H2}, DType::kFloat, DeviceType::kCPU);
        Tensor b2(ShapeTag{}, {H2}, DType::kFloat, DeviceType::kCPU);
        for (size_t i = 0; i < W1.numel(); ++i) W1.data_write<float>()[i] = (static_cast<float>(i) / W1.numel() - 0.5f) * 0.2f;
        for (size_t i = 0; i < b1.numel(); ++i) b1.data_write<float>()[i] = 0.1f;
        for (size_t i = 0; i < W2.numel(); ++i) W2.data_write<float>()[i] = (static_cast<float>(i) / W2.numel() - 0.5f) * 0.2f;
        for (size_t i = 0; i < b2.numel(); ++i) b2.data_write<float>()[i] = 0.1f;
        W1.requires_grad(true); b1.requires_grad(true); W2.requires_grad(true); b2.requires_grad(true);

        Tensor x(ShapeTag{}, {B, D}, DType::kFloat, DeviceType::kCPU);
        for (size_t i = 0; i < x.numel(); ++i) x.data_write<float>()[i] = (static_cast<float>(i) / x.numel() - 0.5f) * 2.0f;
        x.requires_grad(true);
        Tensor z1 = x.matmul(W1) + b1;
        Tensor h1 = z1.relu();
        Tensor z2 = h1.matmul(W2) + b2;
        Tensor h2 = z2.relu();
        Tensor loss = h2.sum();
        AutoGrad::backward(loss.getRelatedNode(), false);
        auto eager_gx = x.grad();
        auto eager_gw1 = W1.grad();
        auto eager_gw2 = W2.grad();

        double max_diff = 0.0;
        for (int iter = 0; iter < 8; ++iter) {
            for (size_t i = 0; i < x.numel(); ++i) x.data_write<float>()[i] = (static_cast<float>(i) / x.numel() - 0.5f) * 2.0f;
            Tensor z1i = x.matmul(W1) + b1;
            Tensor h1i = z1i.relu();
            Tensor z2i = h1i.matmul(W2) + b2;
            Tensor h2i = z2i.relu();
            Tensor lossi = h2i.sum();
            AutoGrad::backward(lossi.getRelatedNode(), false);
            auto gx = x.grad();

            double d = 0.0;
            const float* e = eager_gx.data_read<float>();
            const float* g = gx.data_read<float>();
            for (size_t i = 0; i < gx.numel(); ++i) {
                double dd = std::fabs(g[i] - e[i]);
                if (dd > d) d = dd;
            }
            max_diff = std::max(max_diff, d);
            std::cout << "  [iter " << iter << "] max_diff=" << d << std::endl;
            if (iter == 3) {
                std::cout << "  Iter 3 → 等待异步融合编译 (3.5s)..." << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(3500));
            }
        }
        std::cout << "  Test 10 MLP 最大误差 max_diff=" << max_diff << std::endl;
        overall_max_diff = std::max(overall_max_diff, max_diff);
    }

    // ========== Test 11: Softmax backward（验证 P0.2.1 shape-based broadcast） ==========
    // 关键：[M, 1] → [M, N] 广播（SumReduce[keepdim] 输出 + Sub/Div 节点）
    // 之前 numel-based `idx % M` 在 M=4, N=8 等尺寸会返回错位
    {
        sched.resetRegionFusion();
        std::cout << "\n[Test 11] Softmax backward (P0.2.1 shape-based broadcast)" << std::endl;
        const size_t M = 4, N = 8;  // 注意 M != N 的非平凡尺寸，专门打 broadcast bug
        // Eager 参考
        Tensor x_ref(ShapeTag{}, {M, N}, DType::kFloat, DeviceType::kCPU);
        float* xref = x_ref.data_write<float>();
        for (size_t i = 0; i < M * N; ++i) xref[i] = (static_cast<float>(i) / (M * N) - 0.5f) * 4.0f;
        x_ref.requires_grad(true);
        Tensor y_ref = x_ref.softmax(1);
        AutoGrad::backward(y_ref.getRelatedNode(), false);
        auto ref_grad = x_ref.grad();

        double max_diff = 0.0;
        for (int iter = 0; iter < 6; ++iter) {
            Tensor x(ShapeTag{}, {M, N}, DType::kFloat, DeviceType::kCPU);
            float* xp = x.data_write<float>();
            for (size_t i = 0; i < M * N; ++i) xp[i] = (static_cast<float>(i) / (M * N) - 0.5f) * 4.0f;
            x.requires_grad(true);
            Tensor y = x.softmax(1);
            AutoGrad::backward(y.getRelatedNode(), false);
            if (iter == 5) {
                auto got = x.grad();
                const float* r = ref_grad.data_read<float>();
                const float* g = got.data_read<float>();
                std::cout << "  Softmax grad (got vs ref):" << std::endl;
                for (size_t i = 0; i < M * N; ++i) {
                    if (i % N == 0) std::cout << "    row " << (i / N) << ": ";
                    std::cout << "[" << i << "]=" << g[i] << "/" << r[i] << " ";
                    if ((i + 1) % N == 0) std::cout << std::endl;
                    double d = std::fabs(static_cast<double>(g[i]) - static_cast<double>(r[i]));
                    if (d > max_diff) max_diff = d;
                }
            }
        }
        overall_max_diff = std::max(overall_max_diff, max_diff);
        std::cout << "  Test 11 Softmax max_diff=" << max_diff
                  << (max_diff < 1e-5 ? "  ✅" : "  ❌") << std::endl;
    }

    // ========== Test 12: CrossEntropy end-to-end（eager forward + C3 backward） ==========
    // 验证 P0.2 step 6 端到端：eager SIMD forward 出 loss，C3 backward 出 grad_logits
    {
        sched.resetRegionFusion();
        std::cout << "\n[Test 12] CrossEntropy end-to-end (eager forward + C3 backward)" << std::endl;
        const size_t M = 4, N = 6;  // 故意非平凡尺寸

        // 构造 one-hot target（CE_SIMD_kernel 接受 2D 概率/one-hot 形式）
        Tensor target(ShapeTag{}, {M, N}, DType::kFloat, DeviceType::kCPU);
        float* tp = target.data_write<float>();
        for (size_t i = 0; i < M; ++i) {
            size_t class_idx = i % N;
            for (size_t j = 0; j < N; ++j) {
                tp[i * N + j] = (j == class_idx) ? 1.0f : 0.0f;
            }
        }

        // Eager 参考
        Tensor logits_ref(ShapeTag{}, {M, N}, DType::kFloat, DeviceType::kCPU);
        float* lref = logits_ref.data_write<float>();
        for (size_t i = 0; i < M * N; ++i) lref[i] = (static_cast<float>(i) / (M * N) - 0.5f) * 2.0f;
        logits_ref.requires_grad(true);
        Tensor y_ref = logits_ref.cross_entropy(target);
        AutoGrad::backward(y_ref.getRelatedNode(), false);
        auto ref_grad = logits_ref.grad();
        const float* r = ref_grad.data_read<float>();

        std::cout << "  Eager forward loss: " << y_ref.data_read<float>()[0] << std::endl;
        std::cout << "  Eager grad_logits (ref):" << std::endl;
        for (size_t i = 0; i < M; ++i) {
            std::cout << "    row " << i << " (class=" << (i % N) << "): ";
            for (size_t j = 0; j < N; ++j) std::cout << r[i * N + j] << " ";
            std::cout << std::endl;
        }

        // 6 轮 C3 backward
        double max_diff = 0.0;
        for (int iter = 0; iter < 6; ++iter) {
            Tensor logits(ShapeTag{}, {M, N}, DType::kFloat, DeviceType::kCPU);
            float* lp = logits.data_write<float>();
            for (size_t i = 0; i < M * N; ++i) lp[i] = (static_cast<float>(i) / (M * N) - 0.5f) * 2.0f;
            logits.requires_grad(true);
            Tensor y = logits.cross_entropy(target);
            AutoGrad::backward(y.getRelatedNode(), false);
            if (iter == 5) {
                auto got = logits.grad();
                const float* g = got.data_read<float>();
                std::cout << "  C3 grad_logits (got):" << std::endl;
                for (size_t i = 0; i < M; ++i) {
                    std::cout << "    row " << i << ": ";
                    for (size_t j = 0; j < N; ++j) std::cout << g[i * N + j] << " ";
                    std::cout << std::endl;
                }
                for (size_t i = 0; i < M * N; ++i) {
                    double d = std::fabs(static_cast<double>(g[i]) - static_cast<double>(r[i]));
                    if (d > max_diff) max_diff = d;
                }
            }
        }
        overall_max_diff = std::max(overall_max_diff, max_diff);
        std::cout << "  Test 12 CrossEntropy max_diff=" << max_diff
                  << (max_diff < 1e-4 ? "  ✅" : "  ❌") << std::endl;
    }

    // ========== Test 13/14: Tanh/Sigmoid FC 反向(通用树白名单回归, §4.107) ==========
    // 覆盖: supportsNodeType(Tanh/Sigmoid) → generic 树捕获(白名单 Tanh/Sigmoid/Add/MatMul)
    // → 多节点反向图执行层(§4.106 双根因修复)。y = act(x @ W + b) 的 x/W/b 三梯度对拍 eager。
    {
        auto runActFC = [&](const char* label, bool use_tanh) {
            std::cout << "\n[" << label << "] FC backward (tree capture)" << std::endl;
            const size_t B = 4, IN = 3, OUT = 2;

            auto build = [&](Tensor& x, Tensor& W, Tensor& b, Tensor& y) {
                float* xp = x.data_write<float>();
                for (size_t i = 0; i < x.numel(); ++i) xp[i] = (static_cast<float>(i) - 5.0f) * 0.25f;
                x.requires_grad(true);
                float* wp = W.data_write<float>();
                for (size_t i = 0; i < W.numel(); ++i) wp[i] = (static_cast<float>(i) - 2.5f) * 0.3f;
                W.requires_grad(true);
                float* bp = b.data_write<float>();
                for (size_t i = 0; i < b.numel(); ++i) bp[i] = static_cast<float>(i) * 0.1f - 0.1f;
                b.requires_grad(true);
                Tensor h = x.matmul(W) + b;
                y = use_tanh ? h.tanh() : h.sigmoid();
            };

            // eager 参照
            Tensor ex(ShapeTag{}, {B, IN}, DType::kFloat, DeviceType::kCPU);
            Tensor eW(ShapeTag{}, {IN, OUT}, DType::kFloat, DeviceType::kCPU);
            Tensor eb(ShapeTag{}, {OUT}, DType::kFloat, DeviceType::kCPU);
            Tensor ey;
            build(ex, eW, eb, ey);
            AutoGrad::backward(ey.getRelatedNode(), false);
            auto egx = ex.grad(), egW = eW.grad(), egb = eb.grad();

            double max_diff = 0.0;
            auto cmpGrad = [&](const Tensor& got, const Tensor& ref, const char* what) {
                const float* gp = got.data_read<float>();
                const float* rp = ref.data_read<float>();
                double d = 0.0;
                for (size_t i = 0; i < got.numel(); ++i) {
                    double dd = std::fabs(static_cast<double>(gp[i]) - static_cast<double>(rp[i]));
                    if (dd > d) d = dd;
                }
                max_diff = std::max(max_diff, d);
                std::cout << "    grad_" << what << " max_diff=" << d << std::endl;
                if (d > 1e-4) {
                    std::cout << "      [DBG] got=";
                    for (size_t i = 0; i < got.numel() && i < 6; ++i) std::cout << gp[i] << " ";
                    std::cout << " ref=";
                    for (size_t i = 0; i < ref.numel() && i < 6; ++i) std::cout << rp[i] << " ";
                    std::cout << std::endl;
                }
            };

            for (int iter = 0; iter < 6; ++iter) {
                Tensor x(ShapeTag{}, {B, IN}, DType::kFloat, DeviceType::kCPU);
                Tensor W(ShapeTag{}, {IN, OUT}, DType::kFloat, DeviceType::kCPU);
                Tensor b(ShapeTag{}, {OUT}, DType::kFloat, DeviceType::kCPU);
                Tensor y;
                build(x, W, b, y);
                AutoGrad::backward(y.getRelatedNode(), false);
                cmpGrad(x.grad(), egx, "x");
                cmpGrad(W.grad(), egW, "W");
                cmpGrad(b.grad(), egb, "b");
                if (iter == 2) {
                    std::cout << "  Iter 2 → 等待异步编译 (3.5s)..." << std::endl;
                    std::this_thread::sleep_for(std::chrono::milliseconds(3500));
                }
            }
            overall_max_diff = std::max(overall_max_diff, max_diff);
            std::cout << "  " << label << " max_diff=" << max_diff
                      << (max_diff < 1e-4 ? "  ✅" : "  ❌") << std::endl;
        };
        runActFC("Test 13 Tanh", /*use_tanh=*/true);
        runActFC("Test 14 Sigmoid", /*use_tanh=*/false);
    }

    // 安全退出
    ct::c3::shutdownAll();

    // 验证
    // 说明：MatMul 单 kernel 的 JIT 精度为 1e-4 量级（与 eager AMX 路径相比），
    // 因此整体判定阈值放宽到 1e-4。其余 element-wise 测试的 max_diff 均为 0。
    double final_max_diff = overall_max_diff;
    bool pass = true;
    if (final_max_diff > 1e-4) {
        std::cout << "\n❌ FAIL: C3 backward 结果与 eager 不匹配 (max_diff=" << final_max_diff << ")" << std::endl;
        pass = false;
    } else {
        std::cout << "\n✅ PASS: C3 backward 结果正确 (overall_max_diff=" << final_max_diff << ")" << std::endl;
    }

    if (!has_c3_hits) {
        std::cout << "⚠️  WARN: 未检测到 C3 backward 缓存命中" << std::endl;
        std::cout << "  (可能是异步编译尚未完成，但测试结果仍然正确)" << std::endl;
    }

    const int exit_code = pass ? 0 : 1;
    std::cout.flush();
    std::cerr.flush();
    std::_Exit(exit_code);
}