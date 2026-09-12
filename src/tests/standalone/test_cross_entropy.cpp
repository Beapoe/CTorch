#include "Tensor.h"
#include "AutoGrad.h"
#include "CtorchError.h"
#include "src/kernels/kernels.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>

static bool near(float a, float b, float tol = 1e-4f) {
    return std::fabs(a - b) <= tol;
}

static float grad_l2(const Tensor& t) {
    const float* p = t.data_read<float>();
    float s = 0.0f;
    for (size_t i = 0; i < t.numel(); ++i) s += p[i] * p[i];
    return std::sqrt(s);
}

int main() {
    CtorchError::setPrintLevel(PrintLevel::MINIUM);

    const size_t batch = 2;
    const size_t classes = 5;

    Tensor logits_cpu(ShapeTag{}, {batch, classes}, DType::kFloat, DeviceType::kCPU);
    float* l = logits_cpu.data_write<float>();
    // 构造有区分度的 logits
    for (size_t i = 0; i < batch * classes; ++i) {
        l[i] = static_cast<float>(i) * 0.3f - 1.0f;
    }

    Tensor target_onehot_cpu(ShapeTag{}, {batch, classes}, DType::kFloat, DeviceType::kCPU);
    float* t1 = target_onehot_cpu.data_write<float>();
    for (size_t i = 0; i < batch * classes; ++i) t1[i] = 0.0f;
    t1[0 * classes + 2] = 1.0f;
    t1[1 * classes + 4] = 1.0f;

    Tensor target_index_cpu(ShapeTag{}, {batch}, DType::kFloat, DeviceType::kCPU);
    float* t2 = target_index_cpu.data_write<float>();
    t2[0] = 2.0f;
    t2[1] = 4.0f;

    std::cout << "=== CPU CrossEntropy ===" << std::endl;
    Tensor loss_cpu_oh = logits_cpu.cross_entropy(target_onehot_cpu);
    Tensor loss_cpu_idx = logits_cpu.cross_entropy(target_index_cpu);
    float v_cpu_oh = loss_cpu_oh.item<float>();
    float v_cpu_idx = loss_cpu_idx.item<float>();
    std::cout << "one-hot loss: " << std::setprecision(6) << v_cpu_oh << std::endl;
    std::cout << "index  loss: " << std::setprecision(6) << v_cpu_idx << std::endl;

    Tensor logits_mps = logits_cpu.to(DeviceType::kMPS);
    Tensor target_onehot_mps = target_onehot_cpu.to(DeviceType::kMPS);
    Tensor target_index_mps = target_index_cpu.to(DeviceType::kMPS);

    std::cout << "=== MPS CrossEntropy ===" << std::endl;
    Tensor loss_mps_oh = logits_mps.cross_entropy(target_onehot_mps);
    Tensor loss_mps_idx = logits_mps.cross_entropy(target_index_mps);
    float v_mps_oh = loss_mps_oh.item<float>();
    float v_mps_idx = loss_mps_idx.item<float>();
    std::cout << "one-hot loss: " << std::setprecision(6) << v_mps_oh << std::endl;
    std::cout << "index  loss: " << std::setprecision(6) << v_mps_idx << std::endl;

    std::cout << "=== Compare Forward ===" << std::endl;
    bool ok_oh = near(v_cpu_oh, v_mps_oh);
    bool ok_idx = near(v_cpu_idx, v_mps_idx);
    std::cout << "one-hot CPU vs MPS: " << (ok_oh ? "MATCH" : "MISMATCH") << std::endl;
    std::cout << "index  CPU vs MPS: " << (ok_idx ? "MATCH" : "MISMATCH") << std::endl;

    // ===== Softmax 测试 =====
    std::cout << "\n=== Softmax Forward ===" << std::endl;
    {
        Tensor softmax_cpu = logits_cpu.softmax(1);
        Tensor softmax_mps = logits_mps.softmax(1).to(DeviceType::kCPU);
        std::cout << "CPU softmax: ";
        for (size_t i = 0; i < softmax_cpu.numel(); ++i) std::cout << softmax_cpu.data_read<float>()[i] << " ";
        std::cout << std::endl;
        std::cout << "MPS softmax: ";
        for (size_t i = 0; i < softmax_mps.numel(); ++i) std::cout << softmax_mps.data_read<float>()[i] << " ";
        std::cout << std::endl;
        bool softmax_ok = true;
        for (size_t i = 0; i < softmax_cpu.numel(); ++i) {
            if (!near(softmax_cpu.data_read<float>()[i], softmax_mps.data_read<float>()[i], 1e-3f)) softmax_ok = false;
        }
        std::cout << "Softmax CPU vs MPS: " << (softmax_ok ? "MATCH" : "MISMATCH") << std::endl;
        ok_oh = ok_oh && softmax_ok;
    }

    // ===== 手动模拟 CrossEntropyNode backward =====
    std::cout << "\n=== Manual CrossEntropy Backward (no autograd) ===" << std::endl;
    {
        Tensor softmax_cpu = logits_cpu.softmax(1);
        Tensor diff_cpu = softmax_cpu - target_onehot_cpu;
        Tensor grad_cpu = Tensor(1.0f);
        Tensor grad_logits_cpu = grad_cpu * diff_cpu;

        Tensor softmax_mps = logits_mps.softmax(1);
        Tensor diff_mps = softmax_mps - target_onehot_mps;
        MPS_flush_wait(true);
        Tensor diff_mps_cpu = diff_mps.to(DeviceType::kCPU);
        Tensor grad_mps = Tensor(1.0f, DeviceType::kMPS);
        Tensor grad_logits_mps = grad_mps * diff_mps;
        MPS_flush_wait(true);
        Tensor grad_logits_mps_cpu = grad_logits_mps.to(DeviceType::kCPU);

        std::cout << "MPS diff: ";
        for (size_t i = 0; i < diff_mps_cpu.numel(); ++i) std::cout << diff_mps_cpu.data_read<float>()[i] << " ";
        std::cout << std::endl;
        std::cout << "MPS scalar grad: " << grad_mps.item<float>() << std::endl;

        std::cout << "CPU manual grad: ";
        for (size_t i = 0; i < grad_logits_cpu.numel(); ++i) std::cout << grad_logits_cpu.data_read<float>()[i] << " ";
        std::cout << std::endl;
        std::cout << "MPS manual grad: ";
        for (size_t i = 0; i < grad_logits_mps_cpu.numel(); ++i) std::cout << grad_logits_mps_cpu.data_read<float>()[i] << " ";
        std::cout << std::endl;

        bool manual_ok = true;
        for (size_t i = 0; i < grad_logits_cpu.numel(); ++i) {
            if (!near(grad_logits_cpu.data_read<float>()[i], grad_logits_mps_cpu.data_read<float>()[i], 1e-3f)) manual_ok = false;
        }
        std::cout << "Manual backward CPU vs MPS: " << (manual_ok ? "MATCH" : "MISMATCH") << std::endl;
        ok_oh = ok_oh && manual_ok;
    }

    // ===== Backward 测试（one-hot） =====
    std::cout << "\n=== CPU CrossEntropy Backward ===" << std::endl;
    {
        Tensor logits_cpu_grad = logits_cpu.clone();
        logits_cpu_grad.requires_grad(true);
        Tensor loss_cpu_grad = logits_cpu_grad.cross_entropy(target_onehot_cpu);
        AutoGrad::backward(loss_cpu_grad.getRelatedNode(), false);
        Tensor g_cpu = logits_cpu_grad.grad();
        std::cout << "CPU grad L2: " << grad_l2(g_cpu) << std::endl;
        std::cout << "CPU grad: ";
        for (size_t i = 0; i < g_cpu.numel(); ++i) std::cout << g_cpu.data_read<float>()[i] << " ";
        std::cout << std::endl;

        Tensor logits_mps_grad = logits_cpu.clone().to(DeviceType::kMPS);
        logits_mps_grad.requires_grad(true);
        Tensor loss_mps_grad = logits_mps_grad.cross_entropy(target_onehot_mps);
        AutoGrad::backward(loss_mps_grad.getRelatedNode(), false);
        MPS_flush_wait(true);
        Tensor raw_g_mps = logits_mps_grad.grad();
        std::cout << "MPS raw grad device: " << static_cast<int>(raw_g_mps.device())
                  << " numel: " << raw_g_mps.numel()
                  << " ptr: " << raw_g_mps.storage().data<float>() << std::endl;
        if (raw_g_mps.numel() > 0 && raw_g_mps.storage().data<float>()) {
            std::cout << "MPS raw grad first (before to): " << raw_g_mps.storage().data<float>()[raw_g_mps.storage_offset()] << std::endl;
        }
        Tensor g_mps = raw_g_mps.to(DeviceType::kCPU);
        std::cout << "MPS grad L2: " << grad_l2(g_mps) << std::endl;
        std::cout << "MPS grad: ";
        for (size_t i = 0; i < g_mps.numel(); ++i) std::cout << g_mps.data_read<float>()[i] << " ";
        std::cout << std::endl;

        bool grad_ok = true;
        for (size_t i = 0; i < g_cpu.numel(); ++i) {
            if (!near(g_cpu.data_read<float>()[i], g_mps.data_read<float>()[i], 1e-3f)) grad_ok = false;
        }
        std::cout << "Gradient CPU vs MPS: " << (grad_ok ? "MATCH" : "MISMATCH") << std::endl;
        ok_oh = ok_oh && grad_ok;
    }

    // [Fix 2026-09-10 §4.95 P1-02] 解析梯度校验: CE(mean) 反向必须 =
    // (softmax(logits)-target)/batch。此前缺 1/N(梯度被放大 batch 倍), 且双实现
    // 数值对拍无法发现(两侧一致)。本段为独立数学参照(手工 softmax + 1/N)。
    {
        AutoGrad::EnableGrad = true;
        Tensor lg(ShapeTag{}, {batch, classes}, DType::kFloat, DeviceType::kCPU);
        float* lp = lg.data_write<float>();
        for (size_t i = 0; i < batch * classes; ++i) lp[i] = static_cast<float>(i) * 0.3f - 1.0f;
        lg.requires_grad(true);

        Tensor loss = lg.cross_entropy(target_onehot_cpu);
        AutoGrad::backward(loss.getRelatedNode(), false);
        MPS_flush_wait(true);
        Tensor g = lg.grad();

        bool analytic_ok = (g.numel() == batch * classes);
        if (analytic_ok) {
            const float* gp = g.data_read<float>();
            const float* tp = target_onehot_cpu.data_read<float>();
            for (size_t i = 0; i < batch; ++i) {
                float mx = -1e30f;
                for (size_t j = 0; j < classes; ++j) {
                    float v = lp[i * classes + j];
                    if (v > mx) mx = v;
                }
                float sum = 0.0f;
                for (size_t j = 0; j < classes; ++j) sum += std::exp(lp[i * classes + j] - mx);
                for (size_t j = 0; j < classes; ++j) {
                    float sm = std::exp(lp[i * classes + j] - mx) / sum;
                    float ref = (sm - tp[i * classes + j]) / static_cast<float>(batch);
                    if (!near(gp[i * classes + j], ref, 1e-4f)) analytic_ok = false;
                }
            }
        }
        std::cout << "CE analytic grad (softmax-target)/N: "
                  << (analytic_ok ? "MATCH" : "MISMATCH") << std::endl;
        ok_oh = ok_oh && analytic_ok;
    }

    if (ok_oh && ok_idx) {
        std::cout << "\nAll CrossEntropy checks passed." << std::endl;
        return 0;
    }
    std::cout << "\nCrossEntropy checks failed." << std::endl;
    return 1;
}
