// 微型诊断：跑 3 step，对比 logits/grad 数值
#include <cstdio>
#include <cstring>
#include <vector>
#include <cmath>
#include <cstdlib>
#include "Tensor.h"
#include "CtorchScheduler.h"
#include "AutoGrad.h"
#include "C3/C3Engine.h"
#include "C3/C3KernelRegistry.h"
#include "kernels/kernels.h"

using namespace ct;
using namespace ct::c3;

int main() {
    CtorchError::setPrintLevel(PrintLevel::MINIUM);
    CtorchScheduler::getInstance();
    auto& engine = C3Engine::getInstance();

    const int BS = 128;
    Tensor x(ShapeTag{}, {BS, 784}, DType::kFloat, DeviceType::kCPU);
    float* xp = x.data_write<float>();
    for (size_t i = 0; i < x.numel(); ++i) xp[i] = (i / 784.0f - 0.5f) * 0.5f;
    x.requires_grad(true);

    Tensor W1(ShapeTag{}, {784, 256}, DType::kFloat, DeviceType::kCPU);
    Tensor b1(ShapeTag{}, {256}, DType::kFloat, DeviceType::kCPU);
    Tensor W2(ShapeTag{}, {256, 128}, DType::kFloat, DeviceType::kCPU);
    Tensor b2(ShapeTag{}, {128}, DType::kFloat, DeviceType::kCPU);
    Tensor W3(ShapeTag{}, {128, 10}, DType::kFloat, DeviceType::kCPU);
    Tensor b3(ShapeTag{}, {10}, DType::kFloat, DeviceType::kCPU);

    auto xavierInit = [](Tensor& t, int fan_in, int fan_out) {
        float* p = t.data_write<float>();
        float scale = std::sqrt(2.0f / (fan_in + fan_out));
        for (size_t i = 0; i < t.numel(); ++i) p[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
    };
    srand(42);
    xavierInit(W1, 784, 256);
    xavierInit(W2, 256, 128);
    xavierInit(W3, 128, 10);
    b1.zero(); b2.zero(); b3.zero();

    Tensor params[6] = {W1, b1, W2, b2, W3, b3};
    for (auto& p : params) p.requires_grad(true);

    Tensor one_hot(ShapeTag{}, {BS, 10}, DType::kFloat, DeviceType::kCPU);
    one_hot.zero();
    float* oh = one_hot.data_write<float>();
    for (int i = 0; i < BS; ++i) oh[i * 10 + (i % 10)] = 1.0f;

    const float LR = 0.128f;

    for (int step = 0; step < 1; ++step) {
        Tensor z1 = x.matmul(W1) + b1;
        Tensor h1 = z1.relu();
        Tensor z2 = h1.matmul(W2) + b2;
        Tensor h2 = z2.relu();
        Tensor logits = h2.matmul(W3) + b3;
        Tensor loss = logits.cross_entropy(one_hot);
        float loss_val = loss.item<float>();

        const float* l = logits.data_read<float>();
        fprintf(stderr, "step %d loss=%.6f logits[0:3]=%.6f %.6f %.6f\n",
                step, loss_val, l[0], l[1], l[2]);

        AutoGrad::backward(loss.getRelatedNode(), false);

        const float* g = W1.grad_ptr();
        double sum = 0;
        for (size_t i = 0; i < 5; ++i) sum += g[i];
        fprintf(stderr, "  W1.grad[0:5] sum=%.6e\n", sum);

        for (auto& p : params) {
            float* gp = p.grad_ptr();
            float* pd = p.data_write<float>();
            for (size_t i = 0; i < p.numel(); ++i) pd[i] -= gp[i] * LR;
        }
    }

    // 跑 step 1 看 weight update 后的 logits/grad
    {
        Tensor z1 = x.matmul(W1) + b1;
        Tensor h1 = z1.relu();
        Tensor z2 = h1.matmul(W2) + b2;
        Tensor h2 = z2.relu();
        Tensor logits = h2.matmul(W3) + b3;
        Tensor loss = logits.cross_entropy(one_hot);
        float loss_val = loss.item<float>();

        const float* l = logits.data_read<float>();
        fprintf(stderr, "step 1 (after 1 weight update) loss=%.6f logits[0:3]=%.6f %.6f %.6f\n",
                loss_val, l[0], l[1], l[2]);

        AutoGrad::backward(loss.getRelatedNode(), false);

        const float* g = W1.grad_ptr();
        double sum = 0;
        for (size_t i = 0; i < 5; ++i) sum += g[i];
        fprintf(stderr, "  W1.grad[0:5] sum=%.6e\n", sum);
    }

    return 0;
}
