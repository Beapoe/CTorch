/**
 * @file exp_mnist_cold_start.cpp
 * @brief 真实用户冷启动实验：纯净 MNIST 训练，代码中不出现任何 C3 API / include。
 *        清空 JIT cache 后直接运行，完全依赖调度器自行检测热路径、异步编译、
 *        运行时替换 JIT kernel。模拟"新用户第一次打开应用跑训练"的体验。
 *
 * 网络: 784 → 256(ReLU) → 128(ReLU) → 10
 * 训练: 2 epochs, batch=128, lr=0.001
 * 输出: 逐 batch 耗时 (ms) + 逐 epoch 总耗时 + loss/accuracy
 */
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

#include "Tensor.h"
#include "CtorchScheduler.h"
#include "AutoGrad.h"
#include "Ctools.h"
#include "mnist/mnist_loader.h"
#include "ctQALS/Random.h"

using namespace ct;

static constexpr size_t BATCH_SIZE = 128;
static constexpr size_t HIDDEN1    = 256;
static constexpr size_t HIDDEN2    = 128;
static constexpr float  LR         = 0.128f;
static constexpr int    EPOCHS     = 2;

static inline double now_ms() {
    using clock = std::chrono::high_resolution_clock;
    auto t = clock::now().time_since_epoch();
    return std::chrono::duration<double, std::milli>(t).count();
}

static ctQALS::rng::Xoshiro256PlusPlus g_rng(42);

static void xavierInit(Tensor& W, size_t fan_in, size_t fan_out) {
    float std = std::sqrt(2.0f / (fan_in + fan_out));
    float* data = W.data_write<float>();
    for (size_t i = 0; i < W.numel(); ++i) {
        float r = 2.0f * g_rng.uniform_f32() - 1.0f;
        data[i] = r * std;
    }
}
static void initParams(Tensor params[6]) {
    xavierInit(params[0], 784, 256);
    params[1].zero();
    xavierInit(params[2], 256, 128);
    params[3].zero();
    xavierInit(params[4], 128, 10);
    params[5].zero();
}
static float accuracy(const Tensor& logits, const Tensor& labels) {
    const float* l = logits.data_read<float>();
    const float* y = labels.data_read<float>();
    size_t b = logits.shape()[0];
    size_t c = 0;
    for (size_t i = 0; i < b; ++i) {
        size_t pred = 0;
        float m = l[i * 10];
        for (size_t j = 1; j < 10; ++j)
            if (l[i * 10 + j] > m) { m = l[i * 10 + j]; pred = j; }
        if (pred == (size_t)y[i]) ++c;
    }
    return (float)c / b;
}

/**
 * @brief 训练一个 epoch；逐 batch 把 wall-clock 写入 batch_ms（用户侧真实体感延迟）
 */
static double trainEpoch(Tensor params[6],
                         const Tensor& images, const Tensor& labels,
                         int num_batches, float* out_loss,
                         std::vector<double>* batch_ms,
                         std::vector<float>* batch_acc) {
    double t0 = now_ms();
    float total_loss = 0.0f;
    for (int b = 0; b < num_batches; ++b) {
        int start = b * (int)BATCH_SIZE;
        int end = std::min(start + (int)BATCH_SIZE, (int)images.shape()[0]);
        int actual_bs = end - start;

        Tensor bx(ShapeTag{}, {(size_t)actual_bs, 784}, DType::kFloat, DeviceType::kCPU);
        std::memcpy(bx.data_write<float>(), images.data_read<float>() + start * 784,
                    (size_t)actual_bs * 784 * sizeof(float));
        Tensor by(ShapeTag{}, {(size_t)actual_bs}, DType::kFloat, DeviceType::kCPU);
        std::memcpy(by.data_write<float>(), labels.data_read<float>() + start,
                    (size_t)actual_bs * sizeof(float));

        double bt0 = now_ms();

        // —— 纯标准 API：用户不会知道底下有 C3 ——
        Tensor z1 = bx.matmul(params[0]) + params[1];
        Tensor h1 = z1.relu();
        Tensor z2 = h1.matmul(params[2]) + params[3];
        Tensor h2 = z2.relu();
        Tensor logits = h2.matmul(params[4]) + params[5];

        Tensor one_hot(ShapeTag{}, {(size_t)actual_bs, 10}, DType::kFloat, DeviceType::kCPU);
        one_hot.zero();
        const float* yd = by.data_read<float>();
        float* oh = one_hot.data_write<float>();
        for (int i = 0; i < actual_bs; ++i) oh[i * 10 + (int)yd[i]] = 1.0f;

        Tensor loss = logits.cross_entropy(one_hot);
        float loss_val = loss.item<float>();
        AutoGrad::backward(loss.getRelatedNode(), false);
        // —— end of 标准 API ——

        double bt1 = now_ms();
        double bms = bt1 - bt0;

        total_loss += loss_val;
        if (batch_ms)  batch_ms->push_back(bms);
        if (batch_acc) batch_acc->push_back(accuracy(logits, by));

        // SGD
        auto sgd = [](Tensor& p) {
            float* gp = p.grad_ptr();
            float* pd = p.data_write<float>();
            for (size_t i = 0; i < p.numel(); ++i) pd[i] -= gp[i] * LR;
        };
        sgd(params[0]); sgd(params[1]); sgd(params[2]);
        sgd(params[3]); sgd(params[4]); sgd(params[5]);
        params[0].zero_grad(); params[1].zero_grad(); params[2].zero_grad();
        params[3].zero_grad(); params[4].zero_grad(); params[5].zero_grad();
    }
    *out_loss = total_loss / num_batches;
    return now_ms() - t0;
}

int main() {
    CtorchScheduler::getInstance();

    std::cout << "==== Cold-Start MNIST Experiment ====" << std::endl;
    std::cout << "  Net: 784→" << HIDDEN1 << "(ReLU)→" << HIDDEN2 << "(ReLU)→10" << std::endl;
    std::cout << "  Epochs: " << EPOCHS << "  Batch: " << BATCH_SIZE << "  LR: " << LR << std::endl;
    std::cout << "  说明：代码中不调用任何 C3 API，由调度器自动决策" << std::endl;

    MNISTLoader loader(".", DeviceType::kCPU);
    Tensor train_images, train_labels;
    loader.load_training_data(train_images, train_labels);
    const int N_SAMPLES = (int)train_images.shape()[0];
    const int NUM_BATCHES = N_SAMPLES / (int)BATCH_SIZE;
    std::cout << "  训练集: " << N_SAMPLES << " samples  |  " << NUM_BATCHES << " batches/epoch" << std::endl;

    Tensor params[6] = {
        Tensor(ShapeTag{}, {784, HIDDEN1}), Tensor(ShapeTag{}, {HIDDEN1}),
        Tensor(ShapeTag{}, {HIDDEN1, HIDDEN2}), Tensor(ShapeTag{}, {HIDDEN2}),
        Tensor(ShapeTag{}, {HIDDEN2, 10}),  Tensor(ShapeTag{}, {10})
    };
    initParams(params);
    for (int i = 0; i < 6; ++i) params[i].requires_grad(true);

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        std::vector<double> batch_ms;
        std::vector<float>  batch_acc;
        batch_ms.reserve(NUM_BATCHES);
        batch_acc.reserve(NUM_BATCHES);
        float epoch_loss = 0;
        double epoch_ms = trainEpoch(params, train_images, train_labels,
                                     NUM_BATCHES, &epoch_loss, &batch_ms, &batch_acc);

        // 逐 batch 输出（前 40 个 batch 一定包含异步编译触发点；后 20 个看稳态）
        std::cout << "\n  --- Epoch " << (epoch+1) << "/" << EPOCHS
                  << " | loss=" << std::fixed << std::setprecision(4) << epoch_loss
                  << " | total=" << std::setprecision(1) << epoch_ms << "ms" << " ---" << std::endl;
        std::cout << "    [逐 batch 耗时 ms, 前 40 + 后 20]:" << std::endl;
        auto format_ms = [](double v){ char buf[32]; std::snprintf(buf, 32, "%7.2f", v); return std::string(buf); };
        int show_first = std::min(40, (int)batch_ms.size());
        int show_last  = std::min(20, (int)batch_ms.size() - show_first);
        for (int i = 0; i < show_first; ++i) {
            if (i % 8 == 0) std::cout << "    b" << std::setw(3) << i << "-" << std::setw(3)
                                      << std::min(i+7, show_first-1) << ":";
            std::cout << " " << format_ms(batch_ms[i]);
            // 给用户体感标记：> 2 倍 p10 视为尖峰
            if (batch_ms[i] > 50.0) std::cout << "*";
            if ((i + 1) % 8 == 0) std::cout << std::endl;
        }
        if (show_first % 8 != 0) std::cout << std::endl;
        if (show_last > 0) {
            std::cout << "    (尾部 " << show_last << " batches):" << std::endl;
            int start = (int)batch_ms.size() - show_last;
            for (int k = 0; k < show_last; ++k) {
                int i = start + k;
                if (k % 8 == 0) std::cout << "    b" << std::setw(3) << i << "-"
                                          << std::setw(3) << std::min(i+7, (int)batch_ms.size()-1) << ":";
                std::cout << " " << format_ms(batch_ms[i]);
                if ((k + 1) % 8 == 0) std::cout << std::endl;
            }
            if (show_last % 8 != 0) std::cout << std::endl;
        }

        // 统计汇总
        std::sort(batch_ms.begin(), batch_ms.end());
        double p10 = batch_ms[(size_t)(0.10 * (batch_ms.size() - 1))];
        double p50 = batch_ms[(size_t)(0.50 * (batch_ms.size() - 1))];
        double p95 = batch_ms[(size_t)(0.95 * (batch_ms.size() - 1))];
        double peak = batch_ms.back();
        double sum = 0; for (double v : batch_ms) sum += v;
        double mean = sum / batch_ms.size();
        // 尖峰数量：> 3×p50 算抖动
        int spikes = 0;
        for (double v : batch_ms) if (v > p50 * 3.0) spikes++;
        float avg_acc = 0; for (float a : batch_acc) avg_acc += a; avg_acc /= batch_acc.size();
        std::cout << "    Batch latency stats (ms):  mean=" << std::fixed << std::setprecision(2)
                  << mean << "  p10=" << p10 << "  p50=" << p50 << "  p95=" << p95
                  << "  peak=" << peak << std::endl;
        std::cout << "    Spikes (>=3×p50): " << spikes << "/" << batch_ms.size() << std::endl;
        std::cout << "    Avg train accuracy: " << std::setprecision(2) << avg_acc * 100 << "%" << std::endl;
    }
    std::cout << "\nDone." << std::endl;
    return 0;
}
