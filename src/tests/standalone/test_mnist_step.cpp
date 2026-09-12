#include "mnist/mnist_loader.h"
#include "AutoGrad.h"
#include "CtorchError.h"
#include "ctQALS/Random.h"
#include "src/kernels/kernels.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstring>

struct Params {
    Tensor W1, b1, W2, b2, W3, b3;
};

static void xavier_init(Tensor& W, ctQALS::rng::Xoshiro256PlusPlus& rng,
                        size_t fan_in, size_t fan_out) {
    float std = std::sqrt(2.0f / (fan_in + fan_out));
    float* data = W.data_write<float>();
    for (size_t i = 0; i < W.numel(); ++i) {
        float r = 2.0f * rng.uniform_f32() - 1.0f;
        data[i] = r * std;
    }
}

static Params create_params(DeviceType dev, ctQALS::rng::Xoshiro256PlusPlus& rng) {
    Params p;
    p.W1 = Tensor(ShapeTag{}, {784, 256}, DType::kFloat, dev);
    p.b1 = Tensor(ShapeTag{}, {256}, DType::kFloat, dev);
    p.W2 = Tensor(ShapeTag{}, {256, 128}, DType::kFloat, dev);
    p.b2 = Tensor(ShapeTag{}, {128}, DType::kFloat, dev);
    p.W3 = Tensor(ShapeTag{}, {128, 10}, DType::kFloat, dev);
    p.b3 = Tensor(ShapeTag{}, {10}, DType::kFloat, dev);

    p.W1.requires_grad(true); p.b1.requires_grad(true);
    p.W2.requires_grad(true); p.b2.requires_grad(true);
    p.W3.requires_grad(true); p.b3.requires_grad(true);

    xavier_init(p.W1, rng, 784, 256);
    xavier_init(p.W2, rng, 256, 128);
    xavier_init(p.W3, rng, 128, 10);
    p.b1.zero(); p.b2.zero(); p.b3.zero();
    return p;
}

static float tensor_l2(const Tensor& t);

static Tensor forward(const Params& p, const Tensor& x, bool print_dbg = false) {
    Tensor z1 = x.matmul(p.W1) + p.b1;
    Tensor h1 = z1.relu();
    Tensor z2 = h1.matmul(p.W2) + p.b2;
    Tensor h2 = z2.relu();
    Tensor logits = h2.matmul(p.W3) + p.b3;

    if (print_dbg && x.device() == DeviceType::kMPS) {
        MPS_flush_wait(true);
        std::cout << "[fwd MPS] z1 L2=" << tensor_l2(z1)
                  << " h1 L2=" << tensor_l2(h1)
                  << " z2 L2=" << tensor_l2(z2)
                  << " h2 L2=" << tensor_l2(h2)
                  << " logits L2=" << tensor_l2(logits) << std::endl;
    }
    return logits;
}

static Tensor make_one_hot(const Tensor& labels, DeviceType dev) {
    size_t batch = labels.numel();
    Tensor one_hot(ShapeTag{}, {batch, 10}, DType::kFloat, dev);
    std::memset(one_hot.data_write<float>(), 0, batch * 10 * sizeof(float));
    const float* lp = labels.data_read<float>();
    float* op = one_hot.data_write<float>();
    for (size_t i = 0; i < batch; ++i) {
        int lab = static_cast<int>(lp[i]);
        op[i * 10 + lab] = 1.0f;
    }
    return one_hot;
}

static float train_step(Params& p, const Tensor& x, const Tensor& y_onehot,
                        float lr, bool do_update, bool print_dbg = false) {
    Tensor logits = forward(p, x, print_dbg);
    Tensor loss = logits.cross_entropy(y_onehot);
    float loss_value = loss.item<float>();

    AutoGrad::backward(loss.getRelatedNode(), false);

    if (do_update) {
        if (p.W1.device() == DeviceType::kMPS) {
            MPS_flush_wait(true);
            MPS_update_begin();
            SGD_Step_Zero_MPS_kernel(p.W1, p.W1.grad(), lr);
            SGD_Step_Zero_MPS_kernel(p.b1, p.b1.grad(), lr);
            SGD_Step_Zero_MPS_kernel(p.W2, p.W2.grad(), lr);
            SGD_Step_Zero_MPS_kernel(p.b2, p.b2.grad(), lr);
            SGD_Step_Zero_MPS_kernel(p.W3, p.W3.grad(), lr);
            SGD_Step_Zero_MPS_kernel(p.b3, p.b3.grad(), lr);
            MPS_update_end();
        } else {
            auto sgd = [lr](Tensor& param) {
                float* pp = param.data_write<float>();
                const float* gp = param.grad().data_read<float>();
                for (size_t i = 0; i < param.numel(); ++i) {
                    pp[i] -= gp[i] * lr;
                }
            };
            sgd(p.W1); sgd(p.b1); sgd(p.W2); sgd(p.b2); sgd(p.W3); sgd(p.b3);

            p.W1.zero_grad(); p.b1.zero_grad();
            p.W2.zero_grad(); p.b2.zero_grad();
            p.W3.zero_grad(); p.b3.zero_grad();
        }
    }

    return loss_value;
}

static float tensor_l2(const Tensor& t) {
    const float* p = t.data_read<float>();
    float s = 0.0f;
    for (size_t i = 0; i < t.numel(); ++i) s += p[i] * p[i];
    return std::sqrt(s);
}

static bool compare_grad(const Tensor& cpu, const Tensor& mps, const char* name,
                         float tol_l2 = 1e-3f, float tol_max = 1e-3f) {
    Tensor mps_cpu = mps.to(DeviceType::kCPU);
    const float* cp = cpu.data_read<float>();
    const float* mp = mps_cpu.data_read<float>();
    float diff_l2 = 0.0f;
    float max_diff = 0.0f;
    for (size_t i = 0; i < cpu.numel(); ++i) {
        float d = std::fabs(cp[i] - mp[i]);
        diff_l2 += d * d;
        if (d > max_diff) max_diff = d;
    }
    diff_l2 = std::sqrt(diff_l2);
    bool ok = (diff_l2 <= tol_l2) && (max_diff <= tol_max);
    std::cout << std::setw(6) << name << " grad L2 cpu=" << tensor_l2(cpu)
              << " mps=" << tensor_l2(mps)
              << " diff_l2=" << diff_l2 << " max_diff=" << max_diff
              << " -> " << (ok ? "MATCH" : "MISMATCH") << std::endl;
    return ok;
}

int main() {
    CtorchError::setPrintLevel(PrintLevel::MINIUM);
    CtorchScheduler::getInstance();

    // 验证 Tensor 拷贝语义
    {
        Tensor a(ShapeTag{}, {4}, DType::kFloat, DeviceType::kCPU);
        float* ap = a.data_write<float>();
        for (size_t i = 0; i < 4; ++i) ap[i] = static_cast<float>(i + 1);
        Tensor b = a;  // 拷贝构造
        std::cout << "[copy test] a.data=" << a.data_read<float>()
                  << " b.data=" << b.data_read<float>()
                  << " same=" << (a.data_read<float>() == b.data_read<float>()) << std::endl;
    }

    MNISTLoader loader(".", DeviceType::kCPU);
    Tensor train_images, train_labels;
    loader.load_training_data(train_images, train_labels);

    const size_t batch_size = 128;
    Tensor batch_images_cpu(ShapeTag{}, {batch_size, 784}, DType::kFloat, DeviceType::kCPU);
    Tensor batch_labels_cpu(ShapeTag{}, {batch_size}, DType::kFloat, DeviceType::kCPU);
    std::memcpy(batch_images_cpu.data_write<float>(), train_images.data_read<float>(),
                batch_size * 784 * sizeof(float));
    std::memcpy(batch_labels_cpu.data_write<float>(), train_labels.data_read<float>(),
                batch_size * sizeof(float));

    Tensor batch_images_mps = batch_images_cpu.to(DeviceType::kMPS);
    Tensor batch_labels_mps = batch_labels_cpu.to(DeviceType::kMPS);
    batch_images_mps.requires_grad(false);
    batch_labels_mps.requires_grad(false);

    Tensor y_onehot_cpu = make_one_hot(batch_labels_cpu, DeviceType::kCPU);
    Tensor y_onehot_mps = make_one_hot(batch_labels_mps, DeviceType::kMPS);

    ctQALS::rng::Xoshiro256PlusPlus rng(42);
    Params cpu_p = create_params(DeviceType::kCPU, rng);

    Params mps_p;
    mps_p.W1 = cpu_p.W1.to(DeviceType::kMPS); mps_p.W1.requires_grad(true);
    mps_p.b1 = cpu_p.b1.to(DeviceType::kMPS); mps_p.b1.requires_grad(true);
    mps_p.W2 = cpu_p.W2.to(DeviceType::kMPS); mps_p.W2.requires_grad(true);
    mps_p.b2 = cpu_p.b2.to(DeviceType::kMPS); mps_p.b2.requires_grad(true);
    mps_p.W3 = cpu_p.W3.to(DeviceType::kMPS); mps_p.W3.requires_grad(true);
    mps_p.b3 = cpu_p.b3.to(DeviceType::kMPS); mps_p.b3.requires_grad(true);

    float lr = 0.128f;

    std::cout << "=== Single MNIST training step (CPU vs MPS) ===" << std::endl;

    float loss_cpu = train_step(cpu_p, batch_images_cpu, y_onehot_cpu, lr, false, true);
    float loss_mps = train_step(mps_p, batch_images_mps, y_onehot_mps, lr, false, true);

    std::cout << "CPU loss: " << std::setprecision(6) << loss_cpu << std::endl;
    std::cout << "MPS loss: " << std::setprecision(6) << loss_mps << std::endl;
    bool loss_ok = std::fabs(loss_cpu - loss_mps) < 1e-4f;
    std::cout << "Loss diff: " << std::fabs(loss_cpu - loss_mps)
              << " -> " << (loss_ok ? "MATCH" : "MISMATCH") << std::endl;

    // MPS 梯度写回后再读回比较
    MPS_flush_wait(true);

    std::cout << "\n=== Gradient comparison ===" << std::endl;
    bool all_ok = loss_ok;
    all_ok &= compare_grad(cpu_p.W1.grad(), mps_p.W1.grad(), "W1");
    all_ok &= compare_grad(cpu_p.b1.grad(), mps_p.b1.grad(), "b1");
    all_ok &= compare_grad(cpu_p.W2.grad(), mps_p.W2.grad(), "W2");
    all_ok &= compare_grad(cpu_p.b2.grad(), mps_p.b2.grad(), "b2");
    all_ok &= compare_grad(cpu_p.W3.grad(), mps_p.W3.grad(), "W3");
    all_ok &= compare_grad(cpu_p.b3.grad(), mps_p.b3.grad(), "b3");

    if (all_ok) {
        std::cout << "\nAll MNIST step checks passed." << std::endl;
        return 0;
    }
    std::cout << "\nMNIST step checks failed." << std::endl;
    return 1;
}
