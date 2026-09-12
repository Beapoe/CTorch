/**
 * @file test_sum_mean_grad.cpp
 * @brief sum()/mean() 反向传播梯度回归测试
 * @details 修复 2026-09-06: sum() 经 dot(ones) 挂 DotNode(从未实现) / mean() 裸循环无节点,
 *          两者 backward 均静默断链。修复后 sum()/mean() 挂 SumNode/MeanNode。
 *          本测试验证: 全 reduce sum/mean 的梯度正确广播、链式传播正常。
 * @date 2026-09-06
 */
#include <cmath>
#include <cstdlib>
#include <iostream>
#include "Tensor.h"
#include "AutoGrad.h"

using namespace ct;

static int g_fails = 0;
#define CHECK(cond, msg)                                    \
    do {                                                    \
        if (!(cond)) {                                      \
            std::cerr << "FAIL: " << msg << "\n";           \
            ++g_fails;                                      \
        } else {                                            \
            std::cout << "PASS: " << msg << "\n";           \
        }                                                   \
    } while (0)

int main() {
    // Test 1: sum 前向值 + 梯度全 1
    {
        Tensor x(ShapeTag{}, {3, 4}, DType::kFloat, DeviceType::kCPU);
        float* p = x.data_write<float>();
        for (int i = 0; i < 12; ++i) p[i] = i * 0.5f;  // sum = 0..5.5 = 33
        x.requires_grad(true);
        Tensor l = x.sum();
        CHECK(std::abs(l.item<float>() - 33.0f) < 1e-4f, "sum 前向值 = 33");
        AutoGrad::backward(l.getRelatedNode(), false);
        const float* g = x.grad_ptr();
        CHECK(g != nullptr, "sum backward 填了梯度");
        bool ok = (g != nullptr);
        for (int i = 0; ok && i < 12; ++i) ok = (g[i] == 1.0f);
        CHECK(ok, "sum 梯度全 1");
    }

    // Test 2: mean 梯度全 1/n
    {
        Tensor x(ShapeTag{}, {3, 4}, DType::kFloat, DeviceType::kCPU);
        float* p = x.data_write<float>();
        for (int i = 0; i < 12; ++i) p[i] = 1.0f + i;
        x.requires_grad(true);
        Tensor l = x.mean();
        AutoGrad::backward(l.getRelatedNode(), false);
        const float* g = x.grad_ptr();
        CHECK(g != nullptr, "mean backward 填了梯度");
        bool ok = (g != nullptr);
        for (int i = 0; ok && i < 12; ++i) ok = (std::abs(g[i] - 1.0f / 12.0f) < 1e-6f);
        CHECK(ok, "mean 梯度全 1/12");
    }

    // Test 3: 链式 relu→sum, 梯度 = relu mask
    {
        Tensor x(ShapeTag{}, {2, 3}, DType::kFloat, DeviceType::kCPU);
        float* p = x.data_write<float>();
        const float vals[6] = {-2.0f, -0.5f, 0.3f, 1.5f, -1.0f, 2.0f};
        for (int i = 0; i < 6; ++i) p[i] = vals[i];
        x.requires_grad(true);
        Tensor y = x.relu();
        Tensor l = y.sum();
        AutoGrad::backward(l.getRelatedNode(), false);
        const float* g = x.grad_ptr();
        CHECK(g != nullptr, "链式 relu→sum 梯度存在");
        bool ok = (g != nullptr);
        for (int i = 0; ok && i < 6; ++i) {
            float expect = (vals[i] > 0.0f) ? 1.0f : 0.0f;
            if (std::abs(g[i] - expect) > 1e-6f) ok = false;
        }
        CHECK(ok, "relu→sum 梯度 = relu mask");
    }

    // Test 4: sum 作 loss 后参数 SGD 更新真实发生(grad 非空且可更新)
    {
        Tensor w(ShapeTag{}, {4}, DType::kFloat, DeviceType::kCPU);
        float* wp = w.data_write<float>();
        for (int i = 0; i < 4; ++i) wp[i] = 1.0f;
        w.requires_grad(true);
        Tensor x(ShapeTag{}, {4}, DType::kFloat, DeviceType::kCPU);
        float* xp = x.data_write<float>();
        for (int i = 0; i < 4; ++i) xp[i] = 2.0f;
        Tensor out = x * w;          // MulNode, 梯度流到 w
        Tensor l = out.sum();
        AutoGrad::backward(l.getRelatedNode(), false);
        float* gp = w.grad_ptr();
        CHECK(gp != nullptr, "w 收到梯度(x=2 全 → grad 全 2)");
        bool ok = (gp != nullptr);
        for (int i = 0; ok && i < 4; ++i) ok = (std::abs(gp[i] - 2.0f) < 1e-6f);
        CHECK(ok, "w 梯度全 2");
    }

    // Test 5: sum(dim=1) 前向值 + 梯度全 1
    {
        Tensor x(ShapeTag{}, {2, 3}, DType::kFloat, DeviceType::kCPU);
        float* p = x.data_write<float>();
        for (int i = 0; i < 6; ++i) p[i] = i + 1.0f;  // [[1,2,3],[4,5,6]]
        x.requires_grad(true);
        Tensor l = x.sum(1);
        CHECK(l.numel() == 2 && std::abs(l.data_read<float>()[0] - 6.0f) < 1e-5f &&
                  std::abs(l.data_read<float>()[1] - 15.0f) < 1e-5f,
              "sum(dim=1) 前向 = [6,15]");
        AutoGrad::backward(l.getRelatedNode(), false);
        const float* g = x.grad_ptr();
        bool ok = (g != nullptr);
        for (int i = 0; ok && i < 6; ++i) ok = (g[i] == 1.0f);
        CHECK(ok, "sum(dim=1) 梯度全 1");
    }

    // Test 6: mean(dim=0) 前向值 + 梯度全 1/n
    {
        Tensor x(ShapeTag{}, {2, 3}, DType::kFloat, DeviceType::kCPU);
        float* p = x.data_write<float>();
        for (int i = 0; i < 6; ++i) p[i] = i + 1.0f;  // [[1,2,3],[4,5,6]]
        x.requires_grad(true);
        Tensor l = x.mean(0);
        const float* lv = l.data_read<float>();
        CHECK(l.numel() == 3 && std::abs(lv[0] - 2.5f) < 1e-5f && std::abs(lv[1] - 3.5f) < 1e-5f &&
                  std::abs(lv[2] - 4.5f) < 1e-5f,
              "mean(dim=0) 前向 = [2.5,3.5,4.5]");
        AutoGrad::backward(l.getRelatedNode(), false);
        const float* g = x.grad_ptr();
        bool ok = (g != nullptr);
        for (int i = 0; ok && i < 6; ++i) ok = (std::abs(g[i] - 0.5f) < 1e-6f);
        CHECK(ok, "mean(dim=0) 梯度全 1/2");
    }

    // Test 7: 链式 relu→sum(dim=1), 梯度 = relu mask
    {
        Tensor x(ShapeTag{}, {2, 3}, DType::kFloat, DeviceType::kCPU);
        float* p = x.data_write<float>();
        const float vals[6] = {-2.0f, 0.5f, 1.0f, 3.0f, -1.0f, 0.0f};
        for (int i = 0; i < 6; ++i) p[i] = vals[i];
        x.requires_grad(true);
        Tensor y = x.relu();
        Tensor l = y.sum(1);
        AutoGrad::backward(l.getRelatedNode(), false);
        const float* g = x.grad_ptr();
        bool ok = (g != nullptr);
        for (int i = 0; ok && i < 6; ++i) {
            float expect = (vals[i] > 0.0f) ? 1.0f : 0.0f;
            if (std::abs(g[i] - expect) > 1e-6f) ok = false;
        }
        CHECK(ok, "relu→sum(dim=1) 梯度 = relu mask");
    }

    // Test 8: sum(dims={0,1}) 前向 = 全和, 梯度全 1 (多级 DimReduceNode 链)
    {
        Tensor x(ShapeTag{}, {2, 3}, DType::kFloat, DeviceType::kCPU);
        float* p = x.data_write<float>();
        for (int i = 0; i < 6; ++i) p[i] = i + 1.0f;  // 总和 21
        x.requires_grad(true);
        Tensor l = x.sum(std::vector<int>{0, 1});
        CHECK(std::abs(l.item<float>() - 21.0f) < 1e-5f, "sum({0,1}) 前向 = 21");
        AutoGrad::backward(l.getRelatedNode(), false);
        const float* g = x.grad_ptr();
        bool ok = (g != nullptr);
        for (int i = 0; ok && i < 6; ++i) ok = (g[i] == 1.0f);
        CHECK(ok, "sum({0,1}) 梯度全 1");
    }

    // Test 9: mean(dims={0,1}) 前向 = 全均值, 梯度全 1/6
    {
        Tensor x(ShapeTag{}, {2, 3}, DType::kFloat, DeviceType::kCPU);
        float* p = x.data_write<float>();
        for (int i = 0; i < 6; ++i) p[i] = i + 1.0f;  // 均值 3.5
        x.requires_grad(true);
        Tensor l = x.mean(std::vector<int>{0, 1});
        CHECK(std::abs(l.item<float>() - 3.5f) < 1e-5f, "mean({0,1}) 前向 = 3.5");
        AutoGrad::backward(l.getRelatedNode(), false);
        const float* g = x.grad_ptr();
        bool ok = (g != nullptr);
        for (int i = 0; ok && i < 6; ++i) ok = (std::abs(g[i] - 1.0f / 6.0f) < 1e-6f);
        CHECK(ok, "mean({0,1}) 梯度全 1/6");
    }

    // Test 10: 双输入都 requires_grad 时 mul→sum 是否断链(FFN bench 断链根因定位)
    {
        Tensor x(ShapeTag{}, {4}, DType::kFloat, DeviceType::kCPU);
        float* xp = x.data_write<float>();
        for (int i = 0; i < 4; ++i) xp[i] = 1.0f + i;
        x.requires_grad(true);
        Tensor w(ShapeTag{}, {4}, DType::kFloat, DeviceType::kCPU);
        float* wp = w.data_write<float>();
        for (int i = 0; i < 4; ++i) wp[i] = 2.0f + i;
        w.requires_grad(true);
        Tensor out = x * w;
        Tensor l = out.sum();
        AutoGrad::backward(l.getRelatedNode(), false);
        const float* gx = x.grad_ptr();
        const float* gw = w.grad_ptr();
        CHECK(gx != nullptr, "双 requires_grad mul→sum: x 收到梯度");
        bool okx = (gx != nullptr);
        for (int i = 0; okx && i < 4; ++i) okx = (std::abs(gx[i] - (2.0f + i)) < 1e-6f);
        CHECK(okx, "x 梯度 = w 值");
        bool okw = (gw != nullptr);
        for (int i = 0; okw && i < 4; ++i) okw = (std::abs(gw[i] - (1.0f + i)) < 1e-6f);
        CHECK(okw, "w 梯度 = x 值");
    }

    // Test: dot 反向断链回归(§4.97 ⑥)
    {
        AutoGrad::EnableGrad = true;
        Tensor x(ShapeTag{}, {3}, DType::kFloat, DeviceType::kCPU);
        Tensor w(ShapeTag{}, {3}, DType::kFloat, DeviceType::kCPU);
        for (int i = 0; i < 3; ++i) {
            x.data_write<float>()[i] = (float)(i + 1);
            w.data_write<float>()[i] = 10.0f * (float)(i + 1);
        }
        x.requires_grad(true);
        w.requires_grad(true);
        Tensor y = x.dot(w);
        CHECK(std::abs(y.item<float>() - (10.0f + 40.0f + 90.0f)) < 1e-4f, "dot 前向 = 140");
        CHECK(y.getRelatedNode() != nullptr, "dot 挂节点(修复前为 null, 反向断链)");
        AutoGrad::backward(y.getRelatedNode(), false);
        const float* gx = x.grad_ptr();
        const float* gw = w.grad_ptr();
        CHECK(gx != nullptr, "dot 后 x 收到梯度(修复前无)");
        bool okx = (gx != nullptr);
        for (int i = 0; okx && i < 3; ++i) okx = (std::abs(gx[i] - 10.0f * (float)(i + 1)) < 1e-5f);
        CHECK(okx, "dot grad_x = w");
        bool okw = (gw != nullptr);
        for (int i = 0; okw && i < 3; ++i) okw = (std::abs(gw[i] - (float)(i + 1)) < 1e-5f);
        CHECK(okw, "dot grad_w = x");
    }

    std::cout << (g_fails == 0 ? "=== ALL PASS ===" : "=== HAS FAIL ===") << "\n";
    return g_fails;
}
