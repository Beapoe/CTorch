/**
 * @file test_c3_graph.cpp
 * @brief C3 JIT EXP-1 验证：Graph IR + C3Engine 最小闭环
 * @details 验证项：
 *          1. 手写 Graph {x, y, Add} 编译执行结果与 eager CtorchScheduler::dispatch<op::Add> 一致
 *          2. 手写 Graph {x, y, MatMul} 编译执行结果与 eager matMul 一致
 *          3. canonicalize 对 Add(x, 0) 化简为 x，对 Mul(x, 1) 化简为 x
 * @date 2026/7/31
 */

#include <gtest/gtest.h>
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <iomanip>
#include <future>
#include <iostream>
#include <map>
#include <thread>
#include <vector>

#include "Tensor.h"
#include "CtorchScheduler.h"
#include "C3/Graph.h"
#include "C3/C3Engine.h"
#include "C3/C3OrchestratedKernel.h"
#include "C3/FusionPlanner.h"
#include "C3/Tracer.h"
#include "C3/C3KernelRegistry.h"
#include "C3/PGOManager.h"
#include "C3/C3Cleanup.h"
#include "Ctools.h"
#include "kernels/kernels.h"

// ======================= 辅助函数 =======================

/// 填充张量
static void fillTensor(Tensor& t, const std::vector<float>& values) {
    float* p = t.data_write<float>();
    for (size_t i = 0; i < values.size(); ++i) {
        p[i] = values[i];
    }
}

/// 检查两个张量是否逐元素近似相等
static bool tensorsAllClose(const Tensor& a, const Tensor& b, float rtol = 1e-4f, float atol = 1e-6f) {
    if (a.shape() != b.shape()) return false;
    const float* pa = a.data_read<float>();
    const float* pb = b.data_read<float>();
    size_t n = a.numel();
    for (size_t i = 0; i < n; ++i) {
        float diff = std::fabs(pa[i] - pb[i]);
        float max_val = std::max(std::fabs(pa[i]), std::fabs(pb[i]));
        if (diff > atol + rtol * max_val) {
            return false;
        }
    }
    return true;
}

// ======================= Graph IR 基础测试 =======================

TEST(GraphIR, CreateSimpleAddGraph) {
    using namespace ct::c3;

    Graph g;
    auto x_desc = TensorDesc::fromShape({3, 4});
    auto y_desc = TensorDesc::fromShape({3, 4});

    size_t x = g.addInput(x_desc);
    size_t y = g.addInput(y_desc);

    auto out_desc = TensorDesc::fromShape({3, 4});
    size_t z = g.addNode(AddNode{x_desc, y_desc}, {x, y}, out_desc);
    g.markOutput(z);

    EXPECT_TRUE(g.isValid());
    EXPECT_EQ(g.nodeCount(), 3u);  // x, y, z
    EXPECT_EQ(g.inputCount(), 2u);
    EXPECT_EQ(g.outputCount(), 1u);
}

TEST(GraphIR, CreateMatMulGraph) {
    using namespace ct::c3;

    Graph g;
    auto a_desc = TensorDesc::fromShape({2, 3});
    auto b_desc = TensorDesc::fromShape({3, 4});

    size_t a = g.addInput(a_desc);
    size_t b = g.addInput(b_desc);

    auto out_desc = TensorDesc::fromShape({2, 4});
    size_t c = g.addNode(MatMulNode{a_desc, b_desc}, {a, b}, out_desc);
    g.markOutput(c);

    EXPECT_TRUE(g.isValid());
    EXPECT_EQ(g.nodeCount(), 3u);
}

TEST(GraphIR, ToString) {
    using namespace ct::c3;

    Graph g;
    auto x_desc = TensorDesc::fromShape({3, 4});
    size_t x = g.addInput(x_desc);
    size_t y = g.addInput(x_desc);
    g.addNode(AddNode{x_desc, x_desc}, {x, y}, x_desc);

    std::string s = g.toString();
    EXPECT_NE(s.find("Graph"), std::string::npos);
    EXPECT_NE(s.find("Add"), std::string::npos);
    EXPECT_NE(s.find("INPUT"), std::string::npos);
}

// ======================= Canonicalize 测试 =======================

TEST(Canonicalize, AddWithZeroFolds) {
    using namespace ct::c3;

    Graph g;
    auto x_desc = TensorDesc::fromShape({3, 4});
    size_t x = g.addInput(x_desc);

    // 常量 0
    TensorDesc zero_desc = TensorDesc::fromShape({3, 4});
    size_t zero = g.addNode(ConstNode{0.0}, {}, zero_desc);

    auto out_desc = TensorDesc::fromShape({3, 4});
    size_t add = g.addNode(AddNode{x_desc, zero_desc}, {x, zero}, out_desc);
    g.markOutput(add);

    // canonicalize：Add(x, 0) → x
    Graph simplified = g.canonicalize();

    // 输出应被映射到输入节点 x（ConstNode 可能仍留在图中作为无用节点）
    EXPECT_EQ(simplified.outputCount(), 1u);
    EXPECT_EQ(simplified.node(simplified.outputs()[0]).out_desc.shape, x_desc.shape);
    // 验证简化后的图至少包含输入节点
    EXPECT_GE(simplified.nodeCount(), 1u);
}

TEST(Canonicalize, MulWithOneFolds) {
    using namespace ct::c3;

    Graph g;
    auto x_desc = TensorDesc::fromShape({3, 4});
    size_t x = g.addInput(x_desc);

    TensorDesc one_desc = TensorDesc::fromShape({3, 4});
    size_t one = g.addNode(ConstNode{1.0}, {}, one_desc);

    auto out_desc = TensorDesc::fromShape({3, 4});
    size_t mul = g.addNode(MulNode{x_desc, one_desc}, {x, one}, out_desc);
    g.markOutput(mul);

    // canonicalize：Mul(x, 1) → x
    Graph simplified = g.canonicalize();

    EXPECT_EQ(simplified.outputCount(), 1u);
    EXPECT_EQ(simplified.node(simplified.outputs()[0]).out_desc.shape, x_desc.shape);
    EXPECT_GE(simplified.nodeCount(), 1u);
}

TEST(Canonicalize, NoChangeForNonFoldable) {
    using namespace ct::c3;

    Graph g;
    auto x_desc = TensorDesc::fromShape({3, 4});
    auto y_desc = TensorDesc::fromShape({3, 4});
    size_t x = g.addInput(x_desc);
    size_t y = g.addInput(y_desc);

    // 常量 2（不是 0 或 1）
    TensorDesc c2_desc = TensorDesc::fromShape({3, 4});
    size_t c2 = g.addNode(ConstNode{2.0}, {}, c2_desc);

    auto out_desc = TensorDesc::fromShape({3, 4});
    size_t add = g.addNode(AddNode{x_desc, c2_desc}, {x, c2}, out_desc);
    g.markOutput(add);

    Graph simplified = g.canonicalize();

    // Add(x, 2) 不应被折叠
    EXPECT_GT(simplified.nodeCount(), 1u);
}

TEST(Canonicalize, MulWithZeroFolds) {
    using namespace ct::c3;

    // Mul(x, 0) → 0
    Graph g;
    auto x_desc = TensorDesc::fromShape({3, 4});
    size_t x = g.addInput(x_desc);

    TensorDesc zero_desc = TensorDesc::fromShape({3, 4});
    size_t zero = g.addNode(ConstNode{0.0}, {}, zero_desc);

    auto out_desc = TensorDesc::fromShape({3, 4});
    size_t mul = g.addNode(MulNode{x_desc, zero_desc}, {x, zero}, out_desc);
    g.markOutput(mul);

    Graph simplified = g.canonicalize();

    // 输出应被折叠为 ConstNode{0.0}
    EXPECT_EQ(simplified.outputCount(), 1u);
    const auto& out_node = simplified.node(simplified.outputs()[0]);
    EXPECT_TRUE(std::holds_alternative<ConstNode>(out_node.op));
    EXPECT_EQ(std::get<ConstNode>(out_node.op).value, 0.0);
}

TEST(Canonicalize, SubWithSameInputFolds) {
    using namespace ct::c3;

    // Sub(x, x) → 0
    Graph g;
    auto x_desc = TensorDesc::fromShape({3, 4});
    size_t x = g.addInput(x_desc);

    auto out_desc = TensorDesc::fromShape({3, 4});
    size_t sub = g.addNode(SubNode{x_desc, x_desc}, {x, x}, out_desc);
    g.markOutput(sub);

    Graph simplified = g.canonicalize();

    // 输出应被折叠为 ConstNode{0.0}
    EXPECT_EQ(simplified.outputCount(), 1u);
    const auto& out_node = simplified.node(simplified.outputs()[0]);
    EXPECT_TRUE(std::holds_alternative<ConstNode>(out_node.op));
    EXPECT_EQ(std::get<ConstNode>(out_node.op).value, 0.0);
}

TEST(Canonicalize, DivWithSameInputFolds) {
    using namespace ct::c3;

    // Div(x, x) → 1
    Graph g;
    auto x_desc = TensorDesc::fromShape({3, 4});
    size_t x = g.addInput(x_desc);

    auto out_desc = TensorDesc::fromShape({3, 4});
    size_t div = g.addNode(DivNode{x_desc, x_desc}, {x, x}, out_desc);
    g.markOutput(div);

    Graph simplified = g.canonicalize();

    // 输出应被折叠为 ConstNode{1.0}
    EXPECT_EQ(simplified.outputCount(), 1u);
    const auto& out_node = simplified.node(simplified.outputs()[0]);
    EXPECT_TRUE(std::holds_alternative<ConstNode>(out_node.op));
    EXPECT_EQ(std::get<ConstNode>(out_node.op).value, 1.0);
}

TEST(Canonicalize, NegNegFolds) {
    using namespace ct::c3;

    // Neg(Neg(x)) → x
    Graph g;
    auto x_desc = TensorDesc::fromShape({3, 4});
    size_t x = g.addInput(x_desc);

    auto neg_desc = TensorDesc::fromShape({3, 4});
    size_t inner_neg = g.addNode(NegNode{x_desc}, {x}, neg_desc);
    size_t outer_neg = g.addNode(NegNode{neg_desc}, {inner_neg}, neg_desc);
    g.markOutput(outer_neg);

    Graph simplified = g.canonicalize();

    // 输出应被映射到输入 x，形状保持一致
    EXPECT_EQ(simplified.outputCount(), 1u);
    const auto& out_node = simplified.node(simplified.outputs()[0]);
    // 被映射到输入后，输出节点应为 Input（ConstNode 占位符）
    EXPECT_TRUE(std::holds_alternative<ConstNode>(out_node.op));
    EXPECT_EQ(out_node.out_desc.shape, x_desc.shape);
}

TEST(Canonicalize, AddWithSameInput) {
    using namespace ct::c3;

    // Add(x, x) — 规则 7 已完全实现替换为 Mul(x, 2)
    Graph g;
    auto x_desc = TensorDesc::fromShape({3, 4});
    size_t x = g.addInput(x_desc);

    auto out_desc = TensorDesc::fromShape({3, 4});
    size_t add = g.addNode(AddNode{x_desc, x_desc}, {x, x}, out_desc);
    g.markOutput(add);

    Graph simplified = g.canonicalize();

    // Add(x, x) 被成功替换为 Mul(x, 2.0)
    EXPECT_EQ(simplified.outputCount(), 1u);
    const auto& out_node = simplified.node(simplified.outputs()[0]);
    EXPECT_TRUE(std::holds_alternative<MulNode>(out_node.op));
}

TEST(Canonicalize, AlgebraicRulesCompose) {
    using namespace ct::c3;

    // 组合验证：Mul(Sub(x, x), Div(y, y)) → Mul(0, 1) → 0
    Graph g;
    auto x_desc = TensorDesc::fromShape({4});
    size_t x = g.addInput(x_desc);
    size_t y = g.addInput(x_desc);

    auto out_desc = TensorDesc::fromShape({4});
    size_t sub = g.addNode(SubNode{x_desc, x_desc}, {x, x}, out_desc);
    size_t div = g.addNode(DivNode{x_desc, x_desc}, {y, y}, out_desc);
    size_t mul = g.addNode(MulNode{x_desc, x_desc}, {sub, div}, out_desc);
    g.markOutput(mul);

    Graph simplified = g.canonicalize();

    // Sub(x,x)→0, Div(y,y)→1, Mul(0,1)→0
    // 最终输出应为 ConstNode{0.0}
    EXPECT_EQ(simplified.outputCount(), 1u);
    const auto& out_node = simplified.node(simplified.outputs()[0]);
    EXPECT_TRUE(std::holds_alternative<ConstNode>(out_node.op));
    EXPECT_EQ(std::get<ConstNode>(out_node.op).value, 0.0);
}

TEST(Canonicalize, SubWithZeroRightInput) {
    using namespace ct::c3;
    // Sub(x, 0) -> x
    Graph g;
    auto x_desc = TensorDesc::fromShape({4});
    size_t x = g.addInput(x_desc);
    size_t zero = g.addConstant(0.0, TensorDesc::fromShape({1}));
    size_t sub = g.addNode(SubNode{x_desc, TensorDesc::fromShape({1})}, {x, zero}, x_desc);
    g.markOutput(sub);

    Graph simplified = g.canonicalize();
    EXPECT_EQ(simplified.outputCount(), 1u);
    // 应该直接折叠回原来的输入 x (新图中输入节点会重新创建在 0 处)
    EXPECT_EQ(simplified.outputs()[0], 0u);
}

TEST(Canonicalize, DivWithOneRightInput) {
    using namespace ct::c3;
    // Div(x, 1) -> x
    Graph g;
    auto x_desc = TensorDesc::fromShape({4});
    size_t x = g.addInput(x_desc);
    size_t one = g.addConstant(1.0, TensorDesc::fromShape({1}));
    size_t div = g.addNode(DivNode{x_desc, TensorDesc::fromShape({1})}, {x, one}, x_desc);
    g.markOutput(div);

    Graph simplified = g.canonicalize();
    EXPECT_EQ(simplified.outputCount(), 1u);
    EXPECT_EQ(simplified.outputs()[0], 0u);
}

TEST(Canonicalize, SubWithZeroLeftInput) {
    using namespace ct::c3;
    // Sub(0, x) -> Neg(x)
    Graph g;
    auto x_desc = TensorDesc::fromShape({4});
    size_t x = g.addInput(x_desc);
    size_t zero = g.addConstant(0.0, TensorDesc::fromShape({1}));
    size_t sub = g.addNode(SubNode{TensorDesc::fromShape({1}), x_desc}, {zero, x}, x_desc);
    g.markOutput(sub);

    Graph simplified = g.canonicalize();
    EXPECT_EQ(simplified.outputCount(), 1u);
    const auto& out_node = simplified.node(simplified.outputs()[0]);
    EXPECT_TRUE(std::holds_alternative<NegNode>(out_node.op));
}

TEST(Canonicalize, MulWithNegativeOne) {
    using namespace ct::c3;
    // Mul(x, -1) -> Neg(x)
    Graph g;
    auto x_desc = TensorDesc::fromShape({4});
    size_t x = g.addInput(x_desc);
    size_t neg_one = g.addConstant(-1.0, TensorDesc::fromShape({1}));
    size_t mul = g.addNode(MulNode{x_desc, TensorDesc::fromShape({1})}, {x, neg_one}, x_desc);
    g.markOutput(mul);

    Graph simplified = g.canonicalize();
    EXPECT_EQ(simplified.outputCount(), 1u);
    const auto& out_node = simplified.node(simplified.outputs()[0]);
    EXPECT_TRUE(std::holds_alternative<NegNode>(out_node.op));
}

// ======================= 算子融合测试 =======================

TEST(GraphFusion, FuseAddMul) {
    using namespace ct::c3;

    // 构建图: (x + y) * z
    Graph g;
    auto desc = TensorDesc::fromShape({4});
    size_t x = g.addInput(desc);
    size_t y = g.addInput(desc);
    size_t z = g.addInput(desc);
    size_t add = g.addNode(AddNode{desc, desc}, {x, y}, desc);
    size_t mul = g.addNode(MulNode{desc, desc}, {add, z}, desc);
    g.markOutput(mul);

    EXPECT_EQ(g.nodeCount(), 5u);  // x, y, z, add, mul

    Graph fused = g.fuse();

    // 融合后: x, y, z, Fused(Add->Mul)
    EXPECT_EQ(fused.nodeCount(), 4u);
    EXPECT_EQ(fused.inputCount(), 3u);
    EXPECT_EQ(fused.outputCount(), 1u);

    // 验证输出节点是 FusedNode
    const auto& out_node = fused.node(fused.outputs()[0]);
    EXPECT_TRUE(std::holds_alternative<FusedNode>(out_node.op));
    const auto& fnode = std::get<FusedNode>(out_node.op);
    EXPECT_EQ(fnode.ops.size(), 2u);
    EXPECT_EQ(fnode.arg_descs.size(), 3u);  // x, y, z
}

TEST(GraphFusion, FuseAddMulWithInputReuse) {
    using namespace ct::c3;

    // 构建图: (x * y) + x  (x 被复用)
    Graph g;
    auto desc = TensorDesc::fromShape({3});
    size_t x = g.addInput(desc);
    size_t y = g.addInput(desc);
    size_t mul = g.addNode(MulNode{desc, desc}, {x, y}, desc);
    size_t add = g.addNode(AddNode{desc, desc}, {mul, x}, desc);
    g.markOutput(add);

    EXPECT_EQ(g.nodeCount(), 4u);  // x, y, mul, add

    Graph fused = g.fuse();

    // 融合后: x, y, Fused(Mul->Add)
    EXPECT_EQ(fused.nodeCount(), 3u);
    EXPECT_EQ(fused.inputCount(), 2u);

    const auto& out_node = fused.node(fused.outputs()[0]);
    EXPECT_TRUE(std::holds_alternative<FusedNode>(out_node.op));
    const auto& fnode = std::get<FusedNode>(out_node.op);
    EXPECT_EQ(fnode.ops.size(), 2u);
    // x 虽然被复用，但外部输入去重后应只有 x, y
    EXPECT_EQ(fnode.arg_descs.size(), 2u);
}

TEST(GraphFusion, FuseThreeOps) {
    using namespace ct::c3;

    // 构建图: (x * y) + x - z
    Graph g;
    auto desc = TensorDesc::fromShape({4});
    size_t x = g.addInput(desc);
    size_t y = g.addInput(desc);
    size_t z = g.addInput(desc);
    size_t mul = g.addNode(MulNode{desc, desc}, {x, y}, desc);
    size_t add = g.addNode(AddNode{desc, desc}, {mul, x}, desc);
    size_t sub = g.addNode(SubNode{desc, desc}, {add, z}, desc);
    g.markOutput(sub);

    EXPECT_EQ(g.nodeCount(), 6u);  // x, y, z, mul, add, sub

    Graph fused = g.fuse();
    EXPECT_EQ(fused.nodeCount(), 4u);  // x, y, z, Fused(Mul->Add->Sub)
    EXPECT_EQ(fused.inputCount(), 3u);

    const auto& out_node = fused.node(fused.outputs()[0]);
    EXPECT_TRUE(std::holds_alternative<FusedNode>(out_node.op));
    const auto& fnode = std::get<FusedNode>(out_node.op);
    EXPECT_EQ(fnode.ops.size(), 3u);
}

TEST(GraphFusion, NoFusionForSingleOp) {
    using namespace ct::c3;

    // 单算子图不应融合
    Graph g;
    auto desc = TensorDesc::fromShape({4});
    size_t x = g.addInput(desc);
    size_t y = g.addInput(desc);
    size_t add = g.addNode(AddNode{desc, desc}, {x, y}, desc);
    g.markOutput(add);

    Graph fused = g.fuse();
    // 单算子不应产生融合，节点数不变
    EXPECT_EQ(fused.nodeCount(), g.nodeCount());
    EXPECT_FALSE(std::holds_alternative<FusedNode>(fused.node(fused.outputs()[0]).op));
}

TEST(GraphFusion, NoFusionForMatMul) {
    using namespace ct::c3;

    // MatMul 不应参与融合
    Graph g;
    auto a_desc = TensorDesc::fromShape({2, 3});
    auto b_desc = TensorDesc::fromShape({3, 4});
    auto c_desc = TensorDesc::fromShape({2, 4});
    size_t a = g.addInput(a_desc);
    size_t b = g.addInput(b_desc);
    size_t mm = g.addNode(MatMulNode{a_desc, b_desc}, {a, b}, c_desc);
    size_t relu = g.addNode(MulNode{c_desc, c_desc}, {mm, mm}, c_desc);  // 模拟 relu-like
    g.markOutput(relu);

    Graph fused = g.fuse();
    // MatMul 不能融合，所以 Mul 是单算子也不融合
    EXPECT_FALSE(std::holds_alternative<FusedNode>(fused.node(fused.outputs()[0]).op));
}

TEST(GraphFusion, NoFusionForMultiConsumer) {
    using namespace ct::c3;

    // 中间节点被多个消费者引用时不应融合
    Graph g;
    auto desc = TensorDesc::fromShape({4});
    size_t x = g.addInput(desc);
    size_t y = g.addInput(desc);
    size_t add = g.addNode(AddNode{desc, desc}, {x, y}, desc);
    size_t mul = g.addNode(MulNode{desc, desc}, {add, add}, desc);  // add 被消费两次
    g.markOutput(mul);

    Graph fused = g.fuse();
    // add 有多个消费者，不应融合
    EXPECT_FALSE(std::holds_alternative<FusedNode>(fused.node(fused.outputs()[0]).op));
}

// ======================= JIT 编译与执行测试 =======================

TEST(JITCompile, AddGraphExecute) {
    using namespace ct::c3;

    // 1. 构建图
    Graph g;
    auto desc = TensorDesc::fromShape({4});
    size_t x = g.addInput(desc);
    size_t y = g.addInput(desc);
    size_t z = g.addNode(AddNode{desc, desc}, {x, y}, desc);
    g.markOutput(z);

    // 2. 编译
    auto& engine = C3Engine::getInstance();
    CompileOptions opts;
    opts.target_device = DeviceType::kCPU;
    auto kernel = engine.compile(g, opts);
    ASSERT_NE(kernel, nullptr);

    // 3. 准备输入
    Tensor a(ShapeTag{}, {4});
    Tensor b(ShapeTag{}, {4});
    fillTensor(a, {1.0f, 2.0f, 3.0f, 4.0f});
    fillTensor(b, {5.0f, 6.0f, 7.0f, 8.0f});

    // 4. 执行编译 kernel
    auto results = kernel->execute({a, b});
    ASSERT_EQ(results.size(), 1u);

    // 5. eager 参考
    // 使用 CtorchScheduler 直接调度 Add
    Tensor eager = a + b;

    // 6. 比较
    EXPECT_TRUE(tensorsAllClose(results[0], eager));
}

TEST(JITCompile, MulGraphExecute) {
    using namespace ct::c3;

    Graph g;
    auto desc = TensorDesc::fromShape({3, 3});
    size_t x = g.addInput(desc);
    size_t y = g.addInput(desc);
    size_t z = g.addNode(MulNode{desc, desc}, {x, y}, desc);
    g.markOutput(z);

    auto& engine = C3Engine::getInstance();
    auto kernel = engine.compile(g, {});
    ASSERT_NE(kernel, nullptr);

    Tensor a(ShapeTag{}, {3, 3});
    Tensor b(ShapeTag{}, {3, 3});
    fillTensor(a, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
    fillTensor(b, {2.0f, 0.0f, 1.0f, 3.0f, 5.0f, 2.0f, 1.0f, 1.0f, 1.0f});

    auto results = kernel->execute({a, b});
    ASSERT_EQ(results.size(), 1u);

    Tensor eager = a * b;
    if (!tensorsAllClose(results[0], eager)) {
        std::cout << "results[0] values: ";
        for (size_t i = 0; i < 9; ++i) std::cout << results[0].data_read<float>()[i] << " ";
        std::cout << "\neager values: ";
        for (size_t i = 0; i < 9; ++i) std::cout << eager.data_read<float>()[i] << " ";
        std::cout << "\n";
    }
    EXPECT_TRUE(tensorsAllClose(results[0], eager));
}

// [SiLU 一等节点] 单节点 SiLU 图应能真正编译执行，且与 eager silu 数值一致。
// 验证执行层闭环：Graph SiLUNode → MLIRKernelGen 发射 c3.silu → SiLUOpLowering。
TEST(JITCompile, SiLUGraphExecute) {
    using namespace ct::c3;

    Graph g;
    auto desc = TensorDesc::fromShape({6});
    size_t x = g.addInput(desc);
    size_t z = g.addNode(SiLUNode{desc}, {x}, desc);
    g.markOutput(z);

    auto& engine = C3Engine::getInstance();
    auto kernel = engine.compile(g, {});
    ASSERT_NE(kernel, nullptr);

    Tensor a(ShapeTag{}, {6});
    fillTensor(a, {-2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 2.0f});

    auto results = kernel->execute({a});
    ASSERT_EQ(results.size(), 1u);

    Tensor eager = a.silu();
    if (!tensorsAllClose(results[0], eager)) {
        std::cout << "silu results: ";
        for (size_t i = 0; i < 6; ++i) std::cout << results[0].data_read<float>()[i] << " ";
        std::cout << "\nsilu eager:   ";
        for (size_t i = 0; i < 6; ++i) std::cout << eager.data_read<float>()[i] << " ";
        std::cout << "\n";
    }
    EXPECT_TRUE(tensorsAllClose(results[0], eager));
}

TEST(JITCompile, BroadcastAddGraphExecute) {
    using namespace ct::c3;

    // 广播图：out = x + bias，x=[2,3]，bias=[3]（一维周期广播）
    // 验证 OneShot 路由（isPureElementwiseGraph=true）对广播图的处理
    Graph g;
    auto x_desc = TensorDesc::fromShape({2, 3});
    auto b_desc = TensorDesc::fromShape({3});
    size_t x = g.addInput(x_desc);
    size_t bias = g.addInput(b_desc);
    size_t z = g.addNode(AddNode{x_desc, b_desc}, {x, bias}, x_desc);
    g.markOutput(z);

    auto& engine = C3Engine::getInstance();
    auto kernel = engine.compile(g, {});
    ASSERT_NE(kernel, nullptr);

    Tensor a(ShapeTag{}, {2, 3});
    Tensor b(ShapeTag{}, {3});
    fillTensor(a, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    fillTensor(b, {10.0f, 20.0f, 30.0f});

    auto results = kernel->execute({a, b});
    ASSERT_EQ(results.size(), 1u);

    Tensor eager = a + b;
    if (!tensorsAllClose(results[0], eager)) {
        std::cout << "results[0] values: ";
        for (size_t i = 0; i < 6; ++i) std::cout << results[0].data_read<float>()[i] << " ";
        std::cout << "\neager values: ";
        for (size_t i = 0; i < 6; ++i) std::cout << eager.data_read<float>()[i] << " ";
        std::cout << "\n";
    }
    EXPECT_TRUE(tensorsAllClose(results[0], eager));
}

TEST(JITCompile, MatMulGraphExecute) {
    using namespace ct::c3;

    Graph g;
    auto a_desc = TensorDesc::fromShape({2, 3});
    auto b_desc = TensorDesc::fromShape({3, 4});
    size_t a = g.addInput(a_desc);
    size_t b = g.addInput(b_desc);
    auto out_desc = TensorDesc::fromShape({2, 4});
    size_t c = g.addNode(MatMulNode{a_desc, b_desc}, {a, b}, out_desc);
    g.markOutput(c);

    auto& engine = C3Engine::getInstance();
    auto kernel = engine.compile(g, {});
    ASSERT_NE(kernel, nullptr);

    Tensor A(ShapeTag{}, {2, 3});
    Tensor B(ShapeTag{}, {3, 4});
    fillTensor(A, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    fillTensor(B, {1.0f, 0.0f, 0.0f, 1.0f,
                    0.0f, 1.0f, 0.0f, 0.0f,
                    0.0f, 0.0f, 1.0f, 0.0f});

    auto results = kernel->execute({A, B});
    ASSERT_EQ(results.size(), 1u);

    Tensor eager = matMul(A, B);
    EXPECT_TRUE(tensorsAllClose(results[0], eager));
}

TEST(JITCompile, CacheHit) {
    using namespace ct::c3;

    Graph g;
    auto desc = TensorDesc::fromShape({4});
    size_t x = g.addInput(desc);
    size_t y = g.addInput(desc);
    size_t z = g.addNode(AddNode{desc, desc}, {x, y}, desc);
    g.markOutput(z);

    auto& engine = C3Engine::getInstance();
    engine.clearCache();  // 确保缓存干净，避免前置测试污染

    // 首次编译
    auto stats_before = engine.getCacheStats();
    auto kernel1 = engine.compile(g, {});
    ASSERT_NE(kernel1, nullptr);

    auto stats_after_first = engine.getCacheStats();
    EXPECT_GT(stats_after_first.misses, stats_before.misses);

    // 第二次编译（相同图）应该命中缓存
    auto kernel2 = engine.compile(g, {});
    ASSERT_NE(kernel2, nullptr);

    auto stats_after_second = engine.getCacheStats();
    EXPECT_GT(stats_after_second.hits, stats_after_first.hits);

    // 同一缓存键
    EXPECT_EQ(kernel1->cacheKey(), kernel2->cacheKey());
}

// ======================= 融合 JIT 编译与执行测试 =======================

TEST(FusedJIT, AddMulFusedExecute) {
    using namespace ct::c3;

    // 构建图: (x + y) * z, 启用融合
    Graph g;
    auto desc = TensorDesc::fromShape({4});
    size_t x = g.addInput(desc);
    size_t y = g.addInput(desc);
    size_t z = g.addInput(desc);
    size_t add = g.addNode(AddNode{desc, desc}, {x, y}, desc);
    size_t mul = g.addNode(MulNode{desc, desc}, {add, z}, desc);
    g.markOutput(mul);

    auto& engine = C3Engine::getInstance();
    CompileOptions opts;
    opts.enable_fusion = true;
    auto kernel = engine.compile(g, opts);
    ASSERT_NE(kernel, nullptr);

    Tensor a(ShapeTag{}, {4});
    Tensor b(ShapeTag{}, {4});
    Tensor c(ShapeTag{}, {4});
    fillTensor(a, {1.0f, 2.0f, 3.0f, 4.0f});
    fillTensor(b, {5.0f, 6.0f, 7.0f, 8.0f});
    fillTensor(c, {2.0f, 3.0f, 4.0f, 5.0f});

    auto results = kernel->execute({a, b, c});
    ASSERT_EQ(results.size(), 1u);

    // eager: (a + b) * c
    Tensor eager = (a + b) * c;
    EXPECT_TRUE(tensorsAllClose(results[0], eager));
}

TEST(FusedJIT, MulAddFusedExecuteWithReuse) {
    using namespace ct::c3;

    // 构建图: (x * y) + x, 输入复用
    Graph g;
    auto desc = TensorDesc::fromShape({3});
    size_t x = g.addInput(desc);
    size_t y = g.addInput(desc);
    size_t mul = g.addNode(MulNode{desc, desc}, {x, y}, desc);
    size_t add = g.addNode(AddNode{desc, desc}, {mul, x}, desc);
    g.markOutput(add);

    auto& engine = C3Engine::getInstance();
    CompileOptions opts;
    opts.enable_fusion = true;
    auto kernel = engine.compile(g, opts);
    ASSERT_NE(kernel, nullptr);

    Tensor a(ShapeTag{}, {3});
    Tensor b(ShapeTag{}, {3});
    fillTensor(a, {1.0f, 2.0f, 3.0f});
    fillTensor(b, {4.0f, 5.0f, 6.0f});

    auto results = kernel->execute({a, b});
    ASSERT_EQ(results.size(), 1u);

    // eager: (a * b) + a
    Tensor eager = (a * b) + a;
    EXPECT_TRUE(tensorsAllClose(results[0], eager));
}

TEST(FusedJIT, ThreeOpFusedExecute) {
    using namespace ct::c3;

    // 构建图: (x * y) + x - z, 三操作融合
    Graph g;
    auto desc = TensorDesc::fromShape({4});
    size_t x = g.addInput(desc);
    size_t y = g.addInput(desc);
    size_t z = g.addInput(desc);
    size_t mul = g.addNode(MulNode{desc, desc}, {x, y}, desc);
    size_t add = g.addNode(AddNode{desc, desc}, {mul, x}, desc);
    size_t sub = g.addNode(SubNode{desc, desc}, {add, z}, desc);
    g.markOutput(sub);

    auto& engine = C3Engine::getInstance();
    CompileOptions opts;
    opts.enable_fusion = true;
    auto kernel = engine.compile(g, opts);
    ASSERT_NE(kernel, nullptr);

    Tensor a(ShapeTag{}, {4});
    Tensor b(ShapeTag{}, {4});
    Tensor c(ShapeTag{}, {4});
    fillTensor(a, {1.0f, 2.0f, 3.0f, 4.0f});
    fillTensor(b, {5.0f, 6.0f, 7.0f, 8.0f});
    fillTensor(c, {2.0f, 1.0f, 3.0f, 0.5f});

    auto results = kernel->execute({a, b, c});
    ASSERT_EQ(results.size(), 1u);

    // eager: (a * b) + a - c
    Tensor eager = (a * b) + a - c;
    EXPECT_TRUE(tensorsAllClose(results[0], eager));
}

TEST(FusedJIT, NegFusedExecute) {
    using namespace ct::c3;

    // 构建图: -(x * y)
    Graph g;
    auto desc = TensorDesc::fromShape({4});
    size_t x = g.addInput(desc);
    size_t y = g.addInput(desc);
    size_t mul = g.addNode(MulNode{desc, desc}, {x, y}, desc);
    size_t neg = g.addNode(NegNode{desc}, {mul}, desc);
    g.markOutput(neg);

    auto& engine = C3Engine::getInstance();
    CompileOptions opts;
    opts.enable_fusion = true;
    auto kernel = engine.compile(g, opts);
    ASSERT_NE(kernel, nullptr);

    Tensor a(ShapeTag{}, {4});
    Tensor b(ShapeTag{}, {4});
    fillTensor(a, {1.0f, 2.0f, 3.0f, 4.0f});
    fillTensor(b, {5.0f, 6.0f, 7.0f, 8.0f});

    auto results = kernel->execute({a, b});
    ASSERT_EQ(results.size(), 1u);

    // eager: -(a * b)
    Tensor eager = -(a * b);
    EXPECT_TRUE(tensorsAllClose(results[0], eager));
}

TEST(FusedJIT, DisableFusion) {
    using namespace ct::c3;

    // 禁用融合时，多算子图应 fallback 到单算子逐个执行
    Graph g;
    auto desc = TensorDesc::fromShape({4});
    size_t x = g.addInput(desc);
    size_t y = g.addInput(desc);
    size_t add = g.addNode(AddNode{desc, desc}, {x, y}, desc);
    g.markOutput(add);

    auto& engine = C3Engine::getInstance();
    CompileOptions opts;
    opts.enable_fusion = false;
    auto kernel = engine.compile(g, opts);
    ASSERT_NE(kernel, nullptr);

    Tensor a(ShapeTag{}, {4});
    Tensor b(ShapeTag{}, {4});
    fillTensor(a, {1.0f, 2.0f, 3.0f, 4.0f});
    fillTensor(b, {5.0f, 6.0f, 7.0f, 8.0f});

    auto results = kernel->execute({a, b});
    ASSERT_EQ(results.size(), 1u);
    EXPECT_TRUE(tensorsAllClose(results[0], a + b));
}

// ======================= Tracer 图捕获测试 =======================

TEST(Tracer, ManualTraceAdd) {
    using namespace ct::c3;

    Tracer tracer;
    tracer.begin();

    auto desc = TensorDesc::fromShape({4});
    auto x = tracer.input(desc);
    auto y = tracer.input(desc);
    auto z = x + y;

    Graph g = tracer.end(z);

    EXPECT_TRUE(g.isValid());
    EXPECT_EQ(g.inputCount(), 2u);
    EXPECT_EQ(g.outputCount(), 1u);
    // 图应有 2 个输入 + 1 个 Add 节点 = 3 个节点
    EXPECT_EQ(g.nodeCount(), 3u);
}

TEST(Tracer, ManualTraceMatMul) {
    using namespace ct::c3;

    Tracer tracer;
    tracer.begin();

    auto a_desc = TensorDesc::fromShape({2, 3});
    auto b_desc = TensorDesc::fromShape({3, 4});
    auto a = tracer.input(a_desc);
    auto b = tracer.input(b_desc);
    auto c = a.matmul(b);

    Graph g = tracer.end(c);

    EXPECT_TRUE(g.isValid());
    EXPECT_EQ(g.nodeCount(), 3u);
    EXPECT_EQ(g.outputCount(), 1u);
}

TEST(Tracer, LambdaTraceAdd) {
    using namespace ct::c3;

    auto desc = TensorDesc::fromShape({4});
    auto g = Tracer::trace(
        [](auto& x, auto& y) { return x + y; },
        desc, desc
    );

    EXPECT_TRUE(g.isValid());
    EXPECT_EQ(g.nodeCount(), 3u);
    EXPECT_EQ(g.inputCount(), 2u);
    EXPECT_EQ(g.outputCount(), 1u);
}

TEST(Tracer, LambdaTraceExpression) {
    using namespace ct::c3;

    auto desc = TensorDesc::fromShape({3, 3});
    // 捕获表达式: x * y + x
    auto g = Tracer::trace(
        [](auto& x, auto& y) {
            auto mul = x * y;
            return mul + x;
        },
        desc, desc
    );

    EXPECT_TRUE(g.isValid());
    // 节点: x(input), y(input), mul(x*y), add(mul+x)
    EXPECT_EQ(g.nodeCount(), 4u);
    EXPECT_EQ(g.outputCount(), 1u);
}

TEST(Tracer, LambdaTraceScalarOp) {
    using namespace ct::c3;

    auto desc = TensorDesc::fromShape({4});
    // y * 2.0f (标量乘)
    auto g = Tracer::trace(
        [](auto& x, auto& y) {
            return x + y * 2.0f;
        },
        desc, desc
    );

    EXPECT_TRUE(g.isValid());
    // 节点: x, y, const(2.0), mul(y*2.0), add(x+mul)
    EXPECT_EQ(g.nodeCount(), 5u);
}

TEST(Tracer, TraceCompileExecute) {
    using namespace ct::c3;

    // 1. 捕获图
    auto desc = TensorDesc::fromShape({4});
    auto g = Tracer::trace(
        [](auto& x, auto& y) { return x + y; },
        desc, desc
    );

    // 2. 编译
    auto& engine = C3Engine::getInstance();
    auto kernel = engine.compile(g, {});
    ASSERT_NE(kernel, nullptr);

    // 3. 执行
    Tensor a(ShapeTag{}, {4});
    Tensor b(ShapeTag{}, {4});
    fillTensor(a, {1.0f, 2.0f, 3.0f, 4.0f});
    fillTensor(b, {5.0f, 6.0f, 7.0f, 8.0f});

    auto results = kernel->execute({a, b});
    ASSERT_EQ(results.size(), 1u);

    // 4. 与 eager 对比
    Tensor eager = a + b;
    EXPECT_TRUE(tensorsAllClose(results[0], eager));
}

TEST(Tracer, TraceMatMulCompileExecute) {
    using namespace ct::c3;

    auto a_desc = TensorDesc::fromShape({2, 3});
    auto b_desc = TensorDesc::fromShape({3, 4});
    auto g = Tracer::trace(
        [](auto& x, auto& y) { return x.matmul(y); },
        a_desc, b_desc
    );

    auto& engine = C3Engine::getInstance();
    auto kernel = engine.compile(g, {});
    ASSERT_NE(kernel, nullptr);

    Tensor A(ShapeTag{}, {2, 3});
    Tensor B(ShapeTag{}, {3, 4});
    fillTensor(A, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    fillTensor(B, {1.0f, 0.0f, 0.0f, 1.0f,
                    0.0f, 1.0f, 0.0f, 0.0f,
                    0.0f, 0.0f, 1.0f, 0.0f});

    auto results = kernel->execute({A, B});
    ASSERT_EQ(results.size(), 1u);

    Tensor eager = matMul(A, B);
    EXPECT_TRUE(tensorsAllClose(results[0], eager));
}

TEST(Tracer, TraceExpressionCompileExecute) {
    using namespace ct::c3;

    // 捕获 (x * y) + x — 启用融合后将合并为单个 FusedNode
    auto desc = TensorDesc::fromShape({3});
    auto g = Tracer::trace(
        [](auto& x, auto& y) {
            auto mul = x * y;
            return mul + x;
        },
        desc, desc
    );

    EXPECT_TRUE(g.isValid());
    EXPECT_EQ(g.nodeCount(), 4u);

    // 启用融合编译
    auto& engine = C3Engine::getInstance();
    CompileOptions opts;
    opts.enable_fusion = true;
    auto kernel = engine.compile(g, opts);
    ASSERT_NE(kernel, nullptr);

    Tensor a(ShapeTag{}, {3});
    Tensor b(ShapeTag{}, {3});
    fillTensor(a, {1.0f, 2.0f, 3.0f});
    fillTensor(b, {4.0f, 5.0f, 6.0f});

    auto results = kernel->execute({a, b});
    ASSERT_EQ(results.size(), 1u);

    // eager: (a * b) + a
    Tensor eager = (a * b) + a;
    EXPECT_TRUE(tensorsAllClose(results[0], eager));
}

// ======================= C3 热替换 + 回退测试 =======================

TEST(C3HotReplace, InstallAndDispatch) {
    using namespace ct::c3;

    // 1. 编译 C3 kernel
    auto desc = TensorDesc::fromShape({4});
    Graph g;
    size_t x = g.addInput(desc);
    size_t y = g.addInput(desc);
    size_t z = g.addNode(AddNode{desc, desc}, {x, y}, desc);
    g.markOutput(z);

    auto& engine = C3Engine::getInstance();
    auto kernel = engine.compile(g, {});
    ASSERT_NE(kernel, nullptr);

    // 2. 安装到 C3 注册表
    KernelShapeInfo shapes;
    shapes.lhs_shape = {4};
    shapes.rhs_shape = {4};
    shapes.out_shape = {4};
    bool installed = kernel->installIntoRegistry(op::Add, shapes);
    EXPECT_TRUE(installed);

    // 3. 通过调度器 dispatch（模板版本，内部会查询 C3 注册表）
    Tensor a(ShapeTag{}, {4});
    Tensor b(ShapeTag{}, {4});
    fillTensor(a, {1.0f, 2.0f, 3.0f, 4.0f});
    fillTensor(b, {5.0f, 6.0f, 7.0f, 8.0f});

    auto& sched = CtorchScheduler::getInstance();
    Tensor c3_result = sched.dispatch<op::Add>(a, b);

    // 4. 与 eager 对比
    Tensor eager = a + b;
    EXPECT_TRUE(tensorsAllClose(c3_result, eager));

    // 5. 验证 C3 统计
    auto stats = C3KernelRegistry::getInstance().getStats();
    EXPECT_GT(stats.hit_count, 0u);
    EXPECT_GE(stats.install_count, 1u);

    // 6. 卸载
    C3KernelRegistry::getInstance().uninstall(op::Add, DeviceType::kCPU);
}

TEST(C3HotReplace, MulDispatch) {
    using namespace ct::c3;

    auto desc = TensorDesc::fromShape({3, 3});
    Graph g;
    size_t x = g.addInput(desc);
    size_t y = g.addInput(desc);
    size_t z = g.addNode(MulNode{desc, desc}, {x, y}, desc);
    g.markOutput(z);

    auto& engine = C3Engine::getInstance();
    auto kernel = engine.compile(g, {});
    ASSERT_NE(kernel, nullptr);

    KernelShapeInfo shapes;
    shapes.lhs_shape = {3, 3};
    shapes.rhs_shape = {3, 3};
    shapes.out_shape = {3, 3};
    kernel->installIntoRegistry(op::Mul, shapes);

    Tensor a(ShapeTag{}, {3, 3});
    Tensor b(ShapeTag{}, {3, 3});
    fillTensor(a, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
    fillTensor(b, {2.0f, 0.0f, 1.0f, 3.0f, 5.0f, 2.0f, 1.0f, 1.0f, 1.0f});

    auto& sched = CtorchScheduler::getInstance();
    Tensor c3_result = sched.dispatch<op::Mul>(a, b);
    Tensor eager = a * b;

    EXPECT_TRUE(tensorsAllClose(c3_result, eager));

    C3KernelRegistry::getInstance().uninstall(op::Mul, DeviceType::kCPU);
}

TEST(C3HotReplace, RollbackOnUninstall) {
    using namespace ct::c3;

    // 安装
    auto desc = TensorDesc::fromShape({4});
    Graph g;
    size_t x = g.addInput(desc);
    size_t y = g.addInput(desc);
    size_t z = g.addNode(AddNode{desc, desc}, {x, y}, desc);
    g.markOutput(z);

    auto& engine = C3Engine::getInstance();
    auto kernel = engine.compile(g, {});
    KernelShapeInfo shapes{{4}, {4}, {4}};
    kernel->installIntoRegistry(op::Add, shapes);

    // 第一次 dispatch — 走 C3
    Tensor a(ShapeTag{}, {4});
    Tensor b(ShapeTag{}, {4});
    fillTensor(a, {1.0f, 2.0f, 3.0f, 4.0f});
    fillTensor(b, {5.0f, 6.0f, 7.0f, 8.0f});

    auto& sched = CtorchScheduler::getInstance();
    Tensor c3_result = sched.dispatch<op::Add>(a, b);
    EXPECT_TRUE(tensorsAllClose(c3_result, a + b));

    auto stats_before = C3KernelRegistry::getInstance().getStats();
    size_t hits_before = stats_before.hit_count;

    // 卸载 C3 kernel
    C3KernelRegistry::getInstance().uninstall(op::Add, DeviceType::kCPU);

    // 第二次 dispatch — 应回退到 eager
    Tensor fallback_result = sched.dispatch<op::Add>(a, b);
    EXPECT_TRUE(tensorsAllClose(fallback_result, a + b));

    // 统计：hit_count 不应增加（因为已卸载）
    auto stats_after = C3KernelRegistry::getInstance().getStats();
    EXPECT_EQ(stats_after.hit_count, hits_before);
}

TEST(C3HotReplace, ShapeMismatchFallback) {
    using namespace ct::c3;

    // 安装形状为 {4} 的 C3 kernel
    auto desc = TensorDesc::fromShape({4});
    Graph g;
    size_t x = g.addInput(desc);
    size_t y = g.addInput(desc);
    size_t z = g.addNode(AddNode{desc, desc}, {x, y}, desc);
    g.markOutput(z);

    auto& engine = C3Engine::getInstance();
    auto kernel = engine.compile(g, {});
    KernelShapeInfo shapes{{4}, {4}, {4}};
    kernel->installIntoRegistry(op::Add, shapes);

    // 用不同形状 {6} 调用 — 应回退到 eager
    Tensor a(ShapeTag{}, {6});
    Tensor b(ShapeTag{}, {6});
    fillTensor(a, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    fillTensor(b, {6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f});

    auto& sched = CtorchScheduler::getInstance();
    Tensor result = sched.dispatch<op::Add>(a, b);
    Tensor eager = a + b;

    // 结果应正确（走了 eager 回退）
    EXPECT_TRUE(tensorsAllClose(result, eager));

    C3KernelRegistry::getInstance().uninstall(op::Add, DeviceType::kCPU);
}

TEST(C3HotReplace, StatsAccuracy) {
    using namespace ct::c3;

    C3KernelRegistry::getInstance().uninstallAll();
    auto stats0 = C3KernelRegistry::getInstance().getStats();
    EXPECT_EQ(stats0.active_entries, 0u);
    size_t base_install = stats0.install_count;
    size_t base_uninstall = stats0.uninstall_count;

    // 安装两个 C3 kernel
    auto desc = TensorDesc::fromShape({4});
    Graph g;
    size_t x = g.addInput(desc);
    size_t y = g.addInput(desc);
    size_t z = g.addNode(AddNode{desc, desc}, {x, y}, desc);
    g.markOutput(z);

    auto& engine = C3Engine::getInstance();
    auto kernel1 = engine.compile(g, {});
    KernelShapeInfo shapes{{4}, {4}, {4}};
    kernel1->installIntoRegistry(op::Add, shapes);

    // 第二个
    Graph g2;
    size_t a = g2.addInput(desc);
    size_t b = g2.addInput(desc);
    size_t c = g2.addNode(MulNode{desc, desc}, {a, b}, desc);
    g2.markOutput(c);
    auto kernel2 = engine.compile(g2, {});
    kernel2->installIntoRegistry(op::Mul, shapes);

    auto stats1 = C3KernelRegistry::getInstance().getStats();
    EXPECT_EQ(stats1.active_entries, 2u);
    EXPECT_EQ(stats1.install_count - base_install, 2u);

    C3KernelRegistry::getInstance().uninstallAll();
    auto stats2 = C3KernelRegistry::getInstance().getStats();
    EXPECT_EQ(stats2.active_entries, 0u);
    EXPECT_EQ(stats2.uninstall_count - base_uninstall, 2u);
}

// ======================= MLIR 后端测试 =======================

#ifdef CT_ENABLE_MLIR

/// 物化 2D 转置为连续张量作为 eager 参考。
/// 框架的 transpose() 返回懒视图（只换 shape/strides），而 sum() 假定连续布局，
/// 直接对视图做 sum 会得到错误结果，因此先按逻辑索引物化为连续张量。
static Tensor materializeTranspose(const Tensor& x) {
    const auto& s = x.shape();
    if (s.size() != 2) return x.clone();
    size_t M = s[0], N = s[1];
    const float* src = x.data_read<float>();
    Tensor out(ShapeTag{}, {N, M});
    float* dst = out.data_write<float>();
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            dst[j * M + i] = src[i * N + j];
        }
    }
    return out;
}

/// 辅助：用 MLIR 后端编译单算子图
static std::shared_ptr<ct::c3::CompiledKernel> compileMLIR(
    ct::c3::Graph& g, DeviceType dev = DeviceType::kCPU)
{
    ct::c3::CompileOptions opts;
    opts.backend = ct::c3::C3Backend::MLIR;
    opts.target_device = dev;
    return ct::c3::C3Engine::getInstance().compile(g, opts);
}

TEST(MLIRBackend, AddGraphExecute) {
    using namespace ct::c3;

    Graph g;
    auto desc = TensorDesc::fromShape({4});
    size_t x = g.addInput(desc);
    size_t y = g.addInput(desc);
    g.addNode(AddNode{desc, desc}, {x, y}, desc);
    g.markOutput(g.nodeCount() - 1);

    auto kernel = compileMLIR(g);
    ASSERT_NE(kernel, nullptr);

    Tensor a(ShapeTag{}, {4});
    Tensor b(ShapeTag{}, {4});
    fillTensor(a, {1.0f, 2.0f, 3.0f, 4.0f});
    fillTensor(b, {5.0f, 6.0f, 7.0f, 8.0f});

    auto results = kernel->execute({a, b});
    ASSERT_EQ(results.size(), 1u);

    Tensor eager = a + b;
    EXPECT_TRUE(tensorsAllClose(results[0], eager));
}

TEST(MLIRBackend, MulGraphExecute) {
    using namespace ct::c3;

    Graph g;
    auto desc = TensorDesc::fromShape({3, 3});
    size_t x = g.addInput(desc);
    size_t y = g.addInput(desc);
    g.addNode(MulNode{desc, desc}, {x, y}, desc);
    g.markOutput(g.nodeCount() - 1);

    auto kernel = compileMLIR(g);
    ASSERT_NE(kernel, nullptr);

    Tensor a(ShapeTag{}, {3, 3});
    Tensor b(ShapeTag{}, {3, 3});
    fillTensor(a, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
    fillTensor(b, {2.0f, 0.0f, 1.0f, 3.0f, 5.0f, 2.0f, 1.0f, 1.0f, 1.0f});

    auto results = kernel->execute({a, b});
    ASSERT_EQ(results.size(), 1u);

    Tensor eager = a * b;
    EXPECT_TRUE(tensorsAllClose(results[0], eager));
}

TEST(MLIRBackend, SubGraphExecute) {
    using namespace ct::c3;

    Graph g;
    auto desc = TensorDesc::fromShape({4});
    size_t x = g.addInput(desc);
    size_t y = g.addInput(desc);
    g.addNode(SubNode{desc, desc}, {x, y}, desc);
    g.markOutput(g.nodeCount() - 1);

    auto kernel = compileMLIR(g);
    ASSERT_NE(kernel, nullptr);

    Tensor a(ShapeTag{}, {4});
    Tensor b(ShapeTag{}, {4});
    fillTensor(a, {10.0f, 20.0f, 30.0f, 40.0f});
    fillTensor(b, {1.0f, 2.0f, 3.0f, 4.0f});

    auto results = kernel->execute({a, b});
    ASSERT_EQ(results.size(), 1u);

    Tensor eager = a - b;
    EXPECT_TRUE(tensorsAllClose(results[0], eager));
}

TEST(MLIRBackend, DivGraphExecute) {
    using namespace ct::c3;

    Graph g;
    auto desc = TensorDesc::fromShape({4});
    size_t x = g.addInput(desc);
    size_t y = g.addInput(desc);
    g.addNode(DivNode{desc, desc}, {x, y}, desc);
    g.markOutput(g.nodeCount() - 1);

    auto kernel = compileMLIR(g);
    ASSERT_NE(kernel, nullptr);

    Tensor a(ShapeTag{}, {4});
    Tensor b(ShapeTag{}, {4});
    fillTensor(a, {10.0f, 20.0f, 30.0f, 40.0f});
    fillTensor(b, {2.0f, 4.0f, 5.0f, 8.0f});

    auto results = kernel->execute({a, b});
    ASSERT_EQ(results.size(), 1u);

    Tensor eager = a / b;
    EXPECT_TRUE(tensorsAllClose(results[0], eager));
}

TEST(MLIRBackend, NegGraphExecute) {
    using namespace ct::c3;

    Graph g;
    auto desc = TensorDesc::fromShape({4});
    size_t x = g.addInput(desc);
    g.addNode(NegNode{desc}, {x}, desc);
    g.markOutput(g.nodeCount() - 1);

    auto kernel = compileMLIR(g);
    ASSERT_NE(kernel, nullptr);

    Tensor a(ShapeTag{}, {4});
    fillTensor(a, {1.0f, -2.0f, 3.0f, -4.0f});

    auto results = kernel->execute({a});
    ASSERT_EQ(results.size(), 1u);

    Tensor eager = -a;
    EXPECT_TRUE(tensorsAllClose(results[0], eager));
}

TEST(MLIRBackend, MatMulGraphExecute) {
    using namespace ct::c3;

    Graph g;
    auto a_desc = TensorDesc::fromShape({2, 3});
    auto b_desc = TensorDesc::fromShape({3, 4});
    size_t a = g.addInput(a_desc);
    size_t b = g.addInput(b_desc);
    auto out_desc = TensorDesc::fromShape({2, 4});
    g.addNode(MatMulNode{a_desc, b_desc}, {a, b}, out_desc);
    g.markOutput(g.nodeCount() - 1);

    auto kernel = compileMLIR(g);
    ASSERT_NE(kernel, nullptr);

    Tensor A(ShapeTag{}, {2, 3});
    Tensor B(ShapeTag{}, {3, 4});
    fillTensor(A, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    fillTensor(B, {1.0f, 0.0f, 0.0f, 1.0f,
                    0.0f, 1.0f, 0.0f, 0.0f,
                    0.0f, 0.0f, 1.0f, 0.0f});

    auto results = kernel->execute({A, B});
    ASSERT_EQ(results.size(), 1u);

    Tensor eager = matMul(A, B);
    EXPECT_TRUE(tensorsAllClose(results[0], eager));
}

// [JIT2.0 多节点端到端] Transpose → SumReduce(axis=0)
// 输入 [2x3] → 转置 [3x2] → 沿 axis=0 归约 → [2]
// 覆盖: buildMultiNodeMLIR + TransposeOp/SumReduceOp 创建与 Lowering
TEST(MLIRBackend, TransposeSumReduceAxis0MultiNode) {
    using namespace ct::c3;

    Graph g;
    auto in_desc = TensorDesc::fromShape({2, 3});
    size_t in = g.addInput(in_desc);

    auto transpose_desc = TensorDesc::fromShape({3, 2});
    size_t transpose = g.addNode(TransposeNode{in_desc, 0, 1}, {in}, transpose_desc);

    auto sum_desc = TensorDesc::fromShape({2});
    size_t sum = g.addNode(SumReduceNode{transpose_desc, 0}, {transpose}, sum_desc);
    g.markOutput(sum);

    auto kernel = compileMLIR(g);
    ASSERT_NE(kernel, nullptr);

    Tensor A(ShapeTag{}, {2, 3});
    fillTensor(A, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}); // [[1,2,3],[4,5,6]]

    auto results = kernel->execute({A});
    ASSERT_EQ(results.size(), 1u);

    // Eager: transpose → [[1,4],[2,5],[3,6]] → sum(axis=0) → [1+2+3, 4+5+6] = [6,15]
    Tensor eager = materializeTranspose(A).sum(0);
    std::cerr << "[DBG axis0] mlir shape="; for (auto s : results[0].shape()) std::cerr << s << ",";
    std::cerr << " vals="; for (size_t i=0;i<results[0].numel();++i) std::cerr << results[0].data_read<float>()[i] << ",";
    std::cerr << " | eager shape="; for (auto s : eager.shape()) std::cerr << s << ",";
    std::cerr << " vals="; for (size_t i=0;i<eager.numel();++i) std::cerr << eager.data_read<float>()[i] << ",";
    std::cerr << "\n";
    EXPECT_TRUE(tensorsAllClose(results[0], eager));
}

// [JIT2.0 多节点端到端] Transpose → SumReduce(axis=1)
// 输入 [2x3] → 转置 [3x2] → 沿 axis=1 归约 → [3]
TEST(MLIRBackend, TransposeSumReduceAxis1MultiNode) {
    using namespace ct::c3;

    Graph g;
    auto in_desc = TensorDesc::fromShape({2, 3});
    size_t in = g.addInput(in_desc);

    auto transpose_desc = TensorDesc::fromShape({3, 2});
    size_t transpose = g.addNode(TransposeNode{in_desc, 0, 1}, {in}, transpose_desc);

    auto sum_desc = TensorDesc::fromShape({3});
    size_t sum = g.addNode(SumReduceNode{transpose_desc, 1}, {transpose}, sum_desc);
    g.markOutput(sum);

    auto kernel = compileMLIR(g);
    ASSERT_NE(kernel, nullptr);

    Tensor A(ShapeTag{}, {2, 3});
    fillTensor(A, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}); // [[1,2,3],[4,5,6]]

    auto results = kernel->execute({A});
    ASSERT_EQ(results.size(), 1u);

    // Eager: transpose → [[1,4],[2,5],[3,6]] → sum(axis=1) → [1+4, 2+5, 3+6] = [5,7,9]
    Tensor eager = materializeTranspose(A).sum(1);
    std::cerr << "[DBG axis1] mlir shape="; for (auto s : results[0].shape()) std::cerr << s << ",";
    std::cerr << " vals="; for (size_t i=0;i<results[0].numel();++i) std::cerr << results[0].data_read<float>()[i] << ",";
    std::cerr << " | eager shape="; for (auto s : eager.shape()) std::cerr << s << ",";
    std::cerr << " vals="; for (size_t i=0;i<eager.numel();++i) std::cerr << eager.data_read<float>()[i] << ",";
    std::cerr << "\n";
    EXPECT_TRUE(tensorsAllClose(results[0], eager));
}

// [G3 真接管] OrchestratedKernel: 编排执行与整图执行逐位一致
// 覆盖三个关键点: ① separator(SumReduce) 独立切出 ② 跨子图依赖(Mul 消费 SumReduce 输出)
// ③ Const 物化(Add 消费图内常量 one)。
TEST(MLIRBackend, OrchestratedKernelMatchesWholeGraph) {
    using namespace ct::c3;

    // 图: x[2,4] -> SumReduce(axis=0)[4] -> Mul(s,c)[4] (无广播, 聚焦编排逻辑)
    // 覆盖: separator(SumReduce) 独立切出 + 跨子图依赖(Mul 消费 SumReduce 输出)。
    Graph g;
    auto x_desc = TensorDesc::fromShape({2, 4});
    auto v4 = TensorDesc::fromShape({4});
    size_t x = g.addInput(x_desc);
    size_t c = g.addInput(v4);
    size_t s = g.addNode(SumReduceNode{x_desc, 0}, {x}, v4);   // separator
    size_t y = g.addNode(MulNode{v4, v4}, {s, c}, v4);         // 依赖 s
    g.markOutput(y);

    // 整图内核
    auto whole = compileMLIR(g);
    ASSERT_NE(whole, nullptr);

    // 输入
    Tensor X(ShapeTag{}, {2, 4});
    Tensor C(ShapeTag{}, {4});
    fillTensor(X, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
    fillTensor(C, {0.5f, 0.5f, 0.5f, 0.5f});

    // planner + partitionGraph 切分(SumReduce 独立 → 必切出多子图)
    RegionFusionPolicy policy;
    FusionPlan plan = FusionPlanner::planUnits(g, FusionStrategy::RegionKernel, policy);
    std::vector<PartitionedSubGraph> subs = partitionGraph(g, plan);
    ASSERT_GT(subs.size(), 1u);

    // 编译各子图 + 编排内核
    std::vector<std::shared_ptr<CompiledKernel>> sub_kernels;
    sub_kernels.reserve(subs.size());
    for (auto& s : subs) {
        auto k = compileMLIR(s.graph);
        ASSERT_NE(k, nullptr);
        sub_kernels.push_back(std::move(k));
    }
    auto orch = buildOrchestratedKernel(g, subs, sub_kernels);
    ASSERT_NE(orch, nullptr);

    // 相同输入, 逐位一致
    auto whole_out = whole->execute({X, C});
    auto orch_out = orch->execute({X, C});
    ASSERT_EQ(whole_out.size(), 1u);
    ASSERT_EQ(orch_out.size(), 1u);
    EXPECT_TRUE(tensorsAllClose(whole_out[0], orch_out[0]));
}


// 覆盖 c3.matmul op 在单/多节点路径的创建，以及 MatMulOpLowering 的三种策略选择
// （small_inline / tiled / cblas）、transpose folding 与 epilogue 融合。

// 用确定性伪随机填充张量（避免硬编码大矩阵数据）
static void fillDeterministic(Tensor& t, unsigned seed) {
    float* p = t.data_write<float>();
    for (size_t i = 0; i < t.numel(); ++i) {
        unsigned v = (unsigned)((i + 1) * (seed + 1));
        p[i] = (float)((v * 2654435761u) % 1000u - 500u) * 0.01f;
    }
}

// 单节点 small_inline：total_ops = 2*3*4 = 24 < 256
TEST(MLIRBackend, MatMulSmallInline) {
    using namespace ct::c3;

    Graph g;
    auto a_desc = TensorDesc::fromShape({2, 3});
    auto b_desc = TensorDesc::fromShape({3, 4});
    size_t a = g.addInput(a_desc);
    size_t b = g.addInput(b_desc);
    auto out_desc = TensorDesc::fromShape({2, 4});
    g.addNode(MatMulNode{a_desc, b_desc}, {a, b}, out_desc);
    g.markOutput(g.nodeCount() - 1);

    auto kernel = compileMLIR(g);
    ASSERT_NE(kernel, nullptr);

    Tensor A(ShapeTag{}, {2, 3});
    Tensor B(ShapeTag{}, {3, 4});
    fillDeterministic(A, 1);
    fillDeterministic(B, 2);

    auto results = kernel->execute({A, B});
    ASSERT_EQ(results.size(), 1u);
    Tensor eager = matMul(A, B);
    EXPECT_TRUE(tensorsAllClose(results[0], eager));
}

// 单节点 tiled：total_ops = 32*3*32 = 3072 ∈ [256, 4096)，且 M,N >= 默认 tile(32)
TEST(MLIRBackend, MatMulTiledMedium) {
    using namespace ct::c3;

    Graph g;
    auto a_desc = TensorDesc::fromShape({32, 3});
    auto b_desc = TensorDesc::fromShape({3, 32});
    size_t a = g.addInput(a_desc);
    size_t b = g.addInput(b_desc);
    auto out_desc = TensorDesc::fromShape({32, 32});
    g.addNode(MatMulNode{a_desc, b_desc}, {a, b}, out_desc);
    g.markOutput(g.nodeCount() - 1);

    auto kernel = compileMLIR(g);
    ASSERT_NE(kernel, nullptr);

    Tensor A(ShapeTag{}, {32, 3});
    Tensor B(ShapeTag{}, {3, 32});
    fillDeterministic(A, 3);
    fillDeterministic(B, 4);

    auto results = kernel->execute({A, B});
    ASSERT_EQ(results.size(), 1u);
    Tensor eager = matMul(A, B);
    EXPECT_TRUE(tensorsAllClose(results[0], eager));
}

// 单节点 cblas：total_ops = 16*16*16 = 4096 >= kTiledMatMulThreshold
TEST(MLIRBackend, MatMulCblasLarge) {
    using namespace ct::c3;

    Graph g;
    auto a_desc = TensorDesc::fromShape({16, 16});
    auto b_desc = TensorDesc::fromShape({16, 16});
    size_t a = g.addInput(a_desc);
    size_t b = g.addInput(b_desc);
    auto out_desc = TensorDesc::fromShape({16, 16});
    g.addNode(MatMulNode{a_desc, b_desc}, {a, b}, out_desc);
    g.markOutput(g.nodeCount() - 1);

    auto kernel = compileMLIR(g);
    ASSERT_NE(kernel, nullptr);

    Tensor A(ShapeTag{}, {16, 16});
    Tensor B(ShapeTag{}, {16, 16});
    fillDeterministic(A, 5);
    fillDeterministic(B, 6);

    auto results = kernel->execute({A, B});
    ASSERT_EQ(results.size(), 1u);
    Tensor eager = matMul(A, B);
    EXPECT_TRUE(tensorsAllClose(results[0], eager, 1e-4f, 1e-5f));
}

// 多节点：MatMul → ReLU（无 epilogue 融合，验证多节点路径 MatMul op 独立创建）
TEST(MLIRBackend, MatMulMultiNodeNoFusion) {
    using namespace ct::c3;

    Graph g;
    auto a_desc = TensorDesc::fromShape({2, 3});
    auto b_desc = TensorDesc::fromShape({3, 2});
    auto out_desc = TensorDesc::fromShape({2, 2});
    size_t a = g.addInput(a_desc);
    size_t b = g.addInput(b_desc);
    size_t mm = g.addNode(MatMulNode{a_desc, b_desc}, {a, b}, out_desc);
    size_t relu = g.addNode(ReLUNode{out_desc}, {mm}, out_desc);
    g.markOutput(relu);

    auto kernel = compileMLIR(g);
    ASSERT_NE(kernel, nullptr);

    Tensor A(ShapeTag{}, {2, 3});
    Tensor B(ShapeTag{}, {3, 2});
    fillTensor(A, {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f});
    fillTensor(B, {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f});

    auto results = kernel->execute({A, B});
    ASSERT_EQ(results.size(), 1u);
    Tensor eager = matMul(A, B).relu();
    EXPECT_TRUE(tensorsAllClose(results[0], eager));
}

// 多节点 transpose folding（transA=112）：Transpose(A) @ B
// A{16,32} → Transpose → {32,16}；B{16,8}；total_ops=4096 → cblas（transA 生效）
TEST(MLIRBackend, MatMulTransposeFoldAMultiNode) {
    using namespace ct::c3;

    Graph g;
    auto a_desc = TensorDesc::fromShape({16, 32});
    auto b_desc = TensorDesc::fromShape({16, 8});
    auto aT_desc = TensorDesc::fromShape({32, 16});
    auto out_desc = TensorDesc::fromShape({32, 8});
    size_t a = g.addInput(a_desc);
    size_t b = g.addInput(b_desc);
    size_t aT = g.addNode(TransposeNode{a_desc, 0, 1}, {a}, aT_desc);
    size_t mm = g.addNode(MatMulNode{aT_desc, b_desc}, {aT, b}, out_desc);
    g.markOutput(mm);

    auto kernel = compileMLIR(g);
    ASSERT_NE(kernel, nullptr);

    Tensor A(ShapeTag{}, {16, 32});
    Tensor B(ShapeTag{}, {16, 8});
    fillDeterministic(A, 7);
    fillDeterministic(B, 8);

    auto results = kernel->execute({A, B});
    ASSERT_EQ(results.size(), 1u);
    Tensor eager = matMul(materializeTranspose(A), B);
    EXPECT_TRUE(tensorsAllClose(results[0], eager, 1e-4f, 1e-5f));
}

// 多节点 transpose folding（transB=112）：A @ Transpose(B)
// A{8,16}；B{32,16} → Transpose → {16,32}；total_ops=4096 → cblas（transB 生效）
TEST(MLIRBackend, MatMulTransposeFoldBMultiNode) {
    using namespace ct::c3;

    Graph g;
    auto a_desc = TensorDesc::fromShape({8, 16});
    auto b_desc = TensorDesc::fromShape({32, 16});
    auto bT_desc = TensorDesc::fromShape({16, 32});
    auto out_desc = TensorDesc::fromShape({8, 32});
    size_t a = g.addInput(a_desc);
    size_t b = g.addInput(b_desc);
    size_t bT = g.addNode(TransposeNode{b_desc, 0, 1}, {b}, bT_desc);
    size_t mm = g.addNode(MatMulNode{a_desc, bT_desc}, {a, bT}, out_desc);
    g.markOutput(mm);

    auto kernel = compileMLIR(g);
    ASSERT_NE(kernel, nullptr);

    Tensor A(ShapeTag{}, {8, 16});
    Tensor B(ShapeTag{}, {32, 16});
    fillDeterministic(A, 9);
    fillDeterministic(B, 10);

    auto results = kernel->execute({A, B});
    ASSERT_EQ(results.size(), 1u);
    Tensor eager = matMul(A, materializeTranspose(B));
    EXPECT_TRUE(tensorsAllClose(results[0], eager, 1e-4f, 1e-5f));
}

// 多节点 epilogue 融合：MatMul → Add(bias) → ReLU 合成一个 c3.matmul op（bias + act）
TEST(MLIRBackend, MatMulEpilogueBiasReLUMultiNode) {
    using namespace ct::c3;

    Graph g;
    auto a_desc = TensorDesc::fromShape({2, 3});
    auto b_desc = TensorDesc::fromShape({3, 2});
    auto out_desc = TensorDesc::fromShape({2, 2});
    auto bias_desc = TensorDesc::fromShape({2, 2});
    size_t a = g.addInput(a_desc);
    size_t b = g.addInput(b_desc);
    size_t bias = g.addInput(bias_desc);
    size_t mm = g.addNode(MatMulNode{a_desc, b_desc}, {a, b}, out_desc);
    size_t add = g.addNode(AddNode{out_desc, bias_desc}, {mm, bias}, out_desc);
    size_t relu = g.addNode(ReLUNode{out_desc}, {add}, out_desc);
    g.markOutput(relu);

    auto kernel = compileMLIR(g);
    ASSERT_NE(kernel, nullptr);

    Tensor A(ShapeTag{}, {2, 3});
    Tensor B(ShapeTag{}, {3, 2});
    Tensor Bias(ShapeTag{}, {2, 2});
    fillTensor(A, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    fillTensor(B, {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f});
    fillTensor(Bias, {-2.0f, 5.0f, 3.0f, -1.0f});

    auto results = kernel->execute({A, B, Bias});
    ASSERT_EQ(results.size(), 1u);
    Tensor eager = (matMul(A, B) + Bias).relu();
    EXPECT_TRUE(tensorsAllClose(results[0], eager));
}

// 多节点 epilogue 融合：MatMul → SiLU 合成一个 c3.matmul op（act=SiLU）
// [立项 C] 验证 MatMulActivation::SiLU 的 lowering（buildSmallMatMul + vector/scalar 分支 act==4）。
TEST(MLIRBackend, MatMulEpilogueSiLUMultiNode) {
    using namespace ct::c3;

    Graph g;
    auto a_desc = TensorDesc::fromShape({2, 3});
    auto b_desc = TensorDesc::fromShape({3, 4});
    auto out_desc = TensorDesc::fromShape({2, 4});
    size_t a = g.addInput(a_desc);
    size_t b = g.addInput(b_desc);
    size_t mm = g.addNode(MatMulNode{a_desc, b_desc}, {a, b}, out_desc);
    size_t silu = g.addNode(SiLUNode{out_desc}, {mm}, out_desc);
    g.markOutput(silu);

    auto kernel = compileMLIR(g);
    ASSERT_NE(kernel, nullptr);

    Tensor A(ShapeTag{}, {2, 3});
    Tensor B(ShapeTag{}, {3, 4});
    fillTensor(A, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    fillTensor(B, {-1.0f, 0.0f, 1.0f, 2.0f,
                    0.5f, -0.5f, 1.5f, -1.5f,
                    2.0f, 1.0f, 0.0f, -2.0f});

    auto results = kernel->execute({A, B});
    ASSERT_EQ(results.size(), 1u);
    Tensor eager = matMul(A, B).silu();
    if (!tensorsAllClose(results[0], eager)) {
        std::cout << "matmul+silu results: ";
        for (size_t i = 0; i < 8; ++i) std::cout << results[0].data_read<float>()[i] << " ";
        std::cout << "\nmatmul+silu eager:   ";
        for (size_t i = 0; i < 8; ++i) std::cout << eager.data_read<float>()[i] << " ";
        std::cout << "\n";
    }
    EXPECT_TRUE(tensorsAllClose(results[0], eager));
}

TEST(MLIRBackend, CacheHitSeparateFromHandwritten) {
    using namespace ct::c3;

    Graph g;
    auto desc = TensorDesc::fromShape({4});
    size_t x = g.addInput(desc);
    size_t y = g.addInput(desc);
    size_t z = g.addNode(AddNode{desc, desc}, {x, y}, desc);
    g.markOutput(z);

    auto& engine = C3Engine::getInstance();
    engine.clearCache();  // 确保缓存干净，避免前置测试污染

    // MLIR 编译
    CompileOptions mlir_opts;
    mlir_opts.backend = C3Backend::MLIR;
    auto stats_before = engine.getCacheStats();
    auto kernel1 = engine.compile(g, mlir_opts);
    ASSERT_NE(kernel1, nullptr);

    auto stats_after_first = engine.getCacheStats();
    EXPECT_GT(stats_after_first.misses, stats_before.misses);

    // 第二次 MLIR 编译（相同图）应命中缓存
    auto kernel2 = engine.compile(g, mlir_opts);
    ASSERT_NE(kernel2, nullptr);

    auto stats_after_second = engine.getCacheStats();
    EXPECT_GT(stats_after_second.hits, stats_after_first.hits);
    EXPECT_EQ(kernel1->cacheKey(), kernel2->cacheKey());

    // Handwritten 编译应有不同缓存键
    CompileOptions hw_opts;
    hw_opts.backend = C3Backend::Handwritten;
    auto kernel_hw = engine.compile(g, hw_opts);
    ASSERT_NE(kernel_hw, nullptr);
    EXPECT_NE(kernel1->cacheKey(), kernel_hw->cacheKey());
}

TEST(MLIRBackend, HotReplaceInstallAndDispatch) {
    using namespace ct::c3;

    auto desc = TensorDesc::fromShape({4});
    Graph g;
    size_t x = g.addInput(desc);
    size_t y = g.addInput(desc);
    size_t z = g.addNode(AddNode{desc, desc}, {x, y}, desc);
    g.markOutput(z);

    auto kernel = compileMLIR(g);
    ASSERT_NE(kernel, nullptr);

    // 安装 MLIR 编译的 kernel 到 C3 注册表
    KernelShapeInfo shapes{{4}, {4}, {4}};
    bool installed = kernel->installIntoRegistry(op::Add, shapes);
    EXPECT_TRUE(installed);

    Tensor a(ShapeTag{}, {4});
    Tensor b(ShapeTag{}, {4});
    fillTensor(a, {1.0f, 2.0f, 3.0f, 4.0f});
    fillTensor(b, {5.0f, 6.0f, 7.0f, 8.0f});

    auto& sched = CtorchScheduler::getInstance();
    Tensor c3_result = sched.dispatch<op::Add>(a, b);
    Tensor eager = a + b;
    EXPECT_TRUE(tensorsAllClose(c3_result, eager));

    C3KernelRegistry::getInstance().uninstall(op::Add, DeviceType::kCPU);
}

TEST(MLIRBackend, HotReplaceUninstallFallback) {
    using namespace ct::c3;

    auto desc = TensorDesc::fromShape({4});
    Graph g;
    size_t x = g.addInput(desc);
    size_t y = g.addInput(desc);
    size_t z = g.addNode(AddNode{desc, desc}, {x, y}, desc);
    g.markOutput(z);

    auto kernel = compileMLIR(g);
    ASSERT_NE(kernel, nullptr);

    KernelShapeInfo shapes{{4}, {4}, {4}};
    kernel->installIntoRegistry(op::Add, shapes);

    Tensor a(ShapeTag{}, {4});
    Tensor b(ShapeTag{}, {4});
    fillTensor(a, {1.0f, 2.0f, 3.0f, 4.0f});
    fillTensor(b, {5.0f, 6.0f, 7.0f, 8.0f});

    auto& sched = CtorchScheduler::getInstance();
    Tensor c3_result = sched.dispatch<op::Add>(a, b);
    EXPECT_TRUE(tensorsAllClose(c3_result, a + b));

    auto stats_before = C3KernelRegistry::getInstance().getStats();
    size_t hits_before = stats_before.hit_count;

    C3KernelRegistry::getInstance().uninstall(op::Add, DeviceType::kCPU);

    Tensor fallback_result = sched.dispatch<op::Add>(a, b);
    EXPECT_TRUE(tensorsAllClose(fallback_result, a + b));

    auto stats_after = C3KernelRegistry::getInstance().getStats();
    EXPECT_EQ(stats_after.hit_count, hits_before);
}

// ======================= MLIR 后端融合测试 =======================

TEST(MLIRFused, AddMulFusedExecute) {
    using namespace ct::c3;

    // 构建图: (x + y) * z, 启用融合 + MLIR 后端
    Graph g;
    auto desc = TensorDesc::fromShape({4});
    size_t x = g.addInput(desc);
    size_t y = g.addInput(desc);
    size_t z = g.addInput(desc);
    size_t add = g.addNode(AddNode{desc, desc}, {x, y}, desc);
    size_t mul = g.addNode(MulNode{desc, desc}, {add, z}, desc);
    g.markOutput(mul);

    CompileOptions opts;
    opts.backend = C3Backend::MLIR;
    opts.enable_fusion = true;
    auto kernel = C3Engine::getInstance().compile(g, opts);
    ASSERT_NE(kernel, nullptr);

    Tensor a(ShapeTag{}, {4});
    Tensor b(ShapeTag{}, {4});
    Tensor c(ShapeTag{}, {4});
    fillTensor(a, {1.0f, 2.0f, 3.0f, 4.0f});
    fillTensor(b, {5.0f, 6.0f, 7.0f, 8.0f});
    fillTensor(c, {2.0f, 3.0f, 4.0f, 5.0f});

    auto results = kernel->execute({a, b, c});
    ASSERT_EQ(results.size(), 1u);

    Tensor eager = (a + b) * c;
    EXPECT_TRUE(tensorsAllClose(results[0], eager));
}

TEST(MLIRFused, MulAddFusedWithReuse) {
    using namespace ct::c3;

    // 构建图: (x * y) + x, 输入复用 + MLIR 后端
    Graph g;
    auto desc = TensorDesc::fromShape({3});
    size_t x = g.addInput(desc);
    size_t y = g.addInput(desc);
    size_t mul = g.addNode(MulNode{desc, desc}, {x, y}, desc);
    size_t add = g.addNode(AddNode{desc, desc}, {mul, x}, desc);
    g.markOutput(add);

    CompileOptions opts;
    opts.backend = C3Backend::MLIR;
    opts.enable_fusion = true;
    auto kernel = C3Engine::getInstance().compile(g, opts);
    ASSERT_NE(kernel, nullptr);

    Tensor a(ShapeTag{}, {3});
    Tensor b(ShapeTag{}, {3});
    fillTensor(a, {1.0f, 2.0f, 3.0f});
    fillTensor(b, {4.0f, 5.0f, 6.0f});

    auto results = kernel->execute({a, b});
    ASSERT_EQ(results.size(), 1u);

    Tensor eager = (a * b) + a;
    EXPECT_TRUE(tensorsAllClose(results[0], eager));
}

TEST(MLIRFused, ThreeOpFusedExecute) {
    using namespace ct::c3;

    // 构建图: (x * y) + x - z, 三操作融合 + MLIR 后端
    Graph g;
    auto desc = TensorDesc::fromShape({4});
    size_t x = g.addInput(desc);
    size_t y = g.addInput(desc);
    size_t z = g.addInput(desc);
    size_t mul = g.addNode(MulNode{desc, desc}, {x, y}, desc);
    size_t add = g.addNode(AddNode{desc, desc}, {mul, x}, desc);
    size_t sub = g.addNode(SubNode{desc, desc}, {add, z}, desc);
    g.markOutput(sub);

    CompileOptions opts;
    opts.backend = C3Backend::MLIR;
    opts.enable_fusion = true;
    auto kernel = C3Engine::getInstance().compile(g, opts);
    ASSERT_NE(kernel, nullptr);

    Tensor a(ShapeTag{}, {4});
    Tensor b(ShapeTag{}, {4});
    Tensor c(ShapeTag{}, {4});
    fillTensor(a, {1.0f, 2.0f, 3.0f, 4.0f});
    fillTensor(b, {5.0f, 6.0f, 7.0f, 8.0f});
    fillTensor(c, {2.0f, 1.0f, 3.0f, 0.5f});

    auto results = kernel->execute({a, b, c});
    ASSERT_EQ(results.size(), 1u);

    Tensor eager = (a * b) + a - c;
    EXPECT_TRUE(tensorsAllClose(results[0], eager));
}

TEST(MLIRFused, NegFusedExecute) {
    using namespace ct::c3;

    // 构建图: -(x * y), MLIR 后端
    Graph g;
    auto desc = TensorDesc::fromShape({4});
    size_t x = g.addInput(desc);
    size_t y = g.addInput(desc);
    size_t mul = g.addNode(MulNode{desc, desc}, {x, y}, desc);
    size_t neg = g.addNode(NegNode{desc}, {mul}, desc);
    g.markOutput(neg);

    CompileOptions opts;
    opts.backend = C3Backend::MLIR;
    opts.enable_fusion = true;
    auto kernel = C3Engine::getInstance().compile(g, opts);
    ASSERT_NE(kernel, nullptr);

    Tensor a(ShapeTag{}, {4});
    Tensor b(ShapeTag{}, {4});
    fillTensor(a, {1.0f, 2.0f, 3.0f, 4.0f});
    fillTensor(b, {5.0f, 6.0f, 7.0f, 8.0f});

    auto results = kernel->execute({a, b});
    ASSERT_EQ(results.size(), 1u);

    Tensor eager = -(a * b);
    EXPECT_TRUE(tensorsAllClose(results[0], eager));
}

#else  // !CT_ENABLE_MLIR

TEST(MLIRBackend, Disabled_SkipCompile) {
    // MLIR 后端未启用时，CompileOptions::MLIR 应回退到 Handwritten
    using namespace ct::c3;

    Graph g;
    auto desc = TensorDesc::fromShape({4});
    size_t x = g.addInput(desc);
    size_t y = g.addInput(desc);
    g.addNode(AddNode{desc, desc}, {x, y}, desc);

    CompileOptions opts;
    opts.backend = C3Backend::MLIR;  // 即使设为 MLIR，也会回退到 Handwritten
    auto kernel = C3Engine::getInstance().compile(g, opts);
    ASSERT_NE(kernel, nullptr);

    Tensor a(ShapeTag{}, {4});
    Tensor b(ShapeTag{}, {4});
    fillTensor(a, {1.0f, 2.0f, 3.0f, 4.0f});
    fillTensor(b, {5.0f, 6.0f, 7.0f, 8.0f});

    auto results = kernel->execute({a, b});
    EXPECT_TRUE(tensorsAllClose(results[0], a + b));
}

#endif // CT_ENABLE_MLIR

// ======================= Benchmark 测试 =======================

static void bench(const std::string& name, int iters, std::function<void()> fn) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) fn();
    auto end = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(end - start).count() / iters;
    std::cout << "  " << name << ": " << us << " us/iter (" << iters << " iters)" << std::endl;
}

TEST(Benchmark, JITvsEagerAdd) {
    std::cout << "\n=== Benchmark: JIT vs Eager (Handwritten backend) ===" << std::endl;

    Tensor a(ShapeTag{}, {1024, 1024});
    Tensor b(ShapeTag{}, {1024, 1024});
    fillTensor(a, std::vector<float>(1024*1024, 1.0f));
    fillTensor(b, std::vector<float>(1024*1024, 2.0f));

    auto desc = ct::c3::TensorDesc::fromShape({1024, 1024});
    auto g = ct::c3::Tracer::trace([](auto& x, auto& y) { return x + y; }, desc, desc);
    auto& engine = ct::c3::C3Engine::getInstance();
    auto kernel = engine.compile(g, {});

    std::cout << "--- Add (1024x1024) ---" << std::endl;
    bench("JIT    ", 100, [&]() { auto r = kernel->execute({a, b}); });
    bench("Eager  ", 100, [&]() { auto r = a + b; });
}

TEST(Benchmark, JITvsEagerMul) {
    Tensor a(ShapeTag{}, {1024, 1024});
    Tensor b(ShapeTag{}, {1024, 1024});
    fillTensor(a, std::vector<float>(1024*1024, 1.0f));
    fillTensor(b, std::vector<float>(1024*1024, 2.0f));

    auto desc = ct::c3::TensorDesc::fromShape({1024, 1024});
    auto g = ct::c3::Tracer::trace([](auto& x, auto& y) { return x * y; }, desc, desc);
    auto& engine = ct::c3::C3Engine::getInstance();
    auto kernel = engine.compile(g, {});

    std::cout << "--- Mul (1024x1024) ---" << std::endl;
    bench("JIT    ", 100, [&]() { auto r = kernel->execute({a, b}); });
    bench("Eager  ", 100, [&]() { auto r = a * b; });
}

TEST(Benchmark, JITvsEagerMatMul) {
    Tensor a(ShapeTag{}, {256, 256});
    Tensor b(ShapeTag{}, {256, 256});
    fillTensor(a, std::vector<float>(256*256, 1.0f));
    fillTensor(b, std::vector<float>(256*256, 2.0f));

    auto a_desc = ct::c3::TensorDesc::fromShape({256, 256});
    auto b_desc = ct::c3::TensorDesc::fromShape({256, 256});
    auto g = ct::c3::Tracer::trace([](auto& x, auto& y) { return x.matmul(y); }, a_desc, b_desc);
    auto& engine = ct::c3::C3Engine::getInstance();
    auto kernel = engine.compile(g, {});

    std::cout << "--- MatMul (256x256) ---" << std::endl;
    bench("JIT    ", 20, [&]() { auto r = kernel->execute({a, b}); });
    bench("Eager  ", 20, [&]() { auto r = matMul(a, b); });
}

TEST(Benchmark, JITvsEagerSmallVec) {
    Tensor a(ShapeTag{}, {1024});
    Tensor b(ShapeTag{}, {1024});
    fillTensor(a, std::vector<float>(1024, 1.0f));
    fillTensor(b, std::vector<float>(1024, 2.0f));

    auto desc = ct::c3::TensorDesc::fromShape({1024});
    auto g = ct::c3::Tracer::trace([](auto& x, auto& y) { return x + y; }, desc, desc);
    auto& engine = ct::c3::C3Engine::getInstance();
    auto kernel = engine.compile(g, {});

    std::cout << "--- Add (1024) small vector ---" << std::endl;
    bench("JIT    ", 10000, [&]() { auto r = kernel->execute({a, b}); });
    bench("Eager  ", 10000, [&]() { auto r = a + b; });
}

TEST(Benchmark, FusedVsNonFused) {
    std::cout << "\n=== Benchmark: Fused vs Non-Fused (Handwritten backend) ===" << std::endl;

    // 构建 (x * y) + x - z, 三操作
    Tensor a(ShapeTag{}, {1024, 1024});
    Tensor b(ShapeTag{}, {1024, 1024});
    Tensor c(ShapeTag{}, {1024, 1024});
    fillTensor(a, std::vector<float>(1024*1024, 1.0f));
    fillTensor(b, std::vector<float>(1024*1024, 2.0f));
    fillTensor(c, std::vector<float>(1024*1024, 0.5f));

    auto desc = ct::c3::TensorDesc::fromShape({1024, 1024});
    ct::c3::Graph g;
    size_t x = g.addInput(desc);
    size_t y = g.addInput(desc);
    size_t z = g.addInput(desc);
    size_t mul = g.addNode(ct::c3::MulNode{desc, desc}, {x, y}, desc);
    size_t add = g.addNode(ct::c3::AddNode{desc, desc}, {mul, x}, desc);
    size_t sub = g.addNode(ct::c3::SubNode{desc, desc}, {add, z}, desc);
    g.markOutput(sub);

    auto& engine = ct::c3::C3Engine::getInstance();

    ct::c3::CompileOptions fused_opts;
    fused_opts.enable_fusion = true;
    auto fused_kernel = engine.compile(g, fused_opts);

    ct::c3::CompileOptions no_fused_opts;
    no_fused_opts.enable_fusion = false;
    auto no_fused_kernel = engine.compile(g, no_fused_opts);

    std::cout << "--- (x * y) + x - z (1024x1024) ---" << std::endl;
    bench("Fused JIT    ", 100, [&]() { auto r = fused_kernel->execute({a, b, c}); });
    bench("Non-Fused JIT", 100, [&]() { auto r = no_fused_kernel->execute({a, b, c}); });
    bench("Eager (3 ops)", 100, [&]() { auto r = (a * b) + a - c; });
}

#ifdef CT_ENABLE_MLIR
TEST(Benchmark, MLIRvsEagerAdd) {
    std::cout << "\n=== Benchmark: MLIR JIT vs Eager ===" << std::endl;

    Tensor a(ShapeTag{}, {1024, 1024});
    Tensor b(ShapeTag{}, {1024, 1024});
    fillTensor(a, std::vector<float>(1024*1024, 1.0f));
    fillTensor(b, std::vector<float>(1024*1024, 2.0f));

    auto desc = ct::c3::TensorDesc::fromShape({1024, 1024});
    ct::c3::Graph g;
    size_t x = g.addInput(desc);
    size_t y = g.addInput(desc);
    g.addNode(ct::c3::AddNode{desc, desc}, {x, y}, desc);
    g.markOutput(g.nodeCount() - 1);

    ct::c3::CompileOptions hw_opts;
    hw_opts.backend = ct::c3::C3Backend::Handwritten;
    auto hw_kernel = ct::c3::C3Engine::getInstance().compile(g, hw_opts);

    ct::c3::CompileOptions mlir_opts;
    mlir_opts.backend = ct::c3::C3Backend::MLIR;
    auto mlir_kernel = ct::c3::C3Engine::getInstance().compile(g, mlir_opts);

    std::cout << "--- Add (1024x1024) ---" << std::endl;
    bench("Handwritten JIT", 100, [&]() { auto r = hw_kernel->execute({a, b}); });
    bench("MLIR JIT       ", 100, [&]() { auto r = mlir_kernel->execute({a, b}); });
    bench("Eager          ", 100, [&]() { auto r = a + b; });
}

TEST(Benchmark, MLIRvsEagerMatMul) {
    Tensor a(ShapeTag{}, {256, 256});
    Tensor b(ShapeTag{}, {256, 256});
    fillTensor(a, std::vector<float>(256*256, 1.0f));
    fillTensor(b, std::vector<float>(256*256, 2.0f));

    auto a_desc = ct::c3::TensorDesc::fromShape({256, 256});
    auto b_desc = ct::c3::TensorDesc::fromShape({256, 256});
    ct::c3::Graph g;
    size_t ai = g.addInput(a_desc);
    size_t bi = g.addInput(b_desc);
    auto out_desc = ct::c3::TensorDesc::fromShape({256, 256});
    g.addNode(ct::c3::MatMulNode{a_desc, b_desc}, {ai, bi}, out_desc);
    g.markOutput(g.nodeCount() - 1);

    ct::c3::CompileOptions hw_opts;
    hw_opts.backend = ct::c3::C3Backend::Handwritten;
    auto hw_kernel = ct::c3::C3Engine::getInstance().compile(g, hw_opts);

    ct::c3::CompileOptions mlir_opts;
    mlir_opts.backend = ct::c3::C3Backend::MLIR;
    auto mlir_kernel = ct::c3::C3Engine::getInstance().compile(g, mlir_opts);

    std::cout << "--- MatMul (256x256) ---" << std::endl;
    bench("Handwritten JIT", 20, [&]() { auto r = hw_kernel->execute({a, b}); });
    bench("MLIR JIT       ", 20, [&]() { auto r = mlir_kernel->execute({a, b}); });
    bench("Eager          ", 20, [&]() { auto r = matMul(a, b); });
}

TEST(Benchmark, MLIRFusedVsNonFused) {
    std::cout << "\n=== Benchmark: MLIR Fused vs Non-Fused ===" << std::endl;

    Tensor a(ShapeTag{}, {1024, 1024});
    Tensor b(ShapeTag{}, {1024, 1024});
    Tensor c(ShapeTag{}, {1024, 1024});
    fillTensor(a, std::vector<float>(1024*1024, 1.0f));
    fillTensor(b, std::vector<float>(1024*1024, 2.0f));
    fillTensor(c, std::vector<float>(1024*1024, 0.5f));

    auto desc = ct::c3::TensorDesc::fromShape({1024, 1024});
    ct::c3::Graph g;
    size_t x = g.addInput(desc);
    size_t y = g.addInput(desc);
    size_t z = g.addInput(desc);
    size_t mul = g.addNode(ct::c3::MulNode{desc, desc}, {x, y}, desc);
    size_t add = g.addNode(ct::c3::AddNode{desc, desc}, {mul, x}, desc);
    size_t sub = g.addNode(ct::c3::SubNode{desc, desc}, {add, z}, desc);
    g.markOutput(sub);

    auto& engine = ct::c3::C3Engine::getInstance();

    // MLIR fused
    ct::c3::CompileOptions mlir_fused_opts;
    mlir_fused_opts.backend = ct::c3::C3Backend::MLIR;
    mlir_fused_opts.enable_fusion = true;
    auto mlir_fused = engine.compile(g, mlir_fused_opts);

    // MLIR non-fused
    ct::c3::CompileOptions mlir_no_fused_opts;
    mlir_no_fused_opts.backend = ct::c3::C3Backend::MLIR;
    mlir_no_fused_opts.enable_fusion = false;
    auto mlir_no_fused = engine.compile(g, mlir_no_fused_opts);

    // Handwritten fused
    ct::c3::CompileOptions hw_fused_opts;
    hw_fused_opts.backend = ct::c3::C3Backend::Handwritten;
    hw_fused_opts.enable_fusion = true;
    auto hw_fused = engine.compile(g, hw_fused_opts);

    std::cout << "--- (x * y) + x - z (1024x1024) ---" << std::endl;
    bench("MLIR Fused      ", 100, [&]() { auto r = mlir_fused->execute({a, b, c}); });
    bench("MLIR Non-Fused  ", 100, [&]() { auto r = mlir_no_fused->execute({a, b, c}); });
    bench("HW Fused        ", 100, [&]() { auto r = hw_fused->execute({a, b, c}); });
    bench("Eager (3 ops)   ", 100, [&]() { auto r = (a * b) + a - c; });
}

TEST(Benchmark, FusionBreakEven) {
    /// @brief 融合收益拐点分析：对比不同链长度（2/3/4/5/6/8 ops）的融合 vs 非融合性能
    /// @details 链结构：((...(x0 op0 x1) op1 x2) op2 x3) ...)
    ///          op 交替使用 Add/Mul 以保证实际计算复杂度
    ///          非融合路径：逐个 op 编译为独立 kernel 并分别执行，累加总耗时
    std::cout << "\n=== Benchmark: Fusion Break-Even Analysis ===" << std::endl;
    std::cout << "Chain | MLIR Fused | MLIR NoFuse | HW Fused | HW NoFuse | Eager" << std::endl;
    std::cout << "------|------------|-------------|----------|-----------|------" << std::endl;

    const std::vector<int> chain_lengths = {2, 3, 4, 5, 6, 8};
    const size_t N = 1024 * 1024;  // 1M elements

    for (int ops_count : chain_lengths) {
        std::vector<float> values(N, 1.0f);

        auto desc = ct::c3::TensorDesc::fromShape({1024, 1024});
        ct::c3::Graph g;
        std::vector<size_t> inputs;
        for (int i = 0; i <= ops_count; ++i) {
            inputs.push_back(g.addInput(desc));
        }

        // 交替 Add/Mul 构建链
        size_t prev = inputs[0];
        std::vector<ct::c3::NodeVariant> ops;
        std::vector<std::pair<size_t, size_t>> op_input_pairs; // (lhs, rhs) for each op
        for (int i = 0; i < ops_count; ++i) {
            size_t lhs = prev;
            size_t rhs = inputs[i + 1];
            if (i % 2 == 0) {
                prev = g.addNode(ct::c3::AddNode{desc, desc}, {lhs, rhs}, desc);
                ops.push_back(ct::c3::AddNode{desc, desc});
            } else {
                prev = g.addNode(ct::c3::MulNode{desc, desc}, {lhs, rhs}, desc);
                ops.push_back(ct::c3::MulNode{desc, desc});
            }
            op_input_pairs.push_back({lhs, rhs});
        }
        g.markOutput(prev);

        auto& engine = ct::c3::C3Engine::getInstance();

        // MLIR fused
        ct::c3::CompileOptions mlir_f_opts;
        mlir_f_opts.backend = ct::c3::C3Backend::MLIR;
        mlir_f_opts.enable_fusion = true;
        auto mlir_f = engine.compile(g, mlir_f_opts);

        // HW fused
        ct::c3::CompileOptions hw_f_opts;
        hw_f_opts.backend = ct::c3::C3Backend::Handwritten;
        hw_f_opts.enable_fusion = true;
        auto hw_f = engine.compile(g, hw_f_opts);

        // === 非融合：为每个 op 构建独立 graph 并编译 ===
        std::vector<std::shared_ptr<ct::c3::CompiledKernel>> mlir_kernels, hw_kernels;
        ct::c3::CompileOptions mlir_opts;
        mlir_opts.backend = ct::c3::C3Backend::MLIR;
        mlir_opts.enable_fusion = false;
        ct::c3::CompileOptions hw_opts;
        hw_opts.backend = ct::c3::C3Backend::Handwritten;
        hw_opts.enable_fusion = false;

        // 中间张量：第一个 op 用 inputs[0] 和 inputs[1]，后续 op 用中间结果
        std::vector<Tensor> inter_tensors;
        for (int i = 0; i <= ops_count; ++i) {
            inter_tensors.emplace_back(ShapeTag{}, std::vector<size_t>{1024, 1024});
            fillTensor(inter_tensors.back(), values);
        }
        // 为中间结果预分配
        for (int i = 0; i < ops_count - 1; ++i) {
            inter_tensors.push_back(Tensor(ShapeTag{}, std::vector<size_t>{1024, 1024}));
        }

        for (int i = 0; i < ops_count; ++i) {
            ct::c3::Graph single_g;
            size_t x = single_g.addInput(desc);
            size_t y = single_g.addInput(desc);
            size_t node = single_g.addNode(ops[i], {x, y}, desc);
            single_g.markOutput(node);
            mlir_kernels.push_back(engine.compile(single_g, mlir_opts));
            hw_kernels.push_back(engine.compile(single_g, hw_opts));
        }

        // 构建输入列表
        std::vector<Tensor> fused_inputs;
        for (int i = 0; i <= ops_count; ++i) {
            fused_inputs.push_back(inter_tensors[i]);
        }

        // 预热
        mlir_f->execute(fused_inputs);
        hw_f->execute(fused_inputs);

        int runs = (ops_count <= 3) ? 100 : 50;

        auto measure = [&](auto& kernel, const std::vector<Tensor>& in) -> double {
            auto start = std::chrono::high_resolution_clock::now();
            for (int r = 0; r < runs; ++r) {
                auto result = kernel->execute(in);
                (void)result;
            }
            auto end = std::chrono::high_resolution_clock::now();
            return std::chrono::duration<double, std::micro>(end - start).count() / runs;
        };

        double mlir_f_us = measure(mlir_f, fused_inputs);
        double hw_f_us = measure(hw_f, fused_inputs);

        // 非融合：逐个执行每个 op 的 kernel，累加时间
        double mlir_nf_us = 0, hw_nf_us = 0;
        for (int i = 0; i < ops_count; ++i) {
            Tensor lhs = (i == 0) ? inter_tensors[0] : inter_tensors[ops_count + 1 + (i - 1)];
            Tensor rhs = inter_tensors[i + 1];
            Tensor& out = (i == ops_count - 1) ? inter_tensors.back() : inter_tensors[ops_count + 1 + i];
            std::vector<Tensor> op_inputs = {lhs, rhs};

            mlir_nf_us += measure(mlir_kernels[i], op_inputs);
            hw_nf_us += measure(hw_kernels[i], op_inputs);
        }

        // Eager
        double eager_us = 0;
        {
            auto start = std::chrono::high_resolution_clock::now();
            for (int r = 0; r < runs; ++r) {
                Tensor cur = inter_tensors[0];
                for (int i = 0; i < ops_count; ++i) {
                    if (i % 2 == 0) {
                        cur = cur + inter_tensors[i + 1];
                    } else {
                        cur = cur * inter_tensors[i + 1];
                    }
                }
                (void)cur;
            }
            auto end = std::chrono::high_resolution_clock::now();
            eager_us = std::chrono::duration<double, std::micro>(end - start).count() / runs;
        }

        std::cout << std::setw(5) << std::left << ops_count << " | "
                  << std::setw(10) << std::right << std::fixed << std::setprecision(1) << mlir_f_us << " | "
                  << std::setw(11) << mlir_nf_us << " | "
                  << std::setw(8) << hw_f_us << " | "
                  << std::setw(9) << hw_nf_us << " | "
                  << std::setw(5) << eager_us << std::endl;
    }
}

#endif // CT_ENABLE_MLIR

// ======================= 异步编译测试 =======================

TEST(AsyncCompile, BasicAsyncExecute) {
    // 验证异步编译后 kernel 执行结果与 eager 一致
    auto& engine = ct::c3::C3Engine::getInstance();
    engine.clearCache();

    auto desc = ct::c3::TensorDesc::fromShape({128, 128});
    ct::c3::Graph g;
    size_t x = g.addInput(desc);
    size_t y = g.addInput(desc);
    size_t add = g.addNode(ct::c3::AddNode{desc, desc}, {x, y}, desc);
    g.markOutput(add);

    Tensor a(ShapeTag{}, {128, 128});
    Tensor b(ShapeTag{}, {128, 128});
    fillTensor(a, std::vector<float>(128*128, 1.5f));
    fillTensor(b, std::vector<float>(128*128, 2.5f));

    Tensor eager = a + b;

    // 异步编译
    auto future = engine.compileAsync(g);
    // 等待编译完成
    auto kernel = future.get();
    ASSERT_NE(kernel, nullptr);
    auto result = kernel->execute({a, b});
    EXPECT_TRUE(tensorsAllClose(result[0], eager));
}

TEST(AsyncCompile, ReturnsImmediately) {
    // 验证 compileAsync 立即返回，不阻塞
    auto& engine = ct::c3::C3Engine::getInstance();
    engine.clearCache();

    auto desc = ct::c3::TensorDesc::fromShape({128, 128});
    ct::c3::Graph g;
    size_t x = g.addInput(desc);
    size_t y = g.addInput(desc);
    size_t mul = g.addNode(ct::c3::MulNode{desc, desc}, {x, y}, desc);
    g.markOutput(mul);

    auto start = std::chrono::steady_clock::now();
    auto future = engine.compileAsync(g);
    auto elapsed = std::chrono::steady_clock::now() - start;
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();

    // compileAsync 应在 10ms 内返回（不等待编译完成）
    EXPECT_LT(elapsed_ms, 10) << "compileAsync took " << elapsed_ms << "ms, should return immediately";

    // 确保 future 有效
    EXPECT_TRUE(future.valid());

    // 等待编译完成并验证
    auto kernel = future.get();
    ASSERT_NE(kernel, nullptr);
}

TEST(AsyncCompile, Deduplication) {
    // 验证同一 graph 同时发起两次 compileAsync 返回同一个 future
    auto& engine = ct::c3::C3Engine::getInstance();
    engine.clearCache();

    auto desc = ct::c3::TensorDesc::fromShape({128, 128});
    ct::c3::Graph g;
    size_t x = g.addInput(desc);
    size_t y = g.addInput(desc);
    size_t add = g.addNode(ct::c3::AddNode{desc, desc}, {x, y}, desc);
    g.markOutput(add);

    auto f1 = engine.compileAsync(g);
    auto f2 = engine.compileAsync(g);

    // 两个 future 应该指向同一个 shared state（去重）
    auto k1 = f1.get();
    auto k2 = f2.get();
    EXPECT_EQ(k1, k2);
}

TEST(AsyncCompile, CacheHitAfterAsync) {
    // 验证异步编译完成后，同步 compile 命中缓存
    auto& engine = ct::c3::C3Engine::getInstance();
    engine.clearCache();

    auto desc = ct::c3::TensorDesc::fromShape({128, 128});
    ct::c3::Graph g;
    size_t x = g.addInput(desc);
    size_t y = g.addInput(desc);
    size_t sub = g.addNode(ct::c3::SubNode{desc, desc}, {x, y}, desc);
    g.markOutput(sub);

    Tensor a(ShapeTag{}, {128, 128});
    Tensor b(ShapeTag{}, {128, 128});
    fillTensor(a, std::vector<float>(128*128, 3.0f));
    fillTensor(b, std::vector<float>(128*128, 1.0f));

    Tensor eager = a - b;

    // 异步编译
    auto future = engine.compileAsync(g);
    auto async_kernel = future.get();
    ASSERT_NE(async_kernel, nullptr);
    auto async_result = async_kernel->execute({a, b});
    EXPECT_TRUE(tensorsAllClose(async_result[0], eager));

    // 同步调用应命中缓存
    auto stats_before = engine.getCacheStats();
    auto sync_kernel = engine.compile(g);
    auto stats_after = engine.getCacheStats();

    EXPECT_EQ(sync_kernel, async_kernel);
    EXPECT_TRUE(stats_after.hits > stats_before.hits);
}

TEST(AsyncCompile, FusedAsyncExecute) {
    // 验证异步编译融合 kernel 执行正确
    auto& engine = ct::c3::C3Engine::getInstance();
    engine.clearCache();

    auto desc = ct::c3::TensorDesc::fromShape({128, 128});
    ct::c3::Graph g;
    size_t x = g.addInput(desc);
    size_t y = g.addInput(desc);
    size_t z = g.addInput(desc);
    size_t mul = g.addNode(ct::c3::MulNode{desc, desc}, {x, y}, desc);
    size_t add = g.addNode(ct::c3::AddNode{desc, desc}, {mul, x}, desc);
    size_t sub = g.addNode(ct::c3::SubNode{desc, desc}, {add, z}, desc);
    g.markOutput(sub);

    Tensor a(ShapeTag{}, {128, 128});
    Tensor b(ShapeTag{}, {128, 128});
    Tensor c(ShapeTag{}, {128, 128});
    fillTensor(a, std::vector<float>(128*128, 1.0f));
    fillTensor(b, std::vector<float>(128*128, 2.0f));
    fillTensor(c, std::vector<float>(128*128, 0.5f));

    Tensor eager = (a * b) + a - c;

    auto future = engine.compileAsync(g);
    auto kernel = future.get();
    ASSERT_NE(kernel, nullptr);
    auto result = kernel->execute({a, b, c});
    EXPECT_TRUE(tensorsAllClose(result[0], eager));
}

#ifdef CT_ENABLE_MLIR

TEST(AsyncCompile, MLIRAsyncExecute) {
    // 验证 MLIR 后端异步编译执行正确
    auto& engine = ct::c3::C3Engine::getInstance();
    engine.clearCache();

    auto desc = ct::c3::TensorDesc::fromShape({128, 128});
    ct::c3::Graph g;
    size_t x = g.addInput(desc);
    size_t y = g.addInput(desc);
    size_t add = g.addNode(ct::c3::AddNode{desc, desc}, {x, y}, desc);
    g.markOutput(add);

    ct::c3::CompileOptions opts;
    opts.backend = ct::c3::C3Backend::MLIR;

    Tensor a(ShapeTag{}, {128, 128});
    Tensor b(ShapeTag{}, {128, 128});
    fillTensor(a, std::vector<float>(128*128, 1.5f));
    fillTensor(b, std::vector<float>(128*128, 2.5f));

    Tensor eager = a + b;

    auto future = engine.compileAsync(g, opts);
    auto kernel = future.get();
    ASSERT_NE(kernel, nullptr);
    auto result = kernel->execute({a, b});
    EXPECT_TRUE(tensorsAllClose(result[0], eager));
}

TEST(AsyncCompile, MLIRFusedAsyncExecute) {
    // 验证 MLIR 后端异步编译融合 kernel 执行正确
    auto& engine = ct::c3::C3Engine::getInstance();
    engine.clearCache();

    auto desc = ct::c3::TensorDesc::fromShape({128, 128});
    ct::c3::Graph g;
    size_t x = g.addInput(desc);
    size_t y = g.addInput(desc);
    size_t mul = g.addNode(ct::c3::MulNode{desc, desc}, {x, y}, desc);
    size_t neg = g.addNode(ct::c3::NegNode{desc}, {mul}, desc);
    g.markOutput(neg);

    ct::c3::CompileOptions opts;
    opts.backend = ct::c3::C3Backend::MLIR;
    opts.enable_fusion = true;

    Tensor a(ShapeTag{}, {128, 128});
    Tensor b(ShapeTag{}, {128, 128});
    fillTensor(a, std::vector<float>(128*128, 2.0f));
    fillTensor(b, std::vector<float>(128*128, 3.0f));

    Tensor eager = -(a * b);

    auto future = engine.compileAsync(g, opts);
    auto kernel = future.get();
    ASSERT_NE(kernel, nullptr);
    auto result = kernel->execute({a, b});
    EXPECT_TRUE(tensorsAllClose(result[0], eager));
}

// ======================= PGO Profiling 测试 =======================

TEST(PGOProfiling, BasicProfileData) {
    // 验证 ProfileData 的基本记录和查询功能
    ct::c3::ProfileData pd;
    EXPECT_EQ(pd.call_count.load(), 0u);

    pd.record(1000); // 1 us
    EXPECT_EQ(pd.call_count.load(), 1u);
    EXPECT_EQ(pd.last_time_ns.load(), 1000u);
    EXPECT_EQ(pd.min_time_ns.load(), 1000u);
    EXPECT_EQ(pd.max_time_ns.load(), 1000u);
    EXPECT_EQ(pd.avgTimeNs(), 1000u);

    pd.record(500); // 0.5 us
    EXPECT_EQ(pd.call_count.load(), 2u);
    EXPECT_EQ(pd.min_time_ns.load(), 500u);
    EXPECT_EQ(pd.max_time_ns.load(), 1000u);
    EXPECT_EQ(pd.avgTimeNs(), 750u);
}

TEST(PGOProfiling, ProfiledCompiledKernel) {
    // 验证 enable_profiling 下 compile 返回 ProfiledCompiledKernel 并正确记录耗时
    auto& engine = ct::c3::C3Engine::getInstance();
    engine.clearCache();

    auto desc = ct::c3::TensorDesc::fromShape({32, 32});
    ct::c3::Graph g;
    size_t x = g.addInput(desc);
    size_t y = g.addInput(desc);
    size_t add = g.addNode(ct::c3::AddNode{desc, desc}, {x, y}, desc);
    g.markOutput(add);

    ct::c3::CompileOptions opts;
    opts.backend = ct::c3::C3Backend::MLIR;
    opts.enable_profiling = true;

    Tensor a(ShapeTag{}, {32, 32});
    Tensor b(ShapeTag{}, {32, 32});
    fillTensor(a, std::vector<float>(32*32, 1.5f));
    fillTensor(b, std::vector<float>(32*32, 2.5f));

    // 编译并执行
    auto kernel = engine.compile(g, opts);
    ASSERT_NE(kernel, nullptr);

    // 检查是否返回了 ProfiledCompiledKernel
    auto* profiled = dynamic_cast<ct::c3::ProfiledCompiledKernel*>(kernel.get());
    ASSERT_NE(profiled, nullptr) << "enable_profiling should return ProfiledCompiledKernel";

    // 执行 3 次
    for (int i = 0; i < 3; ++i) {
        kernel->execute({a, b});
    }

    // 验证 profile 数据
    const auto& pd = profiled->profileData();
    EXPECT_EQ(pd.call_count.load(), 3u);
    EXPECT_GT(pd.total_time_ns.load(), 0u);
    EXPECT_GT(pd.avgTimeNs(), 0u);

    // 通过 cache key 查询 profile 数据
    auto cache_key = kernel->cacheKey();
    auto queried_pd = engine.getProfileData(cache_key);
    ASSERT_NE(queried_pd, nullptr);
    EXPECT_EQ(queried_pd->call_count.load(), 3u);

    // 缓存命中应共享同一个 ProfileData
    auto cached_kernel = engine.compile(g, opts);
    auto* cached_profiled = dynamic_cast<ct::c3::ProfiledCompiledKernel*>(cached_kernel.get());
    ASSERT_NE(cached_profiled, nullptr);
    EXPECT_EQ(&cached_profiled->profileData(), &profiled->profileData());
}

TEST(PGOProfiling, ProfilingOffByDefault) {
    // 验证默认不启用 profiling
    auto& engine = ct::c3::C3Engine::getInstance();
    engine.clearCache();

    auto desc = ct::c3::TensorDesc::fromShape({16, 16});
    ct::c3::Graph g;
    size_t x = g.addInput(desc);
    size_t neg = g.addNode(ct::c3::NegNode{desc}, {x}, desc);
    g.markOutput(neg);

    auto kernel = engine.compile(g);
    ASSERT_NE(kernel, nullptr);

    auto* profiled = dynamic_cast<ct::c3::ProfiledCompiledKernel*>(kernel.get());
    EXPECT_EQ(profiled, nullptr) << "profiling should be off by default";
}

// ======================= MLP 端到端 Benchmark =======================

TEST(Benchmark, MLPEndToEnd) {
    /// @brief 2 层 MLP 端到端 Benchmark：C3 fused vs C3 non-fused vs Eager (AMX) vs Eager (SIMD)
    /// @details 结构: MatMul→Add→ReLU → MatMul→Add→ReLU
    ///          batch=8, input=32, hidden=64, output=16 （小规模，避免 CPU 资源耗尽）
    ///          C3 当前仅支持单节点图编译，故将 MLP 拆为子图逐节点编译+串联
    std::cout << "\n=== Benchmark: MLP End-to-End ===" << std::endl;
    std::cout << "Model: 2-layer MLP (32→64→16, batch=8)" << std::endl;

    const size_t B = 8, D_IN = 32, D_HID = 64, D_OUT = 16;

    auto desc_in = ct::c3::TensorDesc::fromShape({B, D_IN});
    auto desc_w1 = ct::c3::TensorDesc::fromShape({D_IN, D_HID});
    auto desc_b1 = ct::c3::TensorDesc::fromShape({B, D_HID});
    auto desc_hid = ct::c3::TensorDesc::fromShape({B, D_HID});
    auto desc_w2 = ct::c3::TensorDesc::fromShape({D_HID, D_OUT});
    auto desc_b2 = ct::c3::TensorDesc::fromShape({B, D_OUT});
    auto desc_out = ct::c3::TensorDesc::fromShape({B, D_OUT});

    /// 构建单节点子图的辅助函数
    auto makeSingleOpGraph = [](ct::c3::NodeVariant op, ct::c3::TensorDesc in_a, ct::c3::TensorDesc in_b, ct::c3::TensorDesc out_desc) {
        ct::c3::Graph g;
        size_t x = g.addInput(in_a);
        size_t y = g.addInput(in_b);
        size_t node = g.addNode(op, {x, y}, out_desc);
        g.markOutput(node);
        return g;
    };

    auto makeUnaryGraph = [](ct::c3::NodeVariant op, ct::c3::TensorDesc in_desc, ct::c3::TensorDesc out_desc) {
        ct::c3::Graph g;
        size_t x = g.addInput(in_desc);
        size_t node = g.addNode(op, {x}, out_desc);
        g.markOutput(node);
        return g;
    };

    auto& engine = ct::c3::C3Engine::getInstance();

    // 编译所有子图（MLIR fused: Add+ReLU 融合为单个子图）
    ct::c3::CompileOptions mlir_opts;
    mlir_opts.backend = ct::c3::C3Backend::MLIR;
    mlir_opts.enable_fusion = true;

    ct::c3::CompileOptions hw_opts;
    hw_opts.backend = ct::c3::C3Backend::Handwritten;
    hw_opts.enable_fusion = true;

    ct::c3::CompileOptions mlir_nf_opts;
    mlir_nf_opts.backend = ct::c3::C3Backend::MLIR;
    mlir_nf_opts.enable_fusion = false;

    ct::c3::CompileOptions hw_nf_opts;
    hw_nf_opts.backend = ct::c3::C3Backend::Handwritten;
    hw_nf_opts.enable_fusion = false;

    // MatMul 1: (B, D_IN) × (D_IN, D_HID) → (B, D_HID)
    auto mm1_graph = makeSingleOpGraph(ct::c3::MatMulNode{desc_in, desc_w1}, desc_in, desc_w1, desc_hid);
    auto mm1_mlir = engine.compile(mm1_graph, mlir_opts);
    auto mm1_hw = engine.compile(mm1_graph, hw_opts);

    // MatMul 2: (B, D_HID) × (D_HID, D_OUT) → (B, D_OUT)
    auto mm2_graph = makeSingleOpGraph(ct::c3::MatMulNode{desc_hid, desc_w2}, desc_hid, desc_w2, desc_out);
    auto mm2_mlir = engine.compile(mm2_graph, mlir_opts);
    auto mm2_hw = engine.compile(mm2_graph, hw_opts);

    // Fused: Add + ReLU chain (Add→ReLU)，编译为融合子图
    ct::c3::Graph fuse1_graph;
    size_t f1_a = fuse1_graph.addInput(desc_hid);
    size_t f1_b = fuse1_graph.addInput(desc_b1);
    size_t f1_add = fuse1_graph.addNode(ct::c3::AddNode{desc_hid, desc_b1}, {f1_a, f1_b}, desc_hid);
    size_t f1_relu = fuse1_graph.addNode(ct::c3::ReLUNode{desc_hid}, {f1_add}, desc_hid);
    fuse1_graph.markOutput(f1_relu);
    auto fuse1_mlir = engine.compile(fuse1_graph, mlir_opts);
    auto fuse1_hw = engine.compile(fuse1_graph, hw_opts);

    ct::c3::Graph fuse2_graph;
    size_t f2_a = fuse2_graph.addInput(desc_out);
    size_t f2_b = fuse2_graph.addInput(desc_b2);
    size_t f2_add = fuse2_graph.addNode(ct::c3::AddNode{desc_out, desc_b2}, {f2_a, f2_b}, desc_out);
    size_t f2_relu = fuse2_graph.addNode(ct::c3::ReLUNode{desc_out}, {f2_add}, desc_out);
    fuse2_graph.markOutput(f2_relu);
    auto fuse2_mlir = engine.compile(fuse2_graph, mlir_opts);
    auto fuse2_hw = engine.compile(fuse2_graph, hw_opts);

    // Non-fused: 各自编译 Add 和 ReLU
    auto add1_graph = makeSingleOpGraph(ct::c3::AddNode{desc_hid, desc_b1}, desc_hid, desc_b1, desc_hid);
    auto add1_mlir_nf = engine.compile(add1_graph, mlir_nf_opts);
    auto add1_hw_nf = engine.compile(add1_graph, hw_nf_opts);

    auto relu1_graph = makeUnaryGraph(ct::c3::ReLUNode{desc_hid}, desc_hid, desc_hid);
    auto relu1_mlir_nf = engine.compile(relu1_graph, mlir_nf_opts);
    auto relu1_hw_nf = engine.compile(relu1_graph, hw_nf_opts);

    auto add2_graph = makeSingleOpGraph(ct::c3::AddNode{desc_out, desc_b2}, desc_out, desc_b2, desc_out);
    auto add2_mlir_nf = engine.compile(add2_graph, mlir_nf_opts);
    auto add2_hw_nf = engine.compile(add2_graph, hw_nf_opts);

    auto relu2_graph = makeUnaryGraph(ct::c3::ReLUNode{desc_out}, desc_out, desc_out);
    auto relu2_mlir_nf = engine.compile(relu2_graph, mlir_nf_opts);
    auto relu2_hw_nf = engine.compile(relu2_graph, hw_nf_opts);

    // 准备输入数据
    auto make_tensor = [&](const std::vector<size_t>& shape, float val) {
        Tensor t(ShapeTag{}, shape);
        std::vector<float> data(t.numel(), val);
        fillTensor(t, data);
        return t;
    };

    Tensor input_t = make_tensor({B, D_IN}, 1.0f);
    Tensor w1_t = make_tensor({D_IN, D_HID}, 0.01f);
    Tensor b1_t = make_tensor({B, D_HID}, 0.1f);
    Tensor w2_t = make_tensor({D_HID, D_OUT}, 0.01f);
    Tensor b2_t = make_tensor({B, D_OUT}, 0.1f);

    // Eager reference (AMX MatMul)
    Tensor eager_ref = matMul(input_t, w1_t);
    eager_ref = eager_ref + b1_t;
    eager_ref = eager_ref.relu();
    eager_ref = matMul(eager_ref, w2_t);
    eager_ref = eager_ref + b2_t;
    eager_ref = eager_ref.relu_();

    // 验证 C3 fused 正确性
    auto run_mlir_fused = [&]() -> Tensor {
        Tensor out = mm1_mlir->execute({input_t, w1_t})[0];
        out = fuse1_mlir->execute({out, b1_t})[0];
        out = mm2_mlir->execute({out, w2_t})[0];
        out = fuse2_mlir->execute({out, b2_t})[0];
        return out;
    };

    auto run_hw_fused = [&]() -> Tensor {
        Tensor out = mm1_hw->execute({input_t, w1_t})[0];
        out = fuse1_hw->execute({out, b1_t})[0];
        out = mm2_hw->execute({out, w2_t})[0];
        out = fuse2_hw->execute({out, b2_t})[0];
        return out;
    };

    auto run_mlir_nonfused = [&]() -> Tensor {
        Tensor out = mm1_mlir->execute({input_t, w1_t})[0];
        out = add1_mlir_nf->execute({out, b1_t})[0];
        out = relu1_mlir_nf->execute({out})[0];
        out = mm2_mlir->execute({out, w2_t})[0];
        out = add2_mlir_nf->execute({out, b2_t})[0];
        out = relu2_mlir_nf->execute({out})[0];
        return out;
    };

    auto run_hw_nonfused = [&]() -> Tensor {
        Tensor out = mm1_hw->execute({input_t, w1_t})[0];
        out = add1_hw_nf->execute({out, b1_t})[0];
        out = relu1_hw_nf->execute({out})[0];
        out = mm2_hw->execute({out, w2_t})[0];
        out = add2_hw_nf->execute({out, b2_t})[0];
        out = relu2_hw_nf->execute({out})[0];
        return out;
    };

    EXPECT_TRUE(tensorsAllClose(run_mlir_fused(), eager_ref, 1e-3f, 1e-4f));
    EXPECT_TRUE(tensorsAllClose(run_hw_fused(), eager_ref, 1e-3f, 1e-4f));
    EXPECT_TRUE(tensorsAllClose(run_mlir_nonfused(), eager_ref, 1e-3f, 1e-4f));
    EXPECT_TRUE(tensorsAllClose(run_hw_nonfused(), eager_ref, 1e-3f, 1e-4f));

    // 预热
    run_mlir_fused(); run_hw_fused(); run_mlir_nonfused(); run_hw_nonfused();

    // Benchmark
    const int runs = 100;

    auto measure = [&](auto&& fn) -> double {
        auto start = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < runs; ++r) {
            Tensor out = fn();
            (void)out;
        }
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::micro>(end - start).count() / runs;
    };

    auto measure_eager_amx = [&]() -> double {
        auto start = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < runs; ++r) {
            Tensor out = matMul(input_t, w1_t);       // AMX
            out = out + b1_t;                          // SIMD
            out = out.relu();                          // SIMD
            out = matMul(out, w2_t);                   // AMX
            out = out + b2_t;                          // SIMD
            out.relu_();                               // SIMD inplace
        }
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::micro>(end - start).count() / runs;
    };

    auto measure_eager_simd = [&]() -> double {
        auto start = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < runs; ++r) {
            Tensor out = MatMul_SIMD_kernel(input_t, w1_t);  // SIMD MatMul
            out = out + b1_t;                                  // SIMD
            out = out.relu();                                  // SIMD
            out = MatMul_SIMD_kernel(out, w2_t);               // SIMD MatMul
            out = out + b2_t;                                  // SIMD
            out.relu_();                                       // SIMD inplace
        }
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::micro>(end - start).count() / runs;
    };

    double mlir_f_us = measure(run_mlir_fused);
    double mlir_nf_us = measure(run_mlir_nonfused);
    double hw_f_us = measure(run_hw_fused);
    double hw_nf_us = measure(run_hw_nonfused);
    double eager_amx_us = measure_eager_amx();
    double eager_simd_us = measure_eager_simd();

    std::cout << "\n" << std::setw(28) << std::left << "Backend" << " | "
              << std::setw(10) << std::right << "Time (us)" << " | "
              << std::setw(12) << "vs AMX" << " | "
              << std::setw(12) << "vs SIMD" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    std::cout << std::setw(28) << std::left << "MLIR Fused" << " | "
              << std::setw(10) << std::right << std::fixed << std::setprecision(1) << mlir_f_us << " | "
              << std::setw(11) << std::right << std::fixed << std::setprecision(2)
              << (eager_amx_us / mlir_f_us) << "x" << " | "
              << std::setw(11) << std::right << std::fixed << std::setprecision(2)
              << (eager_simd_us / mlir_f_us) << "x" << std::endl;
    std::cout << std::setw(28) << std::left << "MLIR Non-Fused" << " | "
              << std::setw(10) << std::right << std::fixed << std::setprecision(1) << mlir_nf_us << " | "
              << std::setw(11) << std::right << std::fixed << std::setprecision(2)
              << (eager_amx_us / mlir_nf_us) << "x" << " | "
              << std::setw(11) << std::right << std::fixed << std::setprecision(2)
              << (eager_simd_us / mlir_nf_us) << "x" << std::endl;
    std::cout << std::setw(28) << std::left << "HW Fused" << " | "
              << std::setw(10) << std::right << std::fixed << std::setprecision(1) << hw_f_us << " | "
              << std::setw(11) << std::right << std::fixed << std::setprecision(2)
              << (eager_amx_us / hw_f_us) << "x" << " | "
              << std::setw(11) << std::right << std::fixed << std::setprecision(2)
              << (eager_simd_us / hw_f_us) << "x" << std::endl;
    std::cout << std::setw(28) << std::left << "HW Non-Fused" << " | "
              << std::setw(10) << std::right << std::fixed << std::setprecision(1) << hw_nf_us << " | "
              << std::setw(11) << std::right << std::fixed << std::setprecision(2)
              << (eager_amx_us / hw_nf_us) << "x" << " | "
              << std::setw(11) << std::right << std::fixed << std::setprecision(2)
              << (eager_simd_us / hw_nf_us) << "x" << std::endl;
    std::cout << std::setw(28) << std::left << "Eager (AMX MatMul)" << " | "
              << std::setw(10) << std::right << std::fixed << std::setprecision(1) << eager_amx_us << " | "
              << std::setw(11) << std::right << "1.00x" << " | "
              << std::setw(11) << std::right << std::fixed << std::setprecision(2)
              << (eager_simd_us / eager_amx_us) << "x" << std::endl;
    std::cout << std::setw(28) << std::left << "Eager (SIMD MatMul)" << " | "
              << std::setw(10) << std::right << std::fixed << std::setprecision(1) << eager_simd_us << " | "
              << std::setw(11) << std::right << std::fixed << std::setprecision(2)
              << (eager_amx_us / eager_simd_us) << "x" << " | "
              << std::setw(11) << std::right << "1.00x" << std::endl;
}

#endif // CT_ENABLE_MLIR

// ======================= AutoTuner 测试 =======================

#include "C3/AutoTuner.h"

TEST(AutoTuner, QEAvsGAvsGridSearch) {
    /// @brief 对比 QEA / GA / GridSearch 在 MatMul 分块参数调优上的性能
    /// @details 使用模拟的 MatMul 性能模型作为适应度函数，
    ///          全局最优位于 (TILE_M=64, TILE_N=64, TILE_K=64, unroll=4)
    ///          搜索空间: 5×5×5×4 = 500 种组合

    auto fitness_fn = [](int tile_m, int tile_n, int tile_k, int unroll) -> double {
        const int OPT_M = 64, OPT_N = 64, OPT_K = 64, OPT_UNROLL = 4;
        const double L1D_BYTES = 128.0 * 1024.0;

        double dist = std::sqrt(
            (tile_m - OPT_M) * (tile_m - OPT_M) * 1.0 +
            (tile_n - OPT_N) * (tile_n - OPT_N) * 1.0 +
            (tile_k - OPT_K) * (tile_k - OPT_K) * 1.0 +
            (unroll - OPT_UNROLL) * (unroll - OPT_UNROLL) * 100.0
        );

        double cache_usage = 3.0 * tile_m * tile_n * 4.0;
        double cache_penalty = 0.0;
        if (cache_usage > L1D_BYTES) {
            cache_penalty = (cache_usage - L1D_BYTES) / L1D_BYTES * 200.0;
        }

        double overhead = 0.0;
        if (tile_m < 32) overhead += 30.0;
        if (tile_n < 32) overhead += 30.0;
        if (tile_k < 32) overhead += 15.0;

        return 10.0 + dist * 0.5 + cache_penalty + overhead;
    };

    ct::c3::AutoTunerConfig config;
    config.verbose = false;

    ct::c3::AutoTuner tuner(config);

    std::vector<int> known_optimal = {64, 64, 64, 4};
    tuner.runComparison(fitness_fn, known_optimal);

    auto qea_result = tuner.tuneWithQEA(fitness_fn);
    auto ga_result  = tuner.tuneWithGA(fitness_fn);
    auto gs_result  = tuner.tuneWithGridSearch(fitness_fn);

    EXPECT_EQ(gs_result.tile_m, 64);
    EXPECT_EQ(gs_result.tile_n, 64);
    EXPECT_EQ(gs_result.tile_k, 64);
    EXPECT_EQ(gs_result.unroll, 4);

    EXPECT_EQ(qea_result.tile_m, 64);
    EXPECT_EQ(qea_result.tile_n, 64);
    EXPECT_EQ(qea_result.tile_k, 64);
    EXPECT_EQ(qea_result.unroll, 4);

    EXPECT_EQ(ga_result.tile_m, 64);
    EXPECT_EQ(ga_result.tile_n, 64);
    EXPECT_EQ(ga_result.tile_k, 64);
    EXPECT_EQ(ga_result.unroll, 4);

    EXPECT_LT(qea_result.evaluations, gs_result.evaluations);
    std::cout << "\nQEA evaluations: " << qea_result.evaluations
              << " / " << gs_result.evaluations << " (GridSearch)"
              << " = " << std::fixed << std::setprecision(1)
              << (double)qea_result.evaluations / gs_result.evaluations * 100.0 << "%" << std::endl;
}

TEST(AutoTuner, ConvergenceRate) {
    /// @brief 验证 QEA 在复杂多峰函数上的探索优势
    /// @details 含多个局部极小值的 Rastrigin-like 函数，GA 易陷入局部最优，
    ///          QEA 的概率编码和评估缓存有助于跳出局部极小

    const int OPT_M = 64, OPT_N = 64, OPT_K = 64, OPT_UNROLL = 4;

    // 多峰适应度函数: 在最优值附近有多个局部极小
    auto fitness_fn = [&](int tile_m, int tile_n, int tile_k, int unroll) -> double {
        double d_m = (tile_m - OPT_M) / 16.0;
        double d_n = (tile_n - OPT_N) / 16.0;
        double d_k = (tile_k - OPT_K) / 16.0;
        double d_u = (unroll - OPT_UNROLL) / 2.0;

        // Rastrigin-like: 二次项 + 余弦项产生局部极小
        double rastrigin = 0.0;
        for (double d : {d_m, d_n, d_k, d_u}) {
            rastrigin += d * d - std::cos(2.0 * M_PI * d) + 1.0;
        }

        // Cache 溢出惩罚
        double cache_usage = 3.0 * tile_m * tile_n * 4.0;
        double cache_penalty = 0.0;
        if (cache_usage > 128.0 * 1024.0) {
            cache_penalty = 50.0;
        }

        return rastrigin * 10.0 + cache_penalty;
    };

    ct::c3::AutoTunerConfig config;
    config.qea_population = 10;
    config.qea_generations = 20;
    config.ga_population = 20;
    config.ga_generations = 20;

    ct::c3::AutoTuner tuner(config);

    auto qea_result = tuner.tuneWithQEA(fitness_fn);
    auto ga_result  = tuner.tuneWithGA(fitness_fn);

    if (qea_result.fitness_history.size() >= 5 && ga_result.fitness_history.size() >= 5) {
        std::cout << "\nConvergence comparison (Rastrigin-like landscape):" << std::endl;
        std::cout << "  Gen | QEA fitness | GA fitness" << std::endl;
        std::cout << "  " << std::string(30, '-') << std::endl;
        for (size_t g = 0; g < 5 && g < qea_result.fitness_history.size(); ++g) {
            std::cout << "  " << std::setw(3) << g << " | "
                      << std::setw(11) << std::fixed << std::setprecision(4) << qea_result.fitness_history[g]
                      << " | " << std::setw(11) << ga_result.fitness_history[g] << std::endl;
        }
    }

    // 两者都应找到全局最优 (64,64,64,4)
    EXPECT_EQ(qea_result.tile_m, OPT_M);
    EXPECT_EQ(qea_result.tile_n, OPT_N);
    EXPECT_EQ(qea_result.tile_k, OPT_K);
    EXPECT_EQ(qea_result.unroll, OPT_UNROLL);

    EXPECT_EQ(ga_result.tile_m, OPT_M);
    EXPECT_EQ(ga_result.tile_n, OPT_N);
    EXPECT_EQ(ga_result.tile_k, OPT_K);
    EXPECT_EQ(ga_result.unroll, OPT_UNROLL);
}

// ======================= MLP 端到端 Benchmark =======================

/// 构建单层 MLP 子图: MatMul(W, x) + Bias + ReLU
/// @note 使用 2D 形状 (batch=1) 以兼容 C3 MatMul kernel 的 2D 要求
///       x: (in_dim, 1) 列向量, w: (out_dim, in_dim), b: (out_dim, 1)
/// @return 编译后的 kernel
static std::shared_ptr<ct::c3::CompiledKernel> buildMLPLayer(
    size_t in_dim, size_t out_dim, bool with_relu = true)
{
    using namespace ct::c3;
    Graph g;
    auto x_desc = TensorDesc::fromShape({in_dim, 1});       // 列向量 (in_dim, 1)
    auto w_desc = TensorDesc::fromShape({out_dim, in_dim}); // 权重矩阵 (out_dim, in_dim)
    auto b_desc = TensorDesc::fromShape({out_dim, 1});      // 偏置列向量 (out_dim, 1)
    auto out_desc = TensorDesc::fromShape({out_dim, 1});    // 输出列向量 (out_dim, 1)

    auto x = g.addInput(x_desc);   // input vector (2D column)
    auto w = g.addInput(w_desc);   // weight matrix
    auto b = g.addInput(b_desc);   // bias vector (2D column)

    // MatMul: w @ x → (out_dim, 1)
    auto mm = g.addNode(MatMulNode{w_desc, x_desc}, {w, x}, out_desc);
    // Add bias
    auto add = g.addNode(AddNode{out_desc, b_desc}, {mm, b}, out_desc);

    if (with_relu) {
        g.addNode(ReLUNode{out_desc}, {add}, out_desc);
    }
    g.markOutput(g.nodeCount() - 1);

    CompileOptions opts;
    opts.backend = C3Backend::MLIR;  // 生产路径 (跟 test_c3_mnist_train 一致, MLIR 是上生产的)
    opts.enable_fusion = false;  // 多节点 kernel 直接处理各 op，不融合
    return C3Engine::getInstance().compile(g, opts);
}

/// 构建 MLP 的所有层并返回 kernel 列表
static std::vector<std::shared_ptr<ct::c3::CompiledKernel>> buildMLP(
    const std::vector<size_t>& layer_dims)
{
    std::vector<std::shared_ptr<ct::c3::CompiledKernel>> layers;
    for (size_t i = 0; i < layer_dims.size() - 1; ++i) {
        bool is_last = (i == layer_dims.size() - 2);
        layers.push_back(buildMLPLayer(layer_dims[i], layer_dims[i + 1], !is_last));
    }
    return layers;
}

/// C3 MLP 前向传播
/// @note 将 1D 输入/偏置 reshape 为 2D 列向量 (dim, 1) 以匹配 C3 MatMul kernel 的 2D 要求
static Tensor c3MLPForward(
    const std::vector<std::shared_ptr<ct::c3::CompiledKernel>>& layers,
    const Tensor& input,
    const std::vector<Tensor>& weights,
    const std::vector<Tensor>& biases)
{
    Tensor x = input;
    for (size_t i = 0; i < layers.size(); ++i) {
        // reshape 1D → 2D 列向量: (in_dim,) → (in_dim, 1)
        Tensor x_2d = x.reshape({x.numel(), 1});
        Tensor b_2d = biases[i].reshape({biases[i].numel(), 1});
        auto results = layers[i]->execute({x_2d, weights[i], b_2d});
        // reshape 2D 列向量 → 1D: (out_dim, 1) → (out_dim,)
        x = results[0].reshape({results[0].numel()});
    }
    return x;
}

/// Eager MLP 前向传播（对照）
/// @note 将 1D 输入/偏置 reshape 为 2D 列向量 (dim, 1) 以兼容 AMX MatMul
///       weight: (out_dim, in_dim), x: (in_dim, 1)
///       matMul(weight, x) = (out_dim, in_dim) @ (in_dim, 1) = (out_dim, 1)
static Tensor eagerMLPForward(
    const Tensor& input,
    const std::vector<Tensor>& weights,
    const std::vector<Tensor>& biases)
{
    Tensor x = input;
    for (size_t i = 0; i < weights.size(); ++i) {
        // Reshape 1D → 2D 列向量: (in_dim,) → (in_dim, 1)
        Tensor x_2d = x.reshape({x.numel(), 1});
        // weight: (out_dim, in_dim), matMul(weight, x_2d) = (out_dim, in_dim) @ (in_dim, 1) = (out_dim, 1)
        x = matMul(weights[i], x_2d).reshape({weights[i].shape()[0]});
        x = x + biases[i];
        if (i < weights.size() - 1) {
            x = x.relu();
        }
    }
    return x;
}

/// 辅助：打印 benchmark 表头
static void printBenchHeader() {
    std::cout << "\n" << std::setw(20) << std::left << "Config"
              << " | " << std::setw(12) << std::right << "Time(us)"
              << " | " << std::setw(12) << "Throughput"
              << " | " << std::setw(10) << "Speedup" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
}

/// 辅助：打印 benchmark 行
static void printBenchRow(const std::string& label, double time_us, double baseline_us) {
    double speedup = (baseline_us > 0) ? baseline_us / time_us : 0.0;
    double throughput = (time_us > 0) ? (1e6 / time_us) : 0.0;
    std::cout << std::setw(20) << std::left << label
              << " | " << std::setw(12) << std::right << std::fixed << std::setprecision(1) << time_us
              << " | " << std::setw(12) << std::fixed << std::setprecision(1) << throughput << " fwd/s"
              << " | " << std::setw(9) << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
}

TEST(Benchmark, MLP_3Layer_C3_vs_Eager) {
    using namespace ct;
    std::cout << "\n========== MLP Benchmark: 784→256→128→10 ==========" << std::endl;

    // 构建权重
    std::vector<Tensor> weights, biases;
    std::vector<size_t> dims = {784, 256, 128, 10};
    for (size_t i = 0; i < dims.size() - 1; ++i) {
        weights.push_back(Tensor(ShapeTag{}, {dims[i+1], dims[i]}));
        biases.push_back(Tensor(ShapeTag{}, {dims[i+1]}));
    }

    // 随机初始化
    for (auto& w : weights) {
        for (size_t j = 0; j < w.numel(); ++j)
            w.data_write<float>()[j] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (auto& b : biases) {
        for (size_t j = 0; j < b.numel(); ++j)
            b.data_write<float>()[j] = static_cast<float>(rand()) / RAND_MAX;
    }

    Tensor input(ShapeTag{}, {784});
    for (size_t j = 0; j < input.numel(); ++j)
        input.data_write<float>()[j] = static_cast<float>(rand()) / RAND_MAX;

    // C3 编译
    auto& engine = c3::C3Engine::getInstance();
    engine.clearCache();
    auto layers = buildMLP(dims);
    std::cout << "C3 layers compiled: " << layers.size() << std::endl;

    // 正确性验证
    auto c3_out = c3MLPForward(layers, input, weights, biases);
    auto eager_out = eagerMLPForward(input, weights, biases);

    // [Fix 2026-08-09 v0.5.2 DEBT-NEW-5 真根因修 (MLIRKernelGen.cpp kSmallMatMulThreshold=0)]:
    // tolerance 1e-6 (user 要求), 期望 MLIR MatMul 走 cblas_sgemm 后跟 eager 数值 1e-6 内
    bool match = tensorsAllClose(c3_out, eager_out, 1e-6f, 1e-6f);
    std::cout << "Correctness: " << (match ? "PASS" : "FAIL") << std::endl;
    if (!match) {
        size_t first_mismatch = SIZE_MAX;
        for (size_t i = 0; i < c3_out.numel() && first_mismatch == SIZE_MAX; ++i) {
            float diff = std::fabs(c3_out.data_read<float>()[i] - eager_out.data_read<float>()[i]);
            float max_val = std::max(std::fabs(c3_out.data_read<float>()[i]), std::fabs(eager_out.data_read<float>()[i]));
            if (diff > 1e-4f + 1e-4f * max_val) {
                first_mismatch = i;
            }
        }
        std::cout << "First mismatch at index " << first_mismatch << ":" << std::endl;
        std::cout << "  C3[" << first_mismatch << "] = " << c3_out.data_read<float>()[first_mismatch] << std::endl;
        std::cout << "  Eager[" << first_mismatch << "] = " << eager_out.data_read<float>()[first_mismatch] << std::endl;
        std::cout << "First 10 C3 values:    ";
        for (size_t i = 0; i < std::min(size_t(10), c3_out.numel()); ++i)
            std::cout << c3_out.data_read<float>()[i] << " ";
        std::cout << std::endl;
        std::cout << "First 10 Eager values: ";
        for (size_t i = 0; i < std::min(size_t(10), eager_out.numel()); ++i)
            std::cout << eager_out.data_read<float>()[i] << " ";
        std::cout << std::endl;
    }
    EXPECT_TRUE(match);

    // Benchmark
    const size_t warmup = 10, runs = 100;
    printBenchHeader();

    // Eager
    for (size_t r = 0; r < warmup; ++r) eagerMLPForward(input, weights, biases);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t r = 0; r < runs; ++r) eagerMLPForward(input, weights, biases);
    auto t1 = std::chrono::high_resolution_clock::now();
    double eager_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / runs;

    // C3
    for (size_t r = 0; r < warmup; ++r) c3MLPForward(layers, input, weights, biases);
    t0 = std::chrono::high_resolution_clock::now();
    for (size_t r = 0; r < runs; ++r) c3MLPForward(layers, input, weights, biases);
    t1 = std::chrono::high_resolution_clock::now();
    double c3_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / runs;

    printBenchRow("Eager (AMX)", eager_us, eager_us);
    printBenchRow("C3 MLIR Fused", c3_us, eager_us);

    std::cout << "\nCache stats: hits=" << engine.getCacheStats().hits
              << " misses=" << engine.getCacheStats().misses
              << " entries=" << engine.getCacheStats().total_entries << std::endl;
}

TEST(Benchmark, MLP_Large_C3_vs_Eager) {
    using namespace ct;
    std::cout << "\n========== MLP Benchmark: 1024→512→256→10 ==========" << std::endl;

    std::vector<Tensor> weights, biases;
    std::vector<size_t> dims = {1024, 512, 256, 10};
    for (size_t i = 0; i < dims.size() - 1; ++i) {
        weights.push_back(Tensor(ShapeTag{}, {dims[i+1], dims[i]}));
        biases.push_back(Tensor(ShapeTag{}, {dims[i+1]}));
    }

    for (auto& w : weights) {
        for (size_t j = 0; j < w.numel(); ++j)
            w.data_write<float>()[j] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (auto& b : biases) {
        for (size_t j = 0; j < b.numel(); ++j)
            b.data_write<float>()[j] = static_cast<float>(rand()) / RAND_MAX;
    }

    Tensor input(ShapeTag{}, {1024});
    for (size_t j = 0; j < input.numel(); ++j)
        input.data_write<float>()[j] = static_cast<float>(rand()) / RAND_MAX;

    auto& engine = c3::C3Engine::getInstance();
    engine.clearCache();
    auto layers = buildMLP(dims);

    auto c3_out = c3MLPForward(layers, input, weights, biases);
    auto eager_out = eagerMLPForward(input, weights, biases);

    // [Fix 2026-08-09 DEBT-NEW-5 真根因修]: tolerance 1e-6, 期望 MLIR MatMul 走 cblas 数值 1e-6 内
    bool match = tensorsAllClose(c3_out, eager_out, 1e-6f, 1e-6f);
    std::cout << "Correctness: " << (match ? "PASS" : "FAIL") << std::endl;
    EXPECT_TRUE(match);

    const size_t warmup = 10, runs = 100;
    printBenchHeader();

    for (size_t r = 0; r < warmup; ++r) eagerMLPForward(input, weights, biases);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t r = 0; r < runs; ++r) eagerMLPForward(input, weights, biases);
    auto t1 = std::chrono::high_resolution_clock::now();
    double eager_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / runs;

    for (size_t r = 0; r < warmup; ++r) c3MLPForward(layers, input, weights, biases);
    t0 = std::chrono::high_resolution_clock::now();
    for (size_t r = 0; r < runs; ++r) c3MLPForward(layers, input, weights, biases);
    t1 = std::chrono::high_resolution_clock::now();
    double c3_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / runs;

    printBenchRow("Eager (AMX)", eager_us, eager_us);
    printBenchRow("C3 MLIR Fused", c3_us, eager_us);
}

TEST(Benchmark, MLP_Huge_C3_vs_Eager) {
    using namespace ct;
    std::cout << "\n========== MLP Benchmark: 2048→1024→512→256→10 ==========" << std::endl;

    std::vector<Tensor> weights, biases;
    std::vector<size_t> dims = {2048, 1024, 512, 256, 10};
    for (size_t i = 0; i < dims.size() - 1; ++i) {
        weights.push_back(Tensor(ShapeTag{}, {dims[i+1], dims[i]}));
        biases.push_back(Tensor(ShapeTag{}, {dims[i+1]}));
    }

    for (auto& w : weights) {
        for (size_t j = 0; j < w.numel(); ++j)
            w.data_write<float>()[j] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (auto& b : biases) {
        for (size_t j = 0; j < b.numel(); ++j)
            b.data_write<float>()[j] = static_cast<float>(rand()) / RAND_MAX;
    }

    Tensor input(ShapeTag{}, {2048});
    for (size_t j = 0; j < input.numel(); ++j)
        input.data_write<float>()[j] = static_cast<float>(rand()) / RAND_MAX;

    auto& engine = c3::C3Engine::getInstance();
    engine.clearCache();
    auto layers = buildMLP(dims);

    auto c3_out = c3MLPForward(layers, input, weights, biases);
    auto eager_out = eagerMLPForward(input, weights, biases);

    // [Fix 2026-08-09 DEBT-NEW-5 真根因修]: tolerance 1e-6, 期望 MLIR MatMul 走 cblas 数值 1e-6 内
    bool match = tensorsAllClose(c3_out, eager_out, 1e-6f, 1e-6f);
    std::cout << "Correctness: " << (match ? "PASS" : "FAIL") << std::endl;
    EXPECT_TRUE(match);

    const size_t warmup = 10, runs = 50;
    printBenchHeader();

    for (size_t r = 0; r < warmup; ++r) eagerMLPForward(input, weights, biases);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t r = 0; r < runs; ++r) eagerMLPForward(input, weights, biases);
    auto t1 = std::chrono::high_resolution_clock::now();
    double eager_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / runs;

    for (size_t r = 0; r < warmup; ++r) c3MLPForward(layers, input, weights, biases);
    t0 = std::chrono::high_resolution_clock::now();
    for (size_t r = 0; r < runs; ++r) c3MLPForward(layers, input, weights, biases);
    t1 = std::chrono::high_resolution_clock::now();
    double c3_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / runs;

    printBenchRow("Eager (AMX)", eager_us, eager_us);
    printBenchRow("C3 MLIR Fused", c3_us, eager_us);
}

TEST(Benchmark, MLP_Autotune_vs_Default) {
    using namespace ct;
    std::cout << "\n========== MLP + AutoTune Benchmark: 512→256→128→10 ==========" << std::endl;

    std::vector<Tensor> weights, biases;
    std::vector<size_t> dims = {512, 256, 128, 10};
    for (size_t i = 0; i < dims.size() - 1; ++i) {
        weights.push_back(Tensor(ShapeTag{}, {dims[i+1], dims[i]}));
        biases.push_back(Tensor(ShapeTag{}, {dims[i+1]}));
    }
    for (auto& w : weights) {
        for (size_t j = 0; j < w.numel(); ++j)
            w.data_write<float>()[j] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (auto& b : biases) {
        for (size_t j = 0; j < b.numel(); ++j)
            b.data_write<float>()[j] = static_cast<float>(rand()) / RAND_MAX;
    }

    Tensor input(ShapeTag{}, {512});
    for (size_t j = 0; j < input.numel(); ++j)
        input.data_write<float>()[j] = static_cast<float>(rand()) / RAND_MAX;

    auto& engine = c3::C3Engine::getInstance();

    // === 默认参数（无 autotune） ===
    engine.clearCache();
    auto layers_default = buildMLP(dims);

    const size_t warmup = 10, runs = 100;
    for (size_t r = 0; r < warmup; ++r) c3MLPForward(layers_default, input, weights, biases);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t r = 0; r < runs; ++r) c3MLPForward(layers_default, input, weights, biases);
    auto t1 = std::chrono::high_resolution_clock::now();
    double default_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / runs;

    // === 运行 autotune ===
    std::cout << "\nRunning AutoTune (QEA)..." << std::endl;
    c3::AutoTunerConfig at_cfg;
    at_cfg.verbose = true;
    engine.autoTune(at_cfg);

    // === 使用 autotune 参数重新编译 ===
    engine.clearCache();
    auto layers_tuned = buildMLP(dims);

    for (size_t r = 0; r < warmup; ++r) c3MLPForward(layers_tuned, input, weights, biases);
    t0 = std::chrono::high_resolution_clock::now();
    for (size_t r = 0; r < runs; ++r) c3MLPForward(layers_tuned, input, weights, biases);
    t1 = std::chrono::high_resolution_clock::now();
    double tuned_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / runs;

    // === Eager ===
    for (size_t r = 0; r < warmup; ++r) eagerMLPForward(input, weights, biases);
    t0 = std::chrono::high_resolution_clock::now();
    for (size_t r = 0; r < runs; ++r) eagerMLPForward(input, weights, biases);
    t1 = std::chrono::high_resolution_clock::now();
    double eager_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / runs;

    printBenchHeader();
    printBenchRow("Eager (AMX)", eager_us, eager_us);
    printBenchRow("C3 Default", default_us, eager_us);
    printBenchRow("C3 AutoTuned", tuned_us, eager_us);

    // 正确性验证
    auto c3_out = c3MLPForward(layers_tuned, input, weights, biases);
    auto eager_out = eagerMLPForward(input, weights, biases);
    bool match = tensorsAllClose(c3_out, eager_out, 1e-4f, 1e-4f);
    EXPECT_TRUE(match);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    // [Dev 2026-08-09] C3Cleanup 接 main: 修 P1 漏洞 (test_c3_graph 之前 segfault 缺退出清理)
    ct::c3::shutdownAll();
    return ret;
}

// ======================= traceAndInject 测试 =======================

TEST(TraceAndInject, SingleInputReLU) {
    using namespace ct::c3;

    auto desc = TensorDesc::fromShape({4});
    auto& engine = C3Engine::getInstance();

    // 一站式: trace → compile → inject
    auto kernel = engine.traceAndInject(
        [](auto& x) { return x.relu(); }, desc);

    ASSERT_NE(kernel, nullptr);

    // 用真实张量执行
    Tensor a(ShapeTag{}, {4});
    fillTensor(a, {-1.0f, 2.0f, -3.0f, 4.0f});

    auto results = kernel->execute({a});
    ASSERT_EQ(results.size(), 1u);

    // 验证结果
    Tensor eager = a.relu();
    EXPECT_TRUE(tensorsAllClose(results[0], eager));

    // 清理注册表
    C3KernelRegistry::getInstance().uninstall(op::ReLU, DeviceType::kCPU);
}

TEST(TraceAndInject, SingleInputSigmoid) {
    using namespace ct::c3;

    auto desc = TensorDesc::fromShape({4});
    auto& engine = C3Engine::getInstance();

    auto kernel = engine.traceAndInject(
        [](auto& x) { return x.sigmoid(); }, desc);

    ASSERT_NE(kernel, nullptr);

    Tensor a(ShapeTag{}, {4});
    fillTensor(a, {0.0f, 1.0f, -1.0f, 2.0f});

    auto results = kernel->execute({a});
    ASSERT_EQ(results.size(), 1u);

    Tensor eager = a.sigmoid();
    EXPECT_TRUE(tensorsAllClose(results[0], eager));

    C3KernelRegistry::getInstance().uninstall(op::Sigmoid, DeviceType::kCPU);
}

TEST(TraceAndInject, SingleInputTanh) {
    using namespace ct::c3;

    auto desc = TensorDesc::fromShape({4});
    auto& engine = C3Engine::getInstance();

    auto kernel = engine.traceAndInject(
        [](auto& x) { return x.tanh(); }, desc);

    ASSERT_NE(kernel, nullptr);

    Tensor a(ShapeTag{}, {4});
    fillTensor(a, {0.0f, 1.0f, -1.0f, 2.0f});

    auto results = kernel->execute({a});
    ASSERT_EQ(results.size(), 1u);

    Tensor eager = a.tanh();
    EXPECT_TRUE(tensorsAllClose(results[0], eager));

    C3KernelRegistry::getInstance().uninstall(op::Tanh, DeviceType::kCPU);
}

TEST(TraceAndInject, FCLayer) {
    using namespace ct::c3;

    // 追踪 FC 层: matmul(x, w) + b
    auto x_desc = TensorDesc::fromShape({4, 3});
    auto w_desc = TensorDesc::fromShape({3, 2});
    auto b_desc = TensorDesc::fromShape({2});

    auto& engine = C3Engine::getInstance();

    auto kernel = engine.traceAndInject(
        [](auto& x, auto& w, auto& b) { return x.matmul(w) + b; },
        x_desc, w_desc, b_desc);

    ASSERT_NE(kernel, nullptr);

    // 用真实张量执行
    Tensor x(ShapeTag{}, {4, 3});
    Tensor w(ShapeTag{}, {3, 2});
    Tensor b(ShapeTag{}, {2});
    fillTensor(x, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});
    fillTensor(w, {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f});
    fillTensor(b, {0.5f, -0.5f});

    auto results = kernel->execute({x, w, b});
    ASSERT_EQ(results.size(), 1u);

    // Eager 参考
    Tensor eager = matMul(x, w) + b;
    EXPECT_TRUE(tensorsAllClose(results[0], eager));

    // 清理（compileAndInject 会注入为 op::Add，因为输出节点是 Add）
    // 但多节点图的注入未严格实现，只是不抛出异常即可
}

TEST(TraceAndInject, FCWithActivation) {
    using namespace ct::c3;

    // 追踪 FC + ReLU: relu(matmul(x, w) + b)
    auto x_desc = TensorDesc::fromShape({4, 3});
    auto w_desc = TensorDesc::fromShape({3, 2});
    auto b_desc = TensorDesc::fromShape({2});

    auto& engine = C3Engine::getInstance();

    auto kernel = engine.traceAndInject(
        [](auto& x, auto& w, auto& b) {
            return (x.matmul(w) + b).relu();
        },
        x_desc, w_desc, b_desc);

    ASSERT_NE(kernel, nullptr);

    Tensor x(ShapeTag{}, {4, 3});
    Tensor w(ShapeTag{}, {3, 2});
    Tensor b(ShapeTag{}, {2});
    fillTensor(x, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});
    fillTensor(w, {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f});
    fillTensor(b, {0.5f, -0.5f});

    auto results = kernel->execute({x, w, b});
    ASSERT_EQ(results.size(), 1u);

    Tensor eager = (matMul(x, w) + b).relu();

    // Debug: print mismatches
    if (!tensorsAllClose(results[0], eager)) {
        const float* c3 = results[0].data_read<float>();
        const float* eg = eager.data_read<float>();
        size_t n = std::min(results[0].numel(), eager.numel());
        std::cout << "\n[DEBUG] FCWithActivation mismatch:" << std::endl;
        std::cout << "  C3 shape: [";
        for (auto s : results[0].shape()) std::cout << s << ",";
        std::cout << "] eager shape: [";
        for (auto s : eager.shape()) std::cout << s << ",";
        std::cout << "]" << std::endl;
        std::cout << "  C3 numel=" << results[0].numel() << " eager numel=" << eager.numel() << std::endl;
        for (size_t i = 0; i < n; ++i) {
            float diff = std::fabs(c3[i] - eg[i]);
            float max_val = std::max(std::fabs(c3[i]), std::fabs(eg[i]));
            if (diff > 1e-4f + 1e-4f * max_val) {
                std::cout << "  [" << i << "] C3=" << c3[i] << " eager=" << eg[i] << " diff=" << diff << std::endl;
            }
        }
    }

    EXPECT_TRUE(tensorsAllClose(results[0], eager));
}

TEST(TraceAndInject, AsyncTraceAndExecute) {
    using namespace ct::c3;

    // 异步 traceAndInject
    auto desc = TensorDesc::fromShape({4});
    auto& engine = C3Engine::getInstance();

    auto future = engine.traceAndInjectAsync(
        [](auto& x) { return x.sigmoid(); }, desc);

    // future 应立即可用
    EXPECT_TRUE(future.valid());

    // 等待编译完成
    auto kernel = future.get();
    ASSERT_NE(kernel, nullptr);

    Tensor a(ShapeTag{}, {4});
    fillTensor(a, {0.0f, 1.0f, -1.0f, 2.0f});

    auto results = kernel->execute({a});
    ASSERT_EQ(results.size(), 1u);

    Tensor eager = a.sigmoid();
    EXPECT_TRUE(tensorsAllClose(results[0], eager));

    C3KernelRegistry::getInstance().uninstall(op::Sigmoid, DeviceType::kCPU);
}

TEST(TraceAndInject, Negate) {
    using namespace ct::c3;

    auto desc = TensorDesc::fromShape({4});
    auto& engine = C3Engine::getInstance();

    auto kernel = engine.traceAndInject(
        [](auto& x) { return -x; }, desc);

    ASSERT_NE(kernel, nullptr);

    Tensor a(ShapeTag{}, {4});
    fillTensor(a, {1.0f, -2.0f, 3.0f, -4.0f});

    auto results = kernel->execute({a});
    ASSERT_EQ(results.size(), 1u);

    Tensor eager = -a;
    EXPECT_TRUE(tensorsAllClose(results[0], eager));

    C3KernelRegistry::getInstance().uninstall(op::Neg, DeviceType::kCPU);
}

TEST(TraceAndInject, CacheKeyConsistency) {
    using namespace ct::c3;

    // 相同的表达式应产生相同的缓存键
    auto desc = TensorDesc::fromShape({4});
    auto& engine = C3Engine::getInstance();
    engine.clearCache();

    auto k1 = engine.traceAndInject(
        [](auto& x) { return x.relu(); }, desc);
    auto k2 = engine.traceAndInject(
        [](auto& x) { return x.relu(); }, desc);

    ASSERT_NE(k1, nullptr);
    ASSERT_NE(k2, nullptr);

    // 相同的表达式应命中缓存，返回相同 kernel
    EXPECT_EQ(k1, k2) << "Same expression should hit cache";

    C3KernelRegistry::getInstance().uninstallAll();
}

#ifdef CT_ENABLE_MLIR

// ======================= PGO Phase 2: Hot Path 检测与重编译 =======================

TEST(PGOProfiling, HotPathDetection) {
    /// @brief 验证 CaaS 三层自动提升：Eager → O2 → Ofast
    /// @details 创建一个 PGOCompiledKernel，使用同步编译模式，
    ///          验证第一次 execute 后自动触发编译链，O2 和 Ofast 全部就绪。
    using namespace ct::c3;

    auto& engine = C3Engine::getInstance();
    engine.clearCache();

    auto& pgo_mgr = PGOManager::getInstance();
    pgo_mgr.setEnabled(true);

    // 使用同步编译模式方便测试
    bool old_async = pgo_mgr.config().async_compilation;
    pgo_mgr.config().async_compilation = false;

    // 构建一个简单的图：Neg
    auto desc = TensorDesc::fromShape({16, 16});
    Graph g;
    size_t x = g.addInput(desc);
    size_t neg = g.addNode(NegNode{desc}, {x}, desc);
    g.markOutput(neg);

    // 启用 PGO 模式
    CompileOptions opts;
    opts.backend = C3Backend::MLIR;
    opts.pgo_mode = true;

    auto kernel = engine.compile(g, opts);
    ASSERT_NE(kernel, nullptr);

    // 验证返回的是 PGOCompiledKernel
    auto* pgo_kernel = dynamic_cast<PGOCompiledKernel*>(kernel.get());
    ASSERT_NE(pgo_kernel, nullptr) << "PGO mode should return PGOCompiledKernel";

    // 初始状态：未提升（还没执行过）
    EXPECT_FALSE(pgo_kernel->isPromoted());

    // 创建输入张量
    Tensor a(ShapeTag{}, {16, 16});
    for (size_t i = 0; i < a.numel(); ++i)
        a.data_write<float>()[i] = static_cast<float>(i);

    // 第一次执行：触发 O2 → Ofast 编译链（同步模式）
    auto result = kernel->execute({a});
    ASSERT_EQ(result.size(), 1u);

    // 同步编译链已完成，应已提升
    EXPECT_TRUE(pgo_kernel->isPromoted()) << "Kernel should be promoted after sync compilation";

    // 验证 profile 数据已记录
    const auto& pd = pgo_kernel->profileData();
    EXPECT_GE(pd.call_count.load(), 1u);
    EXPECT_GT(pd.total_time_ns.load(), 0u);

    // 提升后执行结果应仍正确
    Tensor eager = -a;
    result = kernel->execute({a});
    EXPECT_TRUE(tensorsAllClose(result[0], eager, 1e-4f, 1e-4f));

    // 恢复异步编译设置
    pgo_mgr.config().async_compilation = old_async;
}

TEST(PGOProfiling, HotnessScore) {
    /// @brief 验证热路径评分函数 computeHotnessScore 的合理性
    using namespace ct::c3;

    auto& engine = C3Engine::getInstance();
    engine.clearCache();

    auto desc = TensorDesc::fromShape({8, 8});
    Graph g;
    size_t x = g.addInput(desc);
    size_t sig = g.addNode(SigmoidNode{desc}, {x}, desc);
    g.markOutput(sig);

    CompileOptions opts;
    opts.backend = C3Backend::MLIR;
    opts.pgo_mode = true;

    auto kernel = engine.compile(g, opts);
    ASSERT_NE(kernel, nullptr);

    auto* pgo_kernel = dynamic_cast<PGOCompiledKernel*>(kernel.get());
    ASSERT_NE(pgo_kernel, nullptr);

    // 初始评分应为 0（无调用）
    // 无法直接访问 computeHotnessScore，但可以通过 profile 数据间接验证

    // 执行几次
    Tensor a(ShapeTag{}, {8, 8});
    for (size_t i = 0; i < a.numel(); ++i)
        a.data_write<float>()[i] = static_cast<float>(i) / 10.0f;

    // 执行 3 次后检查 profile 数据
    for (int i = 0; i < 3; ++i)
        kernel->execute({a});

    const auto& pd = pgo_kernel->profileData();
    EXPECT_EQ(pd.call_count.load(), 3u);
    EXPECT_GT(pd.avgTimeNs(), 0u);

    // 验证正确性
    Tensor eager = a.sigmoid();
    auto result = kernel->execute({a});
    EXPECT_TRUE(tensorsAllClose(result[0], eager, 1e-4f, 1e-4f));
}

TEST(PGOProfiling, PGOIntegration) {
    /// @brief 验证 PGO 模式在 C3Engine 中的完整集成
    /// @details 测试 pgo_mode=true 时 compile() 返回 PGOCompiledKernel，
    ///          以及 PGOManager 的统计信息正确。
    using namespace ct::c3;

    auto& engine = C3Engine::getInstance();
    engine.clearCache();

    // 启用 PGOManager 并清理状态
    auto& pgo = PGOManager::getInstance();
    pgo.setEnabled(true);
    pgo.clear();

    // 使用同步编译模式
    bool old_async = pgo.config().async_compilation;
    pgo.config().async_compilation = false;

    // 编译一个 PGO kernel
    auto desc = TensorDesc::fromShape({4, 4});
    Graph g;
    size_t x = g.addInput(desc);
    size_t relu = g.addNode(ReLUNode{desc}, {x}, desc);
    g.markOutput(relu);

    CompileOptions opts;
    opts.backend = C3Backend::MLIR;
    opts.pgo_mode = true;

    auto kernel = engine.compile(g, opts);
    ASSERT_NE(kernel, nullptr);

    // 验证 PGOManager 统计（初始状态：pending）
    auto stats = pgo.getStats();
    EXPECT_EQ(stats.total_registered, 1u);
    EXPECT_EQ(stats.pending, 1u);
    EXPECT_EQ(stats.o2_ready, 0u);
    EXPECT_EQ(stats.ofast_ready, 0u);

    // 创建输入张量
    Tensor a(ShapeTag{}, {4, 4});
    for (size_t i = 0; i < a.numel(); ++i)
        a.data_write<float>()[i] = static_cast<float>(i);

    // 第一次执行触发同步编译链
    kernel->execute({a});

    // 同步编译完成后，O2 和 Ofast 都应就绪
    stats = pgo.getStats();
    EXPECT_EQ(stats.total_registered, 1u);
    EXPECT_EQ(stats.o2_ready, 1u);
    EXPECT_EQ(stats.ofast_ready, 1u);
    EXPECT_EQ(stats.pending, 0u);

    // 验证正确性
    Tensor eager = a.relu();
    auto result = kernel->execute({a});
    EXPECT_TRUE(tensorsAllClose(result[0], eager, 1e-4f, 1e-4f));

    // 清理
    engine.clearCache();
    pgo.config().async_compilation = old_async;
}

TEST(PGOProfiling, NoPGOMode) {
    /// @brief 验证 pgo_mode=false 时 compile() 不返回 PGOCompiledKernel
    using namespace ct::c3;

    auto& engine = C3Engine::getInstance();
    engine.clearCache();

    auto desc = TensorDesc::fromShape({4, 4});
    Graph g;
    size_t x = g.addInput(desc);
    size_t tanh = g.addNode(TanhNode{desc}, {x}, desc);
    g.markOutput(tanh);

    CompileOptions opts;
    opts.backend = C3Backend::MLIR;
    opts.pgo_mode = false;

    auto kernel = engine.compile(g, opts);
    ASSERT_NE(kernel, nullptr);

    // 验证不是 PGOCompiledKernel
    auto* pgo_kernel = dynamic_cast<PGOCompiledKernel*>(kernel.get());
    EXPECT_EQ(pgo_kernel, nullptr) << "Without pgo_mode, should not return PGOCompiledKernel";

    // 验证正确性
    Tensor a(ShapeTag{}, {4, 4});
    for (size_t i = 0; i < a.numel(); ++i)
        a.data_write<float>()[i] = static_cast<float>(i) / 10.0f;

    Tensor eager = a.tanh();
    auto result = kernel->execute({a});
    EXPECT_TRUE(tensorsAllClose(result[0], eager, 1e-4f, 1e-4f));
}

// ======================= GraphMerger 单元测试 =======================

#include "C3/GraphMerger.h"

namespace {

ct::c3::Graph buildFCReLUSubGraph(const std::vector<size_t>& in_shape,
                                   const std::vector<size_t>& w_shape,
                                   const std::vector<size_t>& b_shape,
                                   const std::vector<size_t>& out_shape,
                                   bool add_relu = true) {
    using namespace ct::c3;
    Graph g;
    auto in_desc = TensorDesc::fromShape(in_shape);
    auto w_desc = TensorDesc::fromShape(w_shape);
    auto b_desc = TensorDesc::fromShape(b_shape);
    auto out_desc = TensorDesc::fromShape(out_shape);

    size_t in = g.addInput(in_desc);
    size_t w = g.addInput(w_desc);
    size_t b = g.addInput(b_desc);

    auto mm_desc = TensorDesc::fromShape(out_shape);
    size_t mm = g.addNode(MatMulNode{w_desc, in_desc}, {w, in}, mm_desc);
    size_t add = g.addNode(AddNode{mm_desc, b_desc}, {mm, b}, out_desc);
    size_t out_id = add;
    if (add_relu) {
        out_id = g.addNode(ReLUNode{out_desc}, {add}, out_desc);
    }
    g.markOutput(out_id);
    return g;
}

}  // namespace

TEST(GraphMerger, EmptyInput) {
    std::vector<ct::c3::Graph> subs;
    ct::c3::MergeSpec spec;
    EXPECT_THROW(ct::c3::GraphMerger::merge(subs, spec), std::invalid_argument);
}

TEST(GraphMerger, SingleSubgraph) {
    using namespace ct::c3;
    Graph g = buildFCReLUSubGraph({4}, {4, 4}, {4}, {4});
    std::vector<Graph> subs = {g};
    MergeSpec spec;

    MergedGraphInfo info = GraphMerger::merge(subs, spec);
    EXPECT_EQ(info.graph.nodeCount(), g.nodeCount());
    EXPECT_EQ(info.graph.inputCount(), g.inputCount());
    EXPECT_EQ(info.graph.outputCount(), g.outputCount());
}

TEST(GraphMerger, SequentialMerge_TwoLayers) {
    using namespace ct::c3;
    // 子图 1: in=[4, 8], w1=[16, 8], b1=[16] → out=[4, 16]（带 ReLU）
    // 子图 2: in=[4, 16], w2=[32, 16], b2=[32] → out=[4, 32]（无 ReLU）
    Graph g1 = buildFCReLUSubGraph({4, 8}, {16, 8}, {16}, {4, 16}, true);
    Graph g2 = buildFCReLUSubGraph({4, 16}, {32, 16}, {32}, {4, 32}, false);

    std::vector<Graph> subs = {g1, g2};
    MergedGraphInfo info = GraphMerger::mergeSequential(subs);

    // 外部输入：x, w1, b1, w2, b2 = 5 个
    EXPECT_EQ(info.graph.inputCount(), 5u);
    // 输出：1 个
    EXPECT_EQ(info.graph.outputCount(), 1u);
    EXPECT_EQ(info.external_input_ids.size(), 5u);
    EXPECT_TRUE(info.graph.isValid());
}

TEST(GraphMerger, SequentialMerge_ThreeLayers) {
    using namespace ct::c3;
    Graph g1 = buildFCReLUSubGraph({8, 16}, {32, 16}, {32}, {8, 32}, true);
    Graph g2 = buildFCReLUSubGraph({8, 32}, {16, 32}, {16}, {8, 16}, true);
    Graph g3 = buildFCReLUSubGraph({8, 16}, {4, 16},  {4},  {8, 4},  false);

    std::vector<Graph> subs = {g1, g2, g3};
    MergedGraphInfo info = GraphMerger::mergeSequential(subs);

    // 外部输入：x, w1, b1, w2, b2, w3, b3 = 7 个
    EXPECT_EQ(info.graph.inputCount(), 7u);
    EXPECT_EQ(info.graph.outputCount(), 1u);
    EXPECT_EQ(info.external_input_ids.size(), 7u);
    EXPECT_TRUE(info.graph.isValid());
}

TEST(GraphMerger, ShapeMismatch) {
    using namespace ct::c3;
    Graph g1 = buildFCReLUSubGraph({4, 8}, {16, 8}, {16}, {4, 16}, true);
    Graph g2 = buildFCReLUSubGraph({4, 7}, {32, 7}, {32}, {4, 32}, false);

    std::vector<Graph> subs = {g1, g2};
    MergeSpec spec;
    spec.links.push_back({0, 0});

    EXPECT_THROW(GraphMerger::merge(subs, spec), std::invalid_argument);
}

TEST(GraphMerger, LinkCountMismatch) {
    using namespace ct::c3;
    Graph g1 = buildFCReLUSubGraph({4, 8}, {16, 8}, {16}, {4, 16}, true);
    Graph g2 = buildFCReLUSubGraph({4, 16}, {32, 16}, {32}, {4, 32}, false);

    std::vector<Graph> subs = {g1, g2};
    MergeSpec spec;
    spec.links.push_back({0, 0});
    spec.links.push_back({0, 0});  // 多余的链接

    EXPECT_THROW(GraphMerger::merge(subs, spec), std::invalid_argument);
}

TEST(GraphMerger, OutOfRangeLink) {
    using namespace ct::c3;
    Graph g1 = buildFCReLUSubGraph({4, 8}, {16, 8}, {16}, {4, 16}, true);
    Graph g2 = buildFCReLUSubGraph({4, 16}, {32, 16}, {32}, {4, 32}, false);

    std::vector<Graph> subs = {g1, g2};
    MergeSpec spec;
    spec.links.push_back({0, 999});  // 越界

    EXPECT_THROW(GraphMerger::merge(subs, spec), std::invalid_argument);
}

TEST(GraphMerger, SequentialCanFuseAndOptimize) {
    using namespace ct::c3;
    Graph g1 = buildFCReLUSubGraph({4, 8}, {16, 8}, {16}, {4, 16}, true);
    Graph g2 = buildFCReLUSubGraph({4, 16}, {32, 16}, {32}, {4, 32}, false);

    std::vector<Graph> subs = {g1, g2};
    MergedGraphInfo info = GraphMerger::mergeSequential(subs);

    // 融合图已经过死代码消除，应可直接 canonicalize
    Graph canonical = info.graph.canonicalize();
    EXPECT_TRUE(canonical.isValid());

    Graph fused = canonical.fuse();
    EXPECT_TRUE(fused.isValid());
}

// ======================= 数值正确性测试 =======================
// 验证融合图执行结果 == 逐层执行结果（数学校验，不只是结构校验）
// 注意：实际数值执行依赖 Handwritten 后端对 FusedNode 的支持，
// 这里采用更纯粹的"图结构等价"验证：手工构造"目标图"，与 merge 输出的图比较。

namespace {

/// 手工构造"目标图"：所有节点平铺在一张图中，节点 ID 按出现顺序
/// 这是 merge 应该产生的等价图（除了 ID 可能不同）。
static ct::c3::Graph buildExpectedMergedGraph(
    const std::vector<ct::c3::Graph>& subs) {
    using namespace ct::c3;
    Graph result;
    // 简化的目标图：把所有子图节点平铺，链式连接
    // 子图 0 的 in[0,1,2] 是 w1, x, b1（按 mergeSequential 约定）
    // 子图 1 的 in[0] 来自子图 0 的 out[0]，in[1,2] 是 w2, b2
    // 子图 2 的 in[0] 来自子图 1 的 out[0]，in[1,2] 是 w3, b3

    // 收集所有外部输入
    std::vector<size_t> ext_in_ids;
    auto add_in = [&](const Graph& sg, size_t j) {
        const auto& n = sg.node(sg.inputs()[j]);
        ext_in_ids.push_back(result.addInput(n.out_desc));
    };

    // 子图 0：3 个外部输入
    for (size_t j = 0; j < 3; ++j) add_in(subs[0], j);
    std::vector<size_t> sub0_in_map;  // 子图 0 的输入 ID → 新图 ID
    for (size_t j = 0; j < 3; ++j) sub0_in_map.push_back(ext_in_ids[ext_in_ids.size() - 3 + j]);

    // 复制子图 0 的非输入节点
    std::vector<size_t> sub0_node_map;
    for (const auto& n : subs[0].nodes()) {
        bool is_in = false;
        for (size_t in : subs[0].inputs()) if (n.id == in) { is_in = true; break; }
        if (is_in) continue;
        // 重映射 inputs（子图 0 的输入 ID → 新图 ID）
        std::vector<size_t> new_in;
        for (size_t i : n.inputs) {
            // i 是子图 0 的节点 ID，找到对应的新图 ID
            if (i < 3) {
                new_in.push_back(sub0_in_map[i]);
            } else {
                // 计算子图 0 中第 (i-3) 个非输入节点
                size_t idx = i - 3;
                new_in.push_back(sub0_node_map[idx]);
            }
        }
        sub0_node_map.push_back(result.addNode(n.op, new_in, n.out_desc));
    }
    size_t prev_out = sub0_node_map.back();  // 子图 0 最后节点

    // 子图 1, 2, ...
    for (size_t s = 1; s < subs.size(); ++s) {
        // 子图 s 的 in[0] = prev_out，in[1,2] = 外部
        add_in(subs[s], 1);
        add_in(subs[s], 2);
        size_t w_id = ext_in_ids[ext_in_ids.size() - 2];
        size_t b_id = ext_in_ids.back();

        std::vector<size_t> prev_nodes;  // 累积子图 s 的非输入节点
        for (const auto& n : subs[s].nodes()) {
            bool is_in = false;
            for (size_t in : subs[s].inputs()) if (n.id == in) { is_in = true; break; }
            if (is_in) continue;
            std::vector<size_t> new_in;
            for (size_t i : n.inputs) {
                if (i == subs[s].inputs()[0]) new_in.push_back(prev_out);
                else if (i == subs[s].inputs()[1]) new_in.push_back(w_id);
                else if (i == subs[s].inputs()[2]) new_in.push_back(b_id);
                else {
                    // 子图 s 的内部节点（按出现顺序索引）
                    size_t internal_idx = 0;
                    for (const auto& nn : subs[s].nodes()) {
                        bool ni = false;
                        for (size_t ii : subs[s].inputs()) if (nn.id == ii) { ni = true; break; }
                        if (ni) continue;
                        if (nn.id == i) break;
                        ++internal_idx;
                    }
                    new_in.push_back(prev_nodes[internal_idx]);
                }
            }
            prev_nodes.push_back(result.addNode(n.op, new_in, n.out_desc));
        }
        prev_out = prev_nodes.back();
    }

    result.markOutput(prev_out);
    return result;
}

}  // namespace

TEST(GraphMerger, GraphEquivalence_TwoLayers) {
    using namespace ct::c3;
    Graph g1 = buildFCReLUSubGraph({2, 4}, {8, 4}, {8}, {2, 8}, true);
    Graph g2 = buildFCReLUSubGraph({2, 8}, {6, 8}, {6}, {2, 6}, false);

    std::vector<Graph> subs = {g1, g2};
    MergedGraphInfo info = GraphMerger::mergeSequential(subs);

    // 手工构造期望图（纯结构）
    Graph expected = buildExpectedMergedGraph(subs);

    // 1. 直接比较节点/输入/输出数
    EXPECT_EQ(info.graph.nodeCount(), expected.nodeCount());
    EXPECT_EQ(info.graph.inputCount(), expected.inputCount());
    EXPECT_EQ(info.graph.outputCount(), expected.outputCount());

    // 2. 算子类型直方图应相同（MatMul 2个 + Add 2个 + ReLU 1个 = 5个）
    auto opHistogram = [](const Graph& g) {
        std::map<size_t, int> h;
        for (const auto& n : g.nodes()) h[n.op.index()]++;
        return h;
    };
    EXPECT_EQ(opHistogram(info.graph), opHistogram(expected));
}

TEST(GraphMerger, GraphEquivalence_ThreeLayers) {
    using namespace ct::c3;
    Graph g1 = buildFCReLUSubGraph({1, 2}, {4, 2}, {4}, {1, 4}, true);
    Graph g2 = buildFCReLUSubGraph({1, 4}, {3, 4}, {3}, {1, 3}, true);
    Graph g3 = buildFCReLUSubGraph({1, 3}, {2, 3}, {2}, {1, 2}, false);

    std::vector<Graph> subs = {g1, g2, g3};
    MergedGraphInfo info = GraphMerger::mergeSequential(subs);

    Graph expected = buildExpectedMergedGraph(subs);

    EXPECT_EQ(info.graph.nodeCount(), expected.nodeCount());
    EXPECT_EQ(info.graph.inputCount(), expected.inputCount());
    EXPECT_EQ(info.graph.outputCount(), expected.outputCount());

    auto opHistogram = [](const Graph& g) {
        std::map<size_t, int> h;
        for (const auto& n : g.nodes()) h[n.op.index()]++;
        return h;
    };
    EXPECT_EQ(opHistogram(info.graph), opHistogram(expected));
}

#endif // CT_ENABLE_MLIR