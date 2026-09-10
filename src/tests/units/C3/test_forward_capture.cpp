/**
 * @file test_forward_capture.cpp
 * @brief ForwardCapture 整图捕获单元测试
 * @date 2026-09-07 (通用图融合 forward 前置)
 * @details 用真实 eager 前向(requires_grad)构造 autograd DAG, 从根输出捕获成 c3::Graph,
 *          再交给 FusionPlanner 验证能自动判定 GEMM_EPILOGUE(证明捕获层产出可规划的真图)。
 */

#include <gtest/gtest.h>

#include <memory>

#include "C3/ForwardCapture.h"
#include "C3/FusionPlanner.h"
#include "Tensor.h"

using namespace ct;
using namespace ct::c3;

namespace {

Tensor makeLeaf(const std::vector<size_t>& shape) {
    Tensor t(ShapeTag{}, shape, DType::kFloat, DeviceType::kCPU);
    size_t n = 1; for (size_t d : shape) n *= d;
    float* q = t.data_write<float>();
    for (size_t i = 0; i < n; ++i) q[i] = 0.5f;   // 填满(值不影响结构断言)
    t.requires_grad(true);
    return t;
}

} // namespace

// MatMul + ReLU 真实前向 → 捕获 → planner 判定 GEMM_EPILOGUE
TEST(ForwardCapture, MatMulReluCaptureAndPlan) {
    Tensor A = makeLeaf({2, 3});
    Tensor B = makeLeaf({3, 4});

    Tensor C = A.matmul(B);   // [2,4]
    Tensor D = C.relu();      // 单消费者尾链
    (void)D.data_read<float>();

    ForwardCaptureResult res = ForwardCapture::capture(D);
    ASSERT_TRUE(res.ok) << res.error;
    EXPECT_EQ(res.graph.nodeCount(), 4u);          // A, B 叶 + mm + relu
    EXPECT_EQ(res.graph.inputCount(), 2u);          // 两个 requires_grad 叶
    EXPECT_EQ(res.graph.outputCount(), 1u);
    EXPECT_EQ(res.external_leaf_count, 2u);
    EXPECT_EQ(res.unsupported_node_count, 0u);

    // 捕获到的真图交给 planner: 应识别 mm + relu 为单个 GEMM_EPILOGUE
    FusionPlan plan = FusionPlanner::planUnits(res.graph);
    EXPECT_EQ(plan.compute_unit_count, 1u);
    // 找到根(relu)所在单元并核对
    const FusionUnit* u = plan.unitOf(res.root_c3_id);
    ASSERT_NE(u, nullptr);
    EXPECT_EQ(u->kind, FusionUnitKind::GEMM_EPILOGUE);
    EXPECT_EQ(u->node_ids.size(), 2u);
}

// 捕获根为叶(无上游计算) → 明确报错而非静默空图
TEST(ForwardCapture, RootLeafFailsCleanly) {
    Tensor x = makeLeaf({4});
    ForwardCaptureResult res = ForwardCapture::capture(x);
    EXPECT_FALSE(res.ok);
    EXPECT_FALSE(res.error.empty());
}

// MatMul + SiLU 真实前向 → 捕获 → planner 判定 GEMM_EPILOGUE
// [SiLU 一等节点] 此前 Graph 层无 SiLUNode，ForwardCapture 遇 autograd SiLUNode
// 会报 unsupported(整个 FFN 前向捕获失败); 补齐后应与 ReLU 路径同构。
TEST(ForwardCapture, MatMulSiLUCaptureAndPlan) {
    Tensor A = makeLeaf({2, 3});
    Tensor B = makeLeaf({3, 4});

    Tensor C = A.matmul(B);   // [2,4]
    Tensor D = C.silu();      // 单消费者尾链 (LLaMA FFN 风格)
    (void)D.data_read<float>();

    ForwardCaptureResult res = ForwardCapture::capture(D);
    ASSERT_TRUE(res.ok) << res.error;
    EXPECT_EQ(res.graph.nodeCount(), 4u);          // A, B 叶 + mm + silu
    EXPECT_EQ(res.graph.inputCount(), 2u);
    EXPECT_EQ(res.graph.outputCount(), 1u);
    EXPECT_EQ(res.external_leaf_count, 2u);
    EXPECT_EQ(res.unsupported_node_count, 0u);

    // 捕获到的真图交给 planner: 应识别 mm + silu 为单个 GEMM_EPILOGUE
    FusionPlan plan = FusionPlanner::planUnits(res.graph);
    EXPECT_EQ(plan.compute_unit_count, 1u);
    const FusionUnit* u = plan.unitOf(res.root_c3_id);
    ASSERT_NE(u, nullptr);
    EXPECT_EQ(u->kind, FusionUnitKind::GEMM_EPILOGUE);
    EXPECT_EQ(u->node_ids.size(), 2u);
}

// silu 单独作为根(无 MatMul) → 仍应被判 ELEMENTWISE 单元
TEST(ForwardCapture, SiluOnlyCapturesAsElementwise) {
    Tensor X = makeLeaf({2, 3});
    Tensor Y = X.silu();
    (void)Y.data_read<float>();

    ForwardCaptureResult res = ForwardCapture::capture(Y);
    ASSERT_TRUE(res.ok) << res.error;
    EXPECT_EQ(res.graph.nodeCount(), 2u);          // X 叶 + silu

    FusionPlan plan = FusionPlanner::planUnits(res.graph);
    const FusionUnit* u = plan.unitOf(res.root_c3_id);
    ASSERT_NE(u, nullptr);
    EXPECT_EQ(u->kind, FusionUnitKind::ELEMENTWISE);
}
