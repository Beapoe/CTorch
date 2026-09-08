/**
 * @file test_fusion_planner.cpp
 * @brief FusionPlanner 通用融合决策层单元测试
 * @date 2026-09-07 (scope: docs/C3_UNIVERSAL_FUSION_DESIGN.md)
 * @details 验证判据数据驱动、非结构特判：
 *   - 纯逐元素链 → 单个 ELEMENTWISE 单元
 *   - MatMul + 单消费者逐元素尾链 → GEMM_EPILOGUE
 *   - 多消费者扇出 → 必须物化割开
 *   - numel 不一致 → 割开（identity-1D ABI 门，与 LinalgOneShot 收紧同判据）
 *   - 两个 GEMM 共享加法 → 不并（共享 GEMM 合并负收益，已证）
 */

#include <gtest/gtest.h>

#include <vector>

#include "C3/FusionPlanner.h"
#include "C3/Graph.h"

using namespace ct::c3;

namespace {

const FusionUnit* unitContaining(const FusionPlan& plan, size_t node_id) {
    return plan.unitOf(node_id);
}

size_t nodeNumelOf(const std::vector<size_t>& shape) {
    size_t v = 1;
    for (size_t d : shape) v *= d;
    return v;
}

} // namespace

// ======================= 纯逐元素链 → 单 ELEMENTWISE =======================

TEST(FusionPlanner, ElementwiseChainSingleUnit) {
    Graph g;
    auto d6 = TensorDesc::fromShape({6});
    size_t x = g.addInput(d6);
    size_t y = g.addInput(d6);
    size_t z = g.addInput(d6);
    size_t mul = g.addNode(MulNode{d6, d6}, {x, y}, d6);
    size_t add = g.addNode(AddNode{d6, d6}, {mul, z}, d6);
    size_t relu = g.addNode(ReLUNode{d6}, {add}, d6);
    size_t neg = g.addNode(NegNode{d6}, {relu}, d6);
    g.markOutput(neg);

    FusionPlan plan = FusionPlanner::planUnits(g);

    // mul/add/relu/neg 共处一个 compute 单元（判据自动，非按名字）
    const FusionUnit* umul = unitContaining(plan, mul);
    ASSERT_NE(umul, nullptr);
    EXPECT_TRUE(umul->isCompute());
    EXPECT_EQ(umul->kind, FusionUnitKind::ELEMENTWISE);
    for (size_t id : {add, relu, neg}) {
        EXPECT_EQ(plan.unitOf(id), umul);
    }
    EXPECT_EQ(umul->node_ids.size(), 4u);
    // 3 个叶输入应为该单元的外部输入
    ASSERT_EQ(umul->external_input_ids.size(), 3u);
    EXPECT_EQ(umul->numel, nodeNumelOf({6}));
}

// ======================= MatMul + 单消费者 epilogue → GEMM_EPILOGUE =======================

TEST(FusionPlanner, GemmEpilogueSingleUnit) {
    Graph g;
    auto xd = TensorDesc::fromShape({2, 3});
    auto wd = TensorDesc::fromShape({3, 4});
    auto bd = TensorDesc::fromShape({4});
    auto md = TensorDesc::fromShape({2, 4});
    size_t x = g.addInput(xd);
    size_t w = g.addInput(wd);
    size_t b = g.addInput(bd);
    size_t mm = g.addNode(MatMulNode{xd, wd}, {x, w}, md);
    size_t add = g.addNode(AddNode{md, bd}, {mm, b}, md);
    size_t relu = g.addNode(ReLUNode{md}, {add}, md);
    g.markOutput(relu);

    FusionPlan plan = FusionPlanner::planUnits(g);

    const FusionUnit* umm = unitContaining(plan, mm);
    ASSERT_NE(umm, nullptr);
    EXPECT_EQ(umm->kind, FusionUnitKind::GEMM_EPILOGUE);
    EXPECT_EQ(plan.unitOf(add), umm);
    EXPECT_EQ(plan.unitOf(relu), umm);
    // GEMM + 2 个 epilogue 节点
    EXPECT_EQ(umm->node_ids.size(), 3u);
    // 外部输入 = x, w, b（3 个叶）
    ASSERT_EQ(umm->external_input_ids.size(), 3u);
    // GEMM 输出与 epilogue 等 numel
    EXPECT_EQ(umm->numel, nodeNumelOf({2, 4}));
}

// ======================= 多消费者扇出 → 物化割开 =======================

TEST(FusionPlanner, FanoutMultiConsumerCut) {
    Graph g;
    auto d6 = TensorDesc::fromShape({6});
    size_t x = g.addInput(d6);
    size_t p = g.addInput(d6);
    size_t q = g.addInput(d6);
    size_t t = g.addNode(ReLUNode{d6}, {x}, d6);          // 被 y,z 两个消费者共享
    size_t y = g.addNode(AddNode{d6, d6}, {t, p}, d6);
    size_t z = g.addNode(MulNode{d6, d6}, {t, q}, d6);
    g.markOutput(y);
    g.markOutput(z);

    FusionPlan plan = FusionPlanner::planUnits(g);

    // t 多消费者 → 不能并入任一消费者 → 自成一个 ELEMENTWISE 单节点单元
    const FusionUnit* ut = unitContaining(plan, t);
    ASSERT_NE(ut, nullptr);
    EXPECT_EQ(ut->kind, FusionUnitKind::ELEMENTWISE);
    EXPECT_EQ(ut->node_ids.size(), 1u);
    // y 与 z 各自独立（不共享上游，也不被并入 t）
    EXPECT_NE(plan.unitOf(y), ut);
    EXPECT_NE(plan.unitOf(z), ut);
    EXPECT_NE(plan.unitOf(y), plan.unitOf(z));
    // 3 个 compute 单元：{t},{y},{z}
    EXPECT_EQ(plan.compute_unit_count, 3u);
}

// ======================= numel 不一致 → 割开（identity-1D ABI 门） =======================

TEST(FusionPlanner, NumelMismatchCut) {
    Graph g;
    auto d6 = TensorDesc::fromShape({6});
    auto d12 = TensorDesc::fromShape({12});
    size_t x = g.addInput(d6);
    size_t z = g.addInput(d12);
    // ReLU 输出 6，Add 声称输出 12（广播/不匹配）——判据必须拒绝并入
    size_t p = g.addNode(ReLUNode{d6}, {x}, d6);
    size_t add = g.addNode(AddNode{d6, d12}, {p, z}, d12);
    g.markOutput(add);

    FusionPlan plan = FusionPlanner::planUnits(g);

    const FusionUnit* up = unitContaining(plan, p);
    const FusionUnit* uadd = unitContaining(plan, add);
    ASSERT_NE(up, nullptr);
    ASSERT_NE(uadd, nullptr);
    EXPECT_NE(up, uadd); // numel 不等 → 割开，不做 identity-1D 融合
    EXPECT_EQ(up->node_ids.size(), 1u);
    EXPECT_EQ(uadd->node_ids.size(), 1u);
    EXPECT_EQ(plan.compute_unit_count, 2u);
}

// ======================= 双 GEMM 共享 Add → 不并 =======================

TEST(FusionPlanner, SharedGemmNotMerged) {
    Graph g;
    auto xd = TensorDesc::fromShape({2, 3});
    auto wd = TensorDesc::fromShape({3, 4});
    auto md = TensorDesc::fromShape({2, 4});
    size_t x = g.addInput(xd);
    size_t w1 = g.addInput(wd);
    size_t w2 = g.addInput(wd);
    size_t mm1 = g.addNode(MatMulNode{xd, wd}, {x, w1}, md);
    size_t mm2 = g.addNode(MatMulNode{xd, wd}, {x, w2}, md);
    size_t add = g.addNode(AddNode{md, md}, {mm1, mm2}, md);
    size_t relu = g.addNode(ReLUNode{md}, {add}, md);
    g.markOutput(relu);

    FusionPlan plan = FusionPlanner::planUnits(g);

    const FusionUnit* umm1 = unitContaining(plan, mm1);
    const FusionUnit* umm2 = unitContaining(plan, mm2);
    ASSERT_NE(umm1, nullptr);
    ASSERT_NE(umm2, nullptr);
    // 两个 GEMM 不得并入同一单元（共享 GEMM 合并负收益，已证）
    EXPECT_NE(umm1, umm2);
    // mm1 吸收单消费者尾链（add, relu），形成 GEMM_EPILOGUE
    EXPECT_EQ(umm1->kind, FusionUnitKind::GEMM_EPILOGUE);
    EXPECT_EQ(plan.unitOf(add), umm1);
    EXPECT_EQ(plan.unitOf(relu), umm1);
    EXPECT_EQ(umm1->node_ids.size(), 3u);
    // mm2 单独成 GEMM 单元
    EXPECT_EQ(umm2->kind, FusionUnitKind::GEMM);
    EXPECT_EQ(umm2->node_ids.size(), 1u);
    // mm1 单元的 epilogue(Add) 以 mm2 的输出为外部输入（物化跨单元）
    bool has_mm2_ext = false;
    for (size_t e : umm1->external_input_ids) if (e == mm2) has_mm2_ext = true;
    EXPECT_TRUE(has_mm2_ext);
}
