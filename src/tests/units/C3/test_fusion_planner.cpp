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

// ======================= RegionKernel: 共享中间量 → 单 region =======================
// 复现 MIMO 语义: 一个中间量被多个 GEMM 消费(多输出), region 策略应并成单内核,
// 而默认(多消费者必物化/双 GEMM 不并)会切成多个单元。

TEST(FusionPlanner, RegionKernelSharedGemmSingleRegion) {
    Graph g;
    auto xd = TensorDesc::fromShape({2, 3});
    auto wd = TensorDesc::fromShape({3, 4});
    auto md = TensorDesc::fromShape({2, 4});
    size_t x = g.addInput(xd);
    size_t w1 = g.addInput(wd);
    size_t w2 = g.addInput(wd);
    size_t mid = g.addNode(ReLUNode{xd}, {x}, xd);       // 共享中间量, 被两个 GEMM 消费
    size_t outA = g.addNode(MatMulNode{xd, wd}, {mid, w1}, md);
    size_t outB = g.addNode(MatMulNode{xd, wd}, {mid, w2}, md);
    g.markOutput(outA);
    g.markOutput(outB);

    FusionPlan def = FusionPlanner::planUnits(g);                       // Default
    FusionPlan region = FusionPlanner::planUnits(g, FusionStrategy::RegionKernel);

    // 默认: 多消费者 → mid 独立物化 + 两个 GEMM 各自单元
    EXPECT_EQ(def.compute_unit_count, 3u);

    // region: 连通区段(mid+两个 GEMM)并成 1 个多输出内核
    EXPECT_EQ(region.compute_unit_count, 1u);
    const FusionUnit* umid = region.unitOf(mid);
    ASSERT_NE(umid, nullptr);
    EXPECT_EQ(umid->kind, FusionUnitKind::REGION_KERNEL);
    // mid 与其两个 GEMM 消费者在同一 region(共享中间量内联)
    EXPECT_EQ(region.unitOf(outA), umid);
    EXPECT_EQ(region.unitOf(outB), umid);
    EXPECT_EQ(umid->node_ids.size(), 3u); // {mid, outA, outB}
}

// ======================= RegionKernel: Transpose 并入 GEMM =======================

TEST(FusionPlanner, RegionKernelIncludesTranspose) {
    Graph g;
    auto xd = TensorDesc::fromShape({2, 3});
    auto wd = TensorDesc::fromShape({3, 4});
    auto md = TensorDesc::fromShape({2, 4});
    auto mtd = TensorDesc::fromShape({4, 2});
    size_t x = g.addInput(xd);
    size_t w = g.addInput(wd);
    size_t mm = g.addNode(MatMulNode{xd, wd}, {x, w}, md);
    size_t mmT = g.addNode(TransposeNode{md, 0, 1}, {mm}, mtd);   // 转置
    size_t relu = g.addNode(ReLUNode{mtd}, {mmT}, mtd);
    g.markOutput(relu);

    FusionPlan region = FusionPlanner::planUnits(g, FusionStrategy::RegionKernel);
    // 默认: mm(GEMM) 与 relu(ELEM) 被 mmT(默认 LEAF 边界)隔开
    FusionPlan def = FusionPlanner::planUnits(g);
    EXPECT_EQ(def.compute_unit_count, 2u); // mm, relu(各自, 中间夹 transpose 不并入)

    // region: transpose 折叠进 GEMM, mm/mmT/relu 连通 → 1 region
    EXPECT_EQ(region.compute_unit_count, 1u);
    const FusionUnit* ummT = region.unitOf(mmT);
    ASSERT_NE(ummT, nullptr);
    EXPECT_EQ(ummT->kind, FusionUnitKind::REGION_KERNEL);
    EXPECT_EQ(region.unitOf(mm), ummT);
    EXPECT_EQ(region.unitOf(relu), ummT);
}

// ======================= RegionKernel: 不相连 → 多个 region(非整图瞎并) =======================

TEST(FusionPlanner, RegionKernelDisjointTwoRegions) {
    Graph g;
    auto d = TensorDesc::fromShape({2, 3});
    size_t x = g.addInput(d);
    size_t y = g.addInput(d);
    // 链1: x -> Neg -> ReLU
    size_t p = g.addNode(NegNode{d}, {x}, d);
    size_t e1 = g.addNode(ReLUNode{d}, {p}, d);
    // 链2: y -> Neg -> Tanh (与链1 无共享 compute)
    size_t r = g.addNode(NegNode{d}, {y}, d);
    size_t e2 = g.addNode(TanhNode{d}, {r}, d);
    g.markOutput(e1);
    g.markOutput(e2);

    FusionPlan region = FusionPlanner::planUnits(g, FusionStrategy::RegionKernel);
    // 两个不相连连通分量 → 2 个 region, 不把整图并成 1 个
    EXPECT_EQ(region.compute_unit_count, 2u);
    const FusionUnit* up = region.unitOf(p);
    const FusionUnit* ur = region.unitOf(r);
    ASSERT_NE(up, nullptr);
    ASSERT_NE(ur, nullptr);
    EXPECT_EQ(up->kind, FusionUnitKind::REGION_KERNEL);
    EXPECT_NE(up, ur);
    EXPECT_EQ(up->node_ids.size(), 2u);   // {p, e1}
    EXPECT_EQ(ur->node_ids.size(), 2u);   // {r, e2}
}

// ======================= RegionKernel 跨分量合并代价门 =======================
// 两个分量只经共享外部输入间接相连: 重读节省大 → 并单 region; 否则保持多 region。

struct SharedExternalGraph {
    Graph g;
    size_t p; // 链1: Neg(x)
    size_t e1;
    size_t r; // 链2: Exp(x)
};

static SharedExternalGraph buildTwoChainsSharedExternal(size_t numel = 1000) {
    SharedExternalGraph out;
    auto d = TensorDesc::fromShape({numel});
    size_t x = out.g.addInput(d);
    // 链1: x -> Neg -> ReLU
    out.p = out.g.addNode(NegNode{d}, {x}, d);
    out.e1 = out.g.addNode(ReLUNode{d}, {out.p}, d);
    // 链2: x -> Exp (与链1 无共享 compute, 仅共享外部输入 x)
    out.r = out.g.addNode(ExpNode{d}, {x}, d);
    out.g.markOutput(out.e1);
    out.g.markOutput(out.r);
    return out;
}

TEST(FusionPlanner, RegionKernelCostMergeWorthy) {
    SharedExternalGraph sg = buildTwoChainsSharedExternal(1000);
    RegionFusionPolicy policy;
    policy.min_benefit_ratio = 0.1;   // 重读省 1000B > 10% live 工作集 1000B → 并
    policy.launch_unit_bytes = 0;     // 关 launch, 只看重读项
    FusionPlan region = FusionPlanner::planUnits(sg.g, FusionStrategy::RegionKernel, policy);
    EXPECT_EQ(region.compute_unit_count, 1u);
    EXPECT_TRUE(region.region_metric.merged);
    EXPECT_EQ(region.region_metric.component_count, 2u);
    EXPECT_EQ(region.region_metric.saved_reload_bytes, 1000ull);
    EXPECT_EQ(region.region_metric.saved_launch_bytes, 0ull);
    // live 中间量: 仅 p(Neg), e1/r 是 graph output 不计入
    EXPECT_EQ(region.region_metric.working_set_bytes, 1000ull);
    const FusionUnit* u = region.unitOf(sg.p);
    ASSERT_NE(u, nullptr);
    EXPECT_EQ(u->kind, FusionUnitKind::REGION_KERNEL);
    EXPECT_EQ(u->node_ids.size(), 3u); // {p, e1, r}
}

TEST(FusionPlanner, RegionKernelCostKeepsSeparate) {
    SharedExternalGraph sg = buildTwoChainsSharedExternal(1000);
    RegionFusionPolicy policy;
    policy.min_benefit_ratio = 1.0;   // 重读省 1000B 不敌 100% 工作集 1000B → 不并
    policy.launch_unit_bytes = 0;
    FusionPlan region = FusionPlanner::planUnits(sg.g, FusionStrategy::RegionKernel, policy);
    EXPECT_EQ(region.compute_unit_count, 2u);
    EXPECT_FALSE(region.region_metric.merged);
    const FusionUnit* up = region.unitOf(sg.p);
    const FusionUnit* ur = region.unitOf(sg.r);
    ASSERT_NE(up, nullptr);
    ASSERT_NE(ur, nullptr);
    EXPECT_NE(up, ur);
    EXPECT_EQ(up->node_ids.size(), 2u); // {p, e1}
    EXPECT_EQ(ur->node_ids.size(), 1u); // {r}
}

// launch 项独立生效: 共享外部输入但重读节省小, 靠省 launch 跨分量合并
TEST(FusionPlanner, RegionKernelLaunchMerges) {
    SharedExternalGraph sg = buildTwoChainsSharedExternal(6);
    RegionFusionPolicy policy;
    policy.min_benefit_ratio = 1.0;    // 单看重读: 6B 不敌 100% live(6B) → 不会并
    policy.launch_unit_bytes = 100;    // 但省 1 次 launch(100B) 补足 → 并
    FusionPlan region = FusionPlanner::planUnits(sg.g, FusionStrategy::RegionKernel, policy);
    EXPECT_EQ(region.compute_unit_count, 1u);
    EXPECT_TRUE(region.region_metric.merged);
    EXPECT_EQ(region.region_metric.saved_reload_bytes, 6ull);
    EXPECT_EQ(region.region_metric.saved_launch_bytes, 100ull); // (k-1)*launch = 1*100
    EXPECT_EQ(region.region_metric.working_set_bytes, 6ull);
}

// 峰值 live 工作集: 两个不重叠的中间量(求和=12)峰值只有 6。
// 证明 working_set 用峰值而非求和, 避免高估寄存器/缓冲压力。
TEST(FusionPlanner, RegionKernelPeakLiveWorkingSet) {
    Graph g;
    auto d = TensorDesc::fromShape({6});
    size_t x = g.addInput(d);
    // 链A: x -> Neg(n1) -> ReLU(out1);  n1 live 于 [n1, out1]
    size_t n1 = g.addNode(NegNode{d}, {x}, d);
    size_t out1 = g.addNode(ReLUNode{d}, {n1}, d);
    // 链B: x -> Exp(n2) -> Tanh(out2);  n2 live 于 [n2, out2] (与 n1 不重叠)
    size_t n2 = g.addNode(ExpNode{d}, {x}, d);
    size_t out2 = g.addNode(TanhNode{d}, {n2}, d);
    g.markOutput(out1);
    g.markOutput(out2);

    RegionFusionPolicy policy;
    policy.min_benefit_ratio = 100.0; // 任意, 只要不合并也能读 metric
    FusionPlan region = FusionPlanner::planUnits(g, FusionStrategy::RegionKernel, policy);
    // n1 与 n2 生命周期不重叠 → 峰值 6, 而非求和 12
    EXPECT_EQ(region.region_metric.working_set_bytes, 6ull);
}
