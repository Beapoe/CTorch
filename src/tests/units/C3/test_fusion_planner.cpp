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

#include <algorithm>
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

// ======================= SiLU 逐元素归类 =======================
// [SiLU 一等节点] Graph 层新增 SiLUNode 后，判据必须把它与 ReLU/Sigmoid/Tanh
// 同等对待(逐元素族)，否则 FFN 前向的 silu 会被判 LEAF、MatMul 尾链无法成
// GEMM_EPILOGUE（这正是 AGENTS 待办 #1 要补的缺口）。

TEST(FusionPlanner, SiLUIsElementwise) {
    Graph g;
    auto d6 = TensorDesc::fromShape({6});
    size_t x = g.addInput(d6);
    size_t silu = g.addNode(SiLUNode{d6}, {x}, d6);
    g.markOutput(silu);

    FusionPlan plan = FusionPlanner::planUnits(g);
    const FusionUnit* u = unitContaining(plan, silu);
    ASSERT_NE(u, nullptr);
    EXPECT_TRUE(u->isCompute());
    EXPECT_EQ(u->kind, FusionUnitKind::ELEMENTWISE);
    EXPECT_EQ(plan.compute_unit_count, 1u);
}

// MatMul + SiLU 单消费者尾链 → GEMM_EPILOGUE（LLaMA FFN 的 x@W_gate 后接 silu）
TEST(FusionPlanner, GemmSiLUEpilogueSingleUnit) {
    Graph g;
    auto xd = TensorDesc::fromShape({2, 3});
    auto wd = TensorDesc::fromShape({3, 4});
    auto md = TensorDesc::fromShape({2, 4});
    size_t x = g.addInput(xd);
    size_t w = g.addInput(wd);
    size_t mm = g.addNode(MatMulNode{xd, wd}, {x, w}, md);
    size_t silu = g.addNode(SiLUNode{md}, {mm}, md);
    g.markOutput(silu);

    FusionPlan plan = FusionPlanner::planUnits(g);
    const FusionUnit* umm = unitContaining(plan, mm);
    ASSERT_NE(umm, nullptr);
    EXPECT_EQ(umm->kind, FusionUnitKind::GEMM_EPILOGUE);
    EXPECT_EQ(plan.unitOf(silu), umm);
    EXPECT_EQ(umm->node_ids.size(), 2u);
    EXPECT_EQ(umm->numel, nodeNumelOf({2, 4}));
}

// SiLU 与其它逐元素算子共处一个 ELEMENTWISE 单元(Mul 门控: SwiGLU 的 g*u 形态)
TEST(FusionPlanner, SiLUChainsWithElementwise) {
    Graph g;
    auto d6 = TensorDesc::fromShape({6});
    size_t x = g.addInput(d6);
    size_t u = g.addInput(d6);
    size_t gate = g.addNode(SiLUNode{d6}, {x}, d6);
    size_t h = g.addNode(MulNode{d6, d6}, {gate, u}, d6);   // SwiGLU 门控
    g.markOutput(h);

    FusionPlan plan = FusionPlanner::planUnits(g);
    const FusionUnit* ug = unitContaining(plan, gate);
    ASSERT_NE(ug, nullptr);
    EXPECT_EQ(ug->kind, FusionUnitKind::ELEMENTWISE);
    EXPECT_EQ(plan.unitOf(h), ug);
    EXPECT_EQ(ug->node_ids.size(), 2u);
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

// ======================= 强制合并（跳过代价门）=======================
// [2026-09-10] 解耦"划分是否正确"(结构等价性) 与"划分是否划算"(收益模型):
// 强制模式下只要结构可并(多分量 + 共享外部输入)就并, 不做收益判定。

TEST(FusionPlanner, ForceMergeBypassesCostGate) {
    SharedExternalGraph sg = buildTwoChainsSharedExternal(1000);
    RegionFusionPolicy policy;
    policy.min_benefit_ratio = 1.0;   // 默认口径下明确不划算 → 不并
    policy.launch_unit_bytes = 0;

    // 默认: 代价门拦住 → 2 个 region
    FusionPlan strict = FusionPlanner::planUnits(sg.g, FusionStrategy::RegionKernel, policy);
    EXPECT_FALSE(strict.region_metric.merged);
    EXPECT_EQ(strict.compute_unit_count, 2u);

    // 强制: 跳过收益判定 → 并成 1 个 region(结构可并前提成立)
    policy.force_merge = true;
    FusionPlan forced = FusionPlanner::planUnits(sg.g, FusionStrategy::RegionKernel, policy);
    EXPECT_TRUE(forced.region_metric.merged);
    EXPECT_EQ(forced.compute_unit_count, 1u);
    const FusionUnit* u = forced.unitOf(sg.p);
    ASSERT_NE(u, nullptr);
    EXPECT_EQ(u->kind, FusionUnitKind::REGION_KERNEL);
    EXPECT_EQ(u->node_ids.size(), 3u);   // {p, e1, r}
    // 强制模式仍填充度量供观测(不做判定依据)
    EXPECT_EQ(forced.region_metric.component_count, 2u);
    EXPECT_EQ(forced.region_metric.saved_reload_bytes, 1000ull);
}

// 强制合并不应越过"结构不可并": 无共享外部输入的分量仍须割开
TEST(FusionPlanner, ForceMergeStillRequiresSharedExternal) {
    Graph g;
    auto d = TensorDesc::fromShape({2, 3});
    size_t x = g.addInput(d);
    size_t y = g.addInput(d);
    // 链1: x -> Neg -> ReLU; 链2: y -> Neg -> Tanh（各自独立, 无共享外部输入）
    size_t p = g.addNode(NegNode{d}, {x}, d);
    size_t e1 = g.addNode(ReLUNode{d}, {p}, d);
    size_t r = g.addNode(NegNode{d}, {y}, d);
    size_t e2 = g.addNode(TanhNode{d}, {r}, d);
    g.markOutput(e1);
    g.markOutput(e2);

    RegionFusionPolicy policy;
    policy.force_merge = true;   // 即使强制, 无共享外部输入也无"并"的意义
    FusionPlan region = FusionPlanner::planUnits(g, FusionStrategy::RegionKernel, policy);
    EXPECT_FALSE(region.region_metric.merged);
    EXPECT_EQ(region.compute_unit_count, 2u);
    const FusionUnit* up = region.unitOf(p);
    const FusionUnit* ur = region.unitOf(r);
    ASSERT_NE(up, nullptr);
    ASSERT_NE(ur, nullptr);
    EXPECT_NE(up, ur);
}

// ======================= ADR-0002 方案 C：跨分量默认合并 =======================

// Allow 策略：即使相对收益门槛不达标，也默认跨分量合并（判别力已下沉到规模保护）
TEST(FusionPlanner, AllowStrategyMergesByDefault) {
    SharedExternalGraph sg = buildTwoChainsSharedExternal(1000);
    RegionFusionPolicy policy;
    policy.min_benefit_ratio = 1.0;   // Strict 口径下明确不划算
    policy.launch_unit_bytes = 0;
    policy.merge_strategy = RegionMergeStrategy::Allow;
    policy.max_region_nodes = 64;     // 宽松规模上限

    FusionPlan region = FusionPlanner::planUnits(sg.g, FusionStrategy::RegionKernel, policy);
    EXPECT_TRUE(region.region_metric.merged);
    EXPECT_EQ(region.compute_unit_count, 1u);
    EXPECT_EQ(region.region_metric.region_node_count, 3u);
}

// Allow 策略：规模保护上限生效时仍不合并（方案 C 保留的唯一判别力）
TEST(FusionPlanner, AllowStrategyRespectsScaleGuard) {
    SharedExternalGraph sg = buildTwoChainsSharedExternal(1000);
    RegionFusionPolicy policy;
    policy.merge_strategy = RegionMergeStrategy::Allow;
    policy.max_region_nodes = 2;      // 3 个 regionable 节点 > 2 → 触发规模保护

    FusionPlan region = FusionPlanner::planUnits(sg.g, FusionStrategy::RegionKernel, policy);
    EXPECT_FALSE(region.region_metric.merged);
    EXPECT_EQ(region.compute_unit_count, 2u);
}

// 默认策略必须是 Strict：行为与 ADR-0002 之前完全一致（回归保护）
TEST(FusionPlanner, DefaultStrategyIsStrictUnchanged) {
    SharedExternalGraph sg = buildTwoChainsSharedExternal(1000);
    RegionFusionPolicy policy;        // 默认构造
    EXPECT_EQ(policy.merge_strategy, RegionMergeStrategy::Strict);
    policy.min_benefit_ratio = 1.0;   // 收益不足
    policy.launch_unit_bytes = 0;

    FusionPlan region = FusionPlanner::planUnits(sg.g, FusionStrategy::RegionKernel, policy);
    EXPECT_FALSE(region.region_metric.merged);   // Strict 下不合并(既有行为)
    EXPECT_EQ(region.compute_unit_count, 2u);
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

// ======================= 图切分 (G3 集成点基础) =======================
// partitionGraph: 按 FusionPlan 的 compute units 切出可独立编译的子图。
// 语义保证: 覆盖全部 compute 节点(无重无漏) + 子图拓扑有效 + 输入输出映射可回填。

TEST(FusionPlanner, PartitionCoversAllComputeNodesNoDupNoLoss) {
    SharedExternalGraph sg = buildTwoChainsSharedExternal(1000);
    FusionPlan plan = FusionPlanner::planUnits(sg.g, FusionStrategy::RegionKernel);
    std::vector<PartitionedSubGraph> subs = partitionGraph(sg.g, plan);

    // 子图数 == compute unit 数
    ASSERT_EQ(subs.size(), plan.compute_unit_count);

    // 覆盖性: 拼接全部子图节点 == 全部 compute 节点, 且无重复
    std::vector<size_t> all;
    for (const auto& s : subs)
        all.insert(all.end(), s.unit_node_ids.begin(), s.unit_node_ids.end());
    std::sort(all.begin(), all.end());

    size_t expected = 0;
    for (const auto& u : plan.units)
        if (u.isCompute()) expected += u.node_ids.size();

    EXPECT_EQ(all.size(), expected);                       // 无漏
    EXPECT_EQ(std::unique(all.begin(), all.end()), all.end());  // 无重
}

TEST(FusionPlanner, PartitionSubGraphsAreValidWithBoundaryMapping) {
    SharedExternalGraph sg = buildTwoChainsSharedExternal(1000);
    // 显式构造"不跨分量合并"的场景(默认 policy 的 launch 项会使其合并成 1 个 unit)
    RegionFusionPolicy strict;
    strict.min_benefit_ratio = 1.0;
    strict.launch_unit_bytes = 0;
    FusionPlan plan = FusionPlanner::planUnits(sg.g, FusionStrategy::RegionKernel, strict);

    std::vector<PartitionedSubGraph> subs = partitionGraph(sg.g, plan);
    ASSERT_EQ(subs.size(), 2u);   // 两个未合并的分量 → 2 个子图
    for (const auto& s : subs) {
        EXPECT_TRUE(s.graph.isValid());          // 拓扑有效 → 可独立编译
        EXPECT_GT(s.graph.outputCount(), 0u);    // 有输出
        EXPECT_EQ(s.graph.inputCount(), s.input_orig_ids.size());
        EXPECT_EQ(s.graph.outputCount(), s.output_orig_ids.size());
        // 映射表能覆盖本子图全部节点
        EXPECT_EQ(s.orig_to_sub.size(), s.unit_node_ids.size());
    }
}

// 两个分量仅共享外部输入、无中间量依赖 → 子图之间无 upstream 依赖
TEST(FusionPlanner, PartitionSharedExternalInputHasNoUpstreamDep) {
    SharedExternalGraph sg = buildTwoChainsSharedExternal(1000);
    RegionFusionPolicy strict;
    strict.min_benefit_ratio = 1.0;
    strict.launch_unit_bytes = 0;
    FusionPlan plan = FusionPlanner::planUnits(sg.g, FusionStrategy::RegionKernel, strict);

    std::vector<PartitionedSubGraph> subs = partitionGraph(sg.g, plan);
    ASSERT_EQ(subs.size(), 2u);
    for (const auto& s : subs) EXPECT_TRUE(s.upstream_units.empty());
}

// 合并成单 region 时只切出 1 个子图(与 MIMO 单内核一致)
TEST(FusionPlanner, PartitionMergedPlanYieldsSingleSubGraph) {
    SharedExternalGraph sg = buildTwoChainsSharedExternal(1000);
    RegionFusionPolicy policy;
    policy.merge_strategy = RegionMergeStrategy::Allow;   // ADR-0002 方案 C: 默认合并
    policy.max_region_nodes = 64;
    FusionPlan plan = FusionPlanner::planUnits(sg.g, FusionStrategy::RegionKernel, policy);
    ASSERT_TRUE(plan.region_metric.merged);

    std::vector<PartitionedSubGraph> subs = partitionGraph(sg.g, plan);
    EXPECT_EQ(subs.size(), 1u);
    EXPECT_EQ(subs[0].unit_node_ids.size(), 3u);   // {p, e1, r}
}

// ======================= 子图边界契约 (G3 编排前提) =======================
// 以下两条刻画 partitionGraph 的边界语义, 是执行/编排层的硬约束:
//   契约 1: Const 节点属 LEAF, 在子图中被暴露为"外部输入" → 驱动方必须喂其真实常量值。
//   契约 2: region 分隔符(如 SumReduce)不进入任何 compute unit → 其产出不被子图覆盖。
// 二者都会影响"子图执行计划是否完整", 故显式固化以防回归。

// 契约 1: Const 节点作为子图外部输入出现, 且 input_orig_ids 能回指该 Const 节点。
TEST(FusionPlanner, PartitionExposesConstAsExternalInputToDriver) {
    Graph g;
    auto d = TensorDesc::fromShape({4});
    size_t x = g.addInput(d);
    TensorDesc one_desc = TensorDesc::fromShape({1});
    size_t one = g.addConstant(1.0, one_desc);   // 图内真实常量(非输入占位)
    size_t sum = g.addNode(AddNode{d, one_desc}, {x, one}, d);
    g.markOutput(sum);

    // 强制不跨分量合并, 确保产生子图而非忽略
    RegionFusionPolicy strict;
    strict.min_benefit_ratio = 1.0;
    strict.launch_unit_bytes = 0;
    FusionPlan plan = FusionPlanner::planUnits(g, FusionStrategy::RegionKernel, strict);
    std::vector<PartitionedSubGraph> subs = partitionGraph(g, plan);
    ASSERT_FALSE(subs.empty());

    // Const 必须出现在某个子图的 input_orig_ids 中(而不是被静默丢弃)
    bool found = false;
    for (const auto& s : subs)
        for (size_t oid : s.input_orig_ids)
            if (oid == one) found = true;
    EXPECT_TRUE(found);
}

// 契约 2: SumReduce 是分隔符, 不进入任何子图; 其产出因此不被子图执行计划覆盖。
TEST(FusionPlanner, PartitionDoesNotCoverRegionSeparatorOutput) {
    Graph g;
    auto dg = TensorDesc::fromShape({2, 4});
    auto dr = TensorDesc::fromShape({4});
    size_t grad = g.addInput(dg);
    // SumReduce(axis=0): [2,4] -> [4], 属 region 分隔符
    size_t reduced = g.addNode(SumReduceNode{dg, 0}, {grad}, dr);
    g.markOutput(reduced);

    FusionPlan plan = FusionPlanner::planUnits(g, FusionStrategy::RegionKernel);
    std::vector<PartitionedSubGraph> subs = partitionGraph(g, plan);

    // 分隔符节点不得出现在任何子图的覆盖集合内
    for (const auto& s : subs)
        for (size_t nid : s.unit_node_ids) EXPECT_NE(nid, reduced);

    // 该输出也不应由任何子图声称(驱动方需另行处理 → 见 runPartitionABTest 的 missing 计数)
    for (const auto& s : subs)
        for (size_t oid : s.output_orig_ids) EXPECT_NE(oid, reduced);
}
