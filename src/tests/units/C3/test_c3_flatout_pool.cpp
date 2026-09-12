/**
 * @file test_c3_flatout_pool.cpp
 * @brief MIMO flat 输出缓冲池的 drain 清理回归测试
 * @date 2026-09-10 (scope: STATUS_CONTEXT §4.93 A1)
 * @details 背景：审查指出 FlatOutPool 用「heap 分配 + never delete」规避静态析构顺序问题，
 *          导致其 free_bufs 中已归还的 buffer 常驻到进程结束（进程级泄漏；实际为
 *          「涨到峰值后稳定、但永不释放」，而非无限增长）。
 *
 *          修复：新增 FlatOutPool::drain()，释放池中**已归还**的 buffer（入池即代表
 *          shared_ptr 引用已归零，故 free 不会 use-after-free），并接入
 *          ct::c3::shutdownAll()。池结构（mutex/map）本身仍不析构，因此清理后若仍有
 *          Tensor 析构，其 deleter 仍可安全访问池（走 draining 分支直接释放），
 *          不会重蹈历史上「静态析构顺序 → lock 已销毁 mutex」的崩溃。
 *
 *          本文件验证：池确实被使用 → drain 清空 → 可重复调用 → 池能自动恢复缓存
 *          且执行结果仍正确。
 */

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "Tensor.h"
#include "C3/Graph.h"
#include "C3/C3Engine.h"

using namespace ct::c3;

namespace {

std::shared_ptr<CompiledKernel> compileMLIRLocal(Graph& g) {
    CompileOptions opts;
    opts.backend = C3Backend::MLIR;
    opts.target_device = DeviceType::kCPU;
    return C3Engine::getInstance().compile(g, opts);
}

/// 构造并执行一个 multi-node 图（Transpose → SumReduce(axis=0)），触发 FlatOutPool。
/// 执行结束后返回的 Tensor 析构 → Storage 释放 → buffer 归还入池。
void runMultiNodeGraphAndCheck() {
    Graph g;
    auto in_desc = TensorDesc::fromShape({2, 3});
    size_t in = g.addInput(in_desc);
    auto tr_desc = TensorDesc::fromShape({3, 2});
    size_t tr = g.addNode(TransposeNode{in_desc, 0, 1}, {in}, tr_desc);
    auto sum_desc = TensorDesc::fromShape({2});
    size_t sum = g.addNode(SumReduceNode{tr_desc, 0}, {tr}, sum_desc);
    g.markOutput(sum);

    auto kernel = compileMLIRLocal(g);
    ASSERT_NE(kernel, nullptr) << "multi-node 图应编译成功";

    Tensor A(ShapeTag{}, {2, 3});
    for (size_t i = 0; i < A.numel(); ++i) A.data_write<float>()[i] = float(i + 1);

    auto results = kernel->execute({A});
    ASSERT_EQ(results.size(), 1u);

    // 数值正确性：[[1,2,3],[4,5,6]] → 转置 [[1,4],[2,5],[3,6]] → sum(axis=0) = [6,15]
    // （drain 只影响内存回收，绝不应改变数值）
    ASSERT_EQ(results[0].numel(), 2u);
    EXPECT_FLOAT_EQ(results[0].data_read<float>()[0], 6.0f);
    EXPECT_FLOAT_EQ(results[0].data_read<float>()[1], 15.0f);
}

/// 线程安全版本: 执行 multi-node 图并验证数值, 成功返回 true(ASSERT 不能用于子线程)
bool runMultiNodeGraphVerify() {
    Graph g;
    auto in_desc = TensorDesc::fromShape({2, 3});
    size_t in = g.addInput(in_desc);
    auto tr_desc = TensorDesc::fromShape({3, 2});
    size_t tr = g.addNode(TransposeNode{in_desc, 0, 1}, {in}, tr_desc);
    auto sum_desc = TensorDesc::fromShape({2});
    size_t sum = g.addNode(SumReduceNode{tr_desc, 0}, {tr}, sum_desc);
    g.markOutput(sum);

    auto kernel = compileMLIRLocal(g);
    if (!kernel) return false;
    Tensor A(ShapeTag{}, {2, 3});
    for (size_t i = 0; i < A.numel(); ++i) A.data_write<float>()[i] = float(i + 1);
    auto results = kernel->execute({A});
    if (results.size() != 1u || results[0].numel() != 2u) return false;
    return results[0].data_read<float>()[0] == 6.0f &&
           results[0].data_read<float>()[1] == 15.0f;
}

} // namespace

// ==================== 池确实被 MIMO 执行使用 ====================

TEST(FlatOutPool, MultiNodeExecutionPopulatesPool) {
    runMultiNodeGraphAndCheck();
    const auto st = C3Engine::getFlatOutPoolStats();
    EXPECT_GT(st.cached_buffers, 0u) << "MIMO 执行后 flat 输出 buffer 应归还入池";
    EXPECT_GT(st.cached_bytes, 0u);
}

// ==================== drain 清空池 ====================

TEST(FlatOutPool, DrainReleasesCachedBuffers) {
    runMultiNodeGraphAndCheck();
    ASSERT_GT(C3Engine::getFlatOutPoolStats().cached_buffers, 0u)
        << "前置条件：池中应先有已归还的 buffer";

    C3Engine::drainFlatOutPool();

    const auto after = C3Engine::getFlatOutPoolStats();
    EXPECT_EQ(after.cached_buffers, 0u) << "drain 应释放池中所有已归还 buffer";
    EXPECT_EQ(after.cached_bytes, 0u);
}

// ==================== drain 幂等 + 池可自动恢复 ====================

TEST(FlatOutPool, DrainIsIdempotentAndPoolRecovers) {
    runMultiNodeGraphAndCheck();

    C3Engine::drainFlatOutPool();
    C3Engine::drainFlatOutPool();  // 重复调用必须安全（空池 drain）
    EXPECT_EQ(C3Engine::getFlatOutPoolStats().cached_buffers, 0u);

    // drain 后池暂停缓存，但下一次 acquire() 应复位该状态使池恢复 ——
    // 否则 shutdown 语义被误用一次就会永久失去池化收益（性能退化）。
    runMultiNodeGraphAndCheck();
    EXPECT_GT(C3Engine::getFlatOutPoolStats().cached_buffers, 0u)
        << "acquire 应复位 draining, 池需恢复缓存且数值仍正确";
}

// ==================== 清理后仍有 Tensor 析构必须安全（历史崩溃点回归） ====================

TEST(FlatOutPool, DrainThenLateTensorDestructionIsSafe) {
    // 模拟「drain 之后仍有 Tensor 析构」的退出顺序：池结构不析构，deleter 应走
    // draining 分支直接释放，既不崩溃也不重新积累。
    C3Engine::drainFlatOutPool();  // 先清空，保证本用例起点状态确定

    std::shared_ptr<CompiledKernel> kernel;
    Tensor A(ShapeTag{}, {2, 3});
    for (size_t i = 0; i < A.numel(); ++i) A.data_write<float>()[i] = float(i + 1);
    std::vector<Tensor> results;

    {
        Graph g;
        auto in_desc = TensorDesc::fromShape({2, 3});
        size_t in = g.addInput(in_desc);
        auto tr_desc = TensorDesc::fromShape({3, 2});
        size_t tr = g.addNode(TransposeNode{in_desc, 0, 1}, {in}, tr_desc);
        auto sum_desc = TensorDesc::fromShape({2});
        size_t sum = g.addNode(SumReduceNode{tr_desc, 0}, {tr}, sum_desc);
        g.markOutput(sum);
        kernel = compileMLIRLocal(g);
        ASSERT_NE(kernel, nullptr);
        results = kernel->execute({A});   // 持有池 buffer（引用未归零 → 尚未入池）
    }

    C3Engine::drainFlatOutPool();         // 在用 buffer 不在池中，drain 不应也不能释放它

    // 数值仍必须可用（若 drain 误释放了在用 buffer，这里会读到脏数据或崩溃）
    ASSERT_EQ(results.size(), 1u);
    EXPECT_FLOAT_EQ(results[0].data_read<float>()[0], 6.0f);
    EXPECT_FLOAT_EQ(results[0].data_read<float>()[1], 15.0f);

    results.clear();                      // 迟到的析构：应走 draining 分支直接 free
    EXPECT_EQ(C3Engine::getFlatOutPoolStats().cached_buffers, 0u)
        << "drain 之后归还的 buffer 不应重新入池";
}

// ==================== 并发执行 + drain 交错：无 UAF / 无崩溃 ====================

TEST(FlatOutPool, ConcurrentExecuteAndDrainStress) {
    C3Engine::drainFlatOutPool();  // 干净起点

    constexpr int kThreads = 4;
    constexpr int kIters = 50;
    std::atomic<int> ok{0};

    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&]() {
            for (int i = 0; i < kIters; ++i) {
                if (runMultiNodeGraphVerify()) ok.fetch_add(1);
            }
        });
    }

    // 主线程与执行并发交错 drain: 验证 drain 与 acquire/release 的竞争安全
    for (int i = 0; i < 200; ++i) {
        C3Engine::drainFlatOutPool();
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    for (auto& th : threads) th.join();

    EXPECT_EQ(ok.load(), kThreads * kIters)
        << "并发执行 + 交错 drain 下全部结果必须正确(若出现 UAF 会读到脏数据或崩溃)";
}
