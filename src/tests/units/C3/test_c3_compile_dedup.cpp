/**
 * @file test_c3_compile_dedup.cpp
 * @brief 同步 compile() 路径的 in-flight 去重回归测试
 * @date 2026-09-10 (scope: STATUS_CONTEXT §4.91 B4)
 * @details 背景：审查发现同步 compile() 的锁域为「锁内查 cache → 锁外 doCompile →
 *          锁内写 cache」，故并发同 key 同时 miss 时各线程会重复编译（结果无害但
 *          白烧 CPU）；而异步路径有 state.pending 去重，二者语义不一致。
 *
 *          修复后引入 compiling_keys（key → 发起线程 id）+ condition_variable：
 *          - 不同线程同 key：后来者等待，唤醒后重查缓存复用他人结果；
 *          - 同线程重入同 key：不等待直接编译（避免自死锁）；
 *          - enable_cache=false：不做去重（结果不共享，等待纯属白费）。
 *
 *          本文件验证上述三条语义，并覆盖编译失败后不得残留 in-flight 标记（RAII）。
 */

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

#include "C3/C3Engine.h"
#include "C3/Graph.h"

using namespace ct::c3;

namespace {

/// 构造一个结构确定的 MatMul 图（各线程独立构造 → 内容相同 → cache key 相同）
Graph makeMatMulGraph(size_t M, size_t K, size_t N) {
    Graph g;
    auto a_desc = TensorDesc::fromShape({M, K});
    auto b_desc = TensorDesc::fromShape({K, N});
    auto c_desc = TensorDesc::fromShape({M, N});
    size_t a = g.addInput(a_desc);
    size_t b = g.addInput(b_desc);
    size_t c = g.addNode(MatMulNode{a_desc, b_desc}, {a, b}, c_desc);
    g.markOutput(c);
    return g;
}

/// 自旋栅栏：让所有线程尽可能同时进入 compile()，制造真实的并发重叠
class SpinBarrier {
public:
    explicit SpinBarrier(int n) : target_(n) {}
    void arriveAndWait() {
        ready_.fetch_add(1);
        while (ready_.load(std::memory_order_acquire) < target_) {
            std::this_thread::yield();
        }
    }
private:
    int target_;
    std::atomic<int> ready_{0};
};

} // namespace

// ==================== 并发同 key：只编译一次 ====================

TEST(CompileDedup, ConcurrentSameKeyCompilesOnce) {
    auto& engine = C3Engine::getInstance();
    engine.clearCache();

    CompileOptions opts;
    opts.enable_cache = true;

    const auto before = engine.getCacheStats();

    constexpr int kThreads = 8;
    SpinBarrier barrier(kThreads);
    std::atomic<int> succeeded{0};

    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int i = 0; i < kThreads; ++i) {
        threads.emplace_back([&]() {
            // 各线程独立构造结构相同的图 → 同 cache key，且无共享读竞争
            Graph g = makeMatMulGraph(256, 256, 256);
            barrier.arriveAndWait();
            auto kernel = engine.compile(g, opts);
            if (kernel) succeeded.fetch_add(1);
        });
    }
    for (auto& t : threads) t.join();

    const auto after = engine.getCacheStats();

    EXPECT_EQ(succeeded.load(), kThreads) << "所有线程都应拿到可用内核";

    // 核心断言：同一 key 的并发同步编译只真正执行一次
    EXPECT_EQ(after.sync_compiles - before.sync_compiles, 1u)
        << "并发同 key 应只编译一次 (修复前此处会 >1)";

    // 其余线程应通过「等待前者完成 + 缓存复用」获得结果
    EXPECT_GT(after.dedup_waits, before.dedup_waits)
        << "应有线程走入等待路径 (若为 0 说明本次未产生并发重叠)";
}

// ==================== enable_cache=false：不做无谓等待 ====================

TEST(CompileDedup, CacheDisabledSkipsWaitPath) {
    auto& engine = C3Engine::getInstance();
    engine.clearCache();

    CompileOptions opts;
    opts.enable_cache = false;  // 结果不写缓存 → 去重无意义

    const auto before = engine.getCacheStats();

    constexpr int kThreads = 4;
    SpinBarrier barrier(kThreads);
    std::atomic<int> finished{0};

    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int i = 0; i < kThreads; ++i) {
        threads.emplace_back([&]() {
            Graph g = makeMatMulGraph(64, 64, 64);
            barrier.arriveAndWait();
            (void)engine.compile(g, opts);
            finished.fetch_add(1);
        });
    }
    for (auto& t : threads) t.join();

    const auto after = engine.getCacheStats();

    EXPECT_EQ(finished.load(), kThreads);
    // 缓存关闭时不应进入等待路径（等待者唤醒后仍须自行编译，纯属白费）
    EXPECT_EQ(after.dedup_waits, before.dedup_waits);
    // 每个线程都应自行完成一次编译
    EXPECT_EQ(after.sync_compiles - before.sync_compiles,
              static_cast<size_t>(kThreads));
}

// ==================== 同线程重复编译：命中缓存且不自等待 ====================

TEST(CompileDedup, SameThreadRepeatedCompileHitsCache) {
    auto& engine = C3Engine::getInstance();
    engine.clearCache();

    CompileOptions opts;
    opts.enable_cache = true;

    Graph g = makeMatMulGraph(32, 32, 32);

    // 若 in-flight 标记未被 RAII 正确释放，第二次调用会自等待而死锁
    auto k1 = engine.compile(g, opts);
    auto k2 = engine.compile(g, opts);

    ASSERT_NE(k1, nullptr);
    ASSERT_NE(k2, nullptr);
    EXPECT_EQ(k1, k2) << "第二次应复用缓存中的同一内核对象";

    const auto st = engine.getCacheStats();
    EXPECT_EQ(st.sync_compiles, 1u) << "同 key 重复请求只应编译一次";
    EXPECT_EQ(st.hits, 1u) << "第二次应是缓存命中";
}

// ==================== 编译失败也必须释放 in-flight（RAII 防死锁） ====================

TEST(CompileDedup, FailureDoesNotPoisonInflight) {
    auto& engine = C3Engine::getInstance();
    engine.clearCache();

    CompileOptions opts;
    opts.enable_cache = true;

    // 维度不匹配的 MatMul：编译应失败（抛异常或返回空），但不得残留 in-flight 标记
    {
        Graph bad;
        auto a_desc = TensorDesc::fromShape({4, 8});
        auto b_desc = TensorDesc::fromShape({4, 8});  // K 不匹配 (8 vs 4)
        auto c_desc = TensorDesc::fromShape({4, 8});
        size_t a = bad.addInput(a_desc);
        size_t b = bad.addInput(b_desc);
        size_t c = bad.addNode(MatMulNode{a_desc, b_desc}, {a, b}, c_desc);
        bad.markOutput(c);
        try {
            (void)engine.compile(bad, opts);
        } catch (...) {
            // 允许失败：本测试只关心失败后 in-flight 是否被清理
        }
    }

    // 带超时保护：若残留标记导致阻塞，测试失败而不是挂死整个测试套件
    std::atomic<bool> done{false};
    std::thread t([&]() {
        Graph good = makeMatMulGraph(16, 16, 16);
        (void)engine.compile(good, opts);
        done.store(true);
    });
    for (int i = 0; i < 400 && !done.load(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    const bool finished = done.load();
    if (finished) {
        t.join();
    } else {
        // 不 join，避免测试进程挂死；失败信息已足以定位问题
        t.detach();
    }
    EXPECT_TRUE(finished) << "编译失败后残留 in-flight 标记 → 后续 compile 被永久阻塞";
}
