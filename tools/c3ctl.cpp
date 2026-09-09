/**
 * @file c3ctl.cpp
 * @brief C3 部署校准 CLI: 首次部署时跑机器探针, 写指纹供运行时 O(1) 读取
 * @date 2026-09-07 (docs/C3_DEPLOY_AUTOTUNE_DESIGN.md 落地骨架)
 *
 * 用法:
 *   c3ctl calibrate [--out <path>] [--label <machine>]   # 测带宽+launch 税, 写指纹
 *   c3ctl show     [--in <path>]                          # 用运行时 O(1) 读取路径打印
 * 默认路径: c3.fingerprint (env C3_FINGERPRINT 可覆盖, 与运行时一致)
 */

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#include "C3/C3Engine.h"
#include "C3/MachineFingerprint.h"
#include "Tensor.h"

using namespace ct;
using namespace ct::c3;

namespace {

std::string argValue(int argc, char** argv, const char* name, const std::string& dflt) {
    for (int i = 1; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], name) == 0) return argv[i + 1];
    return dflt;
}
bool hasFlag(int argc, char** argv, const char* name) {
    for (int i = 1; i < argc; ++i) if (std::strcmp(argv[i], name) == 0) return true;
    return false;
}

std::string nowIso() {
    // 简易 ISO 时间(精度到秒足够用于指纹标注)
    std::time_t t = std::time(nullptr);
    std::tm tm{};
    localtime_r(&t, &tm);
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%04d-%02d-%02dT%02d:%02d:%02d",
                  tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
                  tm.tm_hour, tm.tm_min, tm.tm_sec);
    return std::string(buf);
}

/// 有效内存带宽 GB/s(memcpy 读+写, 不可被编译器消除; 多轮合并计时)
double probeBandwidthGbps() {
    const size_t n = 32u << 20;                 // 32M floats ≈ 128MB
    const size_t bytes = n * sizeof(float);
    std::vector<float> src(n, 1.0f);
    std::vector<float> dst(n, 0.0f);
    // 预热 + 让页落位(分配初始化已做), 再正式多轮合并计时
    volatile size_t sink = 0;
    std::memcpy(dst.data(), src.data(), bytes);
    const int R = 8;
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < R; ++r) {
        std::memcpy(dst.data(), src.data(), bytes);
        sink += (size_t)dst[0]; // 防消除
    }
    auto t1 = std::chrono::steady_clock::now();
    double dt = std::chrono::duration<double>(t1 - t0).count();
    (void)sink;
    if (dt <= 0) return 200.0; // 防护
    double gbps = (double)((uint64_t)R * bytes) / 1e9 / dt;
    if (gbps <= 0 || gbps > 2000.0) gbps = 200.0; // 粗上限防护
    return gbps;
}

/// C3 单次 launch 税 us: 编译一个极小逐元素 kernel 反复 execute 取最优均值
double probeLaunchUs() {
    try {
        Graph g;
        auto d = TensorDesc::fromShape({1024});
        size_t a = g.addInput(d);
        size_t b = g.addInput(d);
        size_t c = g.addNode(AddNode{d, d}, {a, b}, d);
        g.markOutput(c);

        CompileOptions opts;
        opts.backend = C3Backend::MLIR;
        opts.target_device = DeviceType::kCPU;
        opts.enable_cache = false;
        opts.enable_fusion = false;
        opts.enable_autotune = false;

        auto kernel = C3Engine::getInstance().compile(g, opts);
        if (!kernel) { std::fprintf(stderr, "[c3ctl] C3 compile failed, launch probe fallback\n"); return 2.0; }

        auto make = [](float v) { Tensor t(ShapeTag{}, std::vector<size_t>{1024}); auto* p=t.data_write<float>(); for(int i=0;i<1024;i++)p[i]=v; return t; };
        std::vector<Tensor> in = {make(1.0f), make(2.0f)};
        double best = 1e9;
        for (int round = 0; round < 5; ++round) {
            for (int i = 0; i < 20; ++i) kernel->execute(in); // warm
            auto s = std::chrono::steady_clock::now();
            const int R = 200;
            for (int i = 0; i < R; ++i) kernel->execute(in);
            auto e = std::chrono::steady_clock::now();
            double us = std::chrono::duration<double, std::micro>(e - s).count() / R;
            if (us < best) best = us;
        }
        return best;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "[c3ctl] launch probe exception: %s (fallback 2.0us)\n", ex.what());
        return 2.0;
    }
}

int calibrateCmd(int argc, char** argv) {
    std::string out = argValue(argc, argv, "--out", MachineFingerprint::kDefaultPath);
    std::string label = argValue(argc, argv, "--label", "unknown-machine");

    std::fprintf(stderr, "[c3ctl] calibrate: probing...\n");
    double bw = probeBandwidthGbps();
    double launch = probeLaunchUs();
    uint64_t lub = (uint64_t)(launch * bw * 1000.0); // launch_us * bytes/us = bytes
    if (lub < 1) lub = 1;

    FingerprintData d;
    d.machine_label = label;
    d.calibrated_at = nowIso();
    d.method_version = "c3ctl-v1";
    d.bandwidth_gbps = bw;
    d.launch_us = launch;
    d.launch_unit_bytes = lub;

    if (!MachineFingerprint::save(out, d)) {
        std::fprintf(stderr, "[c3ctl] FAILED to write %s\n", out.c_str());
        return 1;
    }
    std::printf("bandwidth_gbps=%.1f\n", bw);
    std::printf("launch_us=%.3f\n", launch);
    std::printf("launch_unit_bytes=%llu\n", (unsigned long long)lub);
    std::printf("written=%s\n", out.c_str());
    return 0;
}

int showCmd(int argc, char** argv) {
    std::string in = argValue(argc, argv, "--in", MachineFingerprint::kDefaultPath);
    if (!MachineFingerprint::instance().load(in)) {
        std::fprintf(stderr, "[c3ctl] cannot load %s (not calibrated yet?)\n", in.c_str());
        return 1;
    }
    // 模拟运行时 O(1) 读取(getter 走已加载内存)
    std::printf("machine_label=%s\n", MachineFingerprint::instance().data().machine_label.c_str());
    std::printf("calibrated_at=%s\n", MachineFingerprint::instance().data().calibrated_at.c_str());
    std::printf("method_version=%s\n", MachineFingerprint::instance().data().method_version.c_str());
    std::printf("bandwidth_gbps=%.1f\n", MachineFingerprint::instance().bandwidthGbps());
    std::printf("launch_us=%.3f\n", MachineFingerprint::instance().launchUs());
    std::printf("launch_unit_bytes=%llu\n", (unsigned long long)MachineFingerprint::instance().launchUnitBytes());
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: c3ctl calibrate [--out p] [--label m] | c3ctl show [--in p]\n");
        return 1;
    }
    std::string cmd = argv[1];
    if (cmd == "calibrate") return calibrateCmd(argc, argv);
    if (cmd == "show") return showCmd(argc, argv);
    std::fprintf(stderr, "unknown subcommand: %s\n", cmd.c_str());
    return 1;
}
