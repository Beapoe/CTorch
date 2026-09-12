/**
 * @file bench_guard.h
 * @brief 基准测试环境守卫 (STATUS §4.97 待办 ⑤)
 * @details 解决「测量环境不可控 → 小效应量结论不可信」问题(§4.90 教训):
 *          每次 bench 计时时自动记录环境快照(load average / 机器 CPU 占用),
 *          收尾时输出同配置重复测量的统计(中位数/均值/CV 离散度/最低值),
 *          并在离散度超过阈值时自动标注「本次结果不可用于定案」。
 *
 *          设计原则:
 *          - 干扰只会让耗时变长、不会变短 ⇒ 最低值(低分位)比中位数更接近真实能力;
 *          - 离散度 CV = 标准差/均值, 是测量可信度的第一判据(§4.90: CV>=效应量则不可定案);
 *          - 环境快照按 label 分组记录, 报告输出可直接粘贴进性能结论文档。
 *
 *          用法(header-only):
 *            #include "bench_guard.h"
 *            BenchGuard guard;
 *            for (rounds) { auto t0=now(); ...work...; guard.record("C3", elapsed_ms); }
 *            guard.report();
 *
 * @date 2026-09-12
 */

#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <map>
#include <string>
#include <vector>

#if defined(__APPLE__)
#include <mach/mach_host.h>
#include <mach/mach.h>
#endif
#include <cstdlib>

namespace benchguard {

struct EnvSnapshot {
    double load1 = 0, load5 = 0, load15 = 0;
    /// 机器级 CPU 占用百分比(0-100); -1 表示无法获取
    double cpu_percent = -1;
};

/// 采集当前环境: load average + (macOS) 机器级 CPU 占用
inline EnvSnapshot snapshot() {
    EnvSnapshot e;
    double loads[3] = {0, 0, 0};
#if defined(__APPLE__)
    getloadavg(loads, 3);
    host_cpu_load_info_data_t cinfo;
    mach_msg_type_number_t count = HOST_CPU_LOAD_INFO_COUNT;
    if (host_statistics(mach_host_self(), HOST_CPU_LOAD_INFO,
                        reinterpret_cast<host_info_t>(&cinfo), &count) == KERN_SUCCESS) {
        uint64_t total = 0;
        for (int i = 0; i < CPU_STATE_MAX; ++i) total += cinfo.cpu_ticks[i];
        if (total > 0) {
            uint64_t busy = total - cinfo.cpu_ticks[CPU_STATE_IDLE];
            e.cpu_percent = 100.0 * static_cast<double>(busy) / static_cast<double>(total);
        }
    }
#else
    // Linux: /proc/loadavg 前三项
    FILE* f = std::fopen("/proc/loadavg", "r");
    if (f) {
        (void)std::fscanf(f, "%lf %lf %lf", &e.load1, &e.load5, &e.load15);
        std::fclose(f);
        return e;
    }
#endif
    e.load1 = loads[0]; e.load5 = loads[1]; e.load15 = loads[2];
    return e;
}

struct BenchGuard {
    struct Run {
        double elapsed_ms = 0;
        EnvSnapshot env;
    };

    /// label → 该组所有测量
    std::map<std::string, std::vector<Run>> runs;

    /// 记录一次测量(内部自动采集环境快照)
    void record(const std::string& label, double elapsed_ms) {
        runs[label].push_back(Run{elapsed_ms, snapshot()});
    }

    /// 组统计
    struct GroupStats {
        size_t n = 0;
        double min = 0, median = 0, mean = 0, cv = 0;  // cv = 变异系数(%)
        double env_load_avg = 0, env_cpu_avg = 0;
        bool reliable = true;  // false ⇒ 离散度过大, 不可定案
    };

    GroupStats stats(const std::string& label, double cv_threshold = 15.0) const {
        GroupStats s;
        auto it = runs.find(label);
        if (it == runs.end()) return s;
        const auto& v = it->second;
        s.n = v.size();
        if (s.n == 0) return s;

        std::vector<double> t(v.size());
        double load_sum = 0, cpu_sum = 0, cpu_cnt = 0;
        for (size_t i = 0; i < v.size(); ++i) {
            t[i] = v[i].elapsed_ms;
            load_sum += v[i].env.load1;
            if (v[i].env.cpu_percent >= 0) { cpu_sum += v[i].env.cpu_percent; cpu_cnt++; }
        }
        std::sort(t.begin(), t.end());
        s.min = t.front();
        s.median = (s.n % 2) ? t[s.n / 2] : (t[s.n / 2 - 1] + t[s.n / 2]) / 2.0;
        double sum = 0;
        for (double x : t) sum += x;
        s.mean = sum / static_cast<double>(s.n);
        double var = 0;
        for (double x : t) var += (x - s.mean) * (x - s.mean);
        double sd = std::sqrt(var / static_cast<double>(s.n));
        s.cv = (s.mean > 0) ? 100.0 * sd / s.mean : 0;
        s.env_load_avg = load_sum / static_cast<double>(s.n);
        s.env_cpu_avg = (cpu_cnt > 0) ? cpu_sum / cpu_cnt : -1;
        // 判定: CV 超阈值 ⇒ 噪声大于效应量量级, 不可定案(§4.90 纪律)
        s.reliable = (s.cv <= cv_threshold);
        return s;
    }

    /// 输出报告(可直接粘贴进性能结论文档)
    void report(FILE* out = stderr, double cv_threshold = 15.0) const {
        std::fprintf(out, "\n[BENCH-GUARD] ============ 测量可信度报告 ============\n");
        EnvSnapshot now = snapshot();
        std::fprintf(out, "[BENCH-GUARD] 收尾环境: load1=%.2f 机器CPU=%.1f%%\n",
                     now.load1, now.cpu_percent);
        for (const auto& kv : runs) {
            const GroupStats s = stats(kv.first, cv_threshold);
            if (s.n == 0) continue;
            std::fprintf(out,
                "[BENCH-GUARD] %-12s n=%zu min=%.2fms median=%.2fms mean=%.2fms "
                "CV=%.1f%% env(load1=%.2f CPU=%.1f%%) %s\n",
                kv.first.c_str(), s.n, s.min, s.median, s.mean, s.cv,
                s.env_load_avg, s.env_cpu_avg,
                s.reliable ? "[可定案]" : "[不可定案: 离散度过大, 仅供趋势]");
        }
        std::fprintf(out, "[BENCH-GUARD] 判定规则: CV>%.0f%% 或环境负载过高时, "
                         "最小值为最接近真实能力的口径(干扰只会使耗时变长)\n",
                     cv_threshold);
        std::fprintf(out, "[BENCH-GUARD] ================================================\n");
    }
};

} // namespace benchguard
