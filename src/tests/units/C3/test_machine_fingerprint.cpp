/**
 * @file test_machine_fingerprint.cpp
 * @brief MachineFingerprint deploy 校准持久化 + O(1) 运行时读取 + 代价门桥接测试
 * @date 2026-09-07
 */

#include <gtest/gtest.h>

#include <cstdio>
#include <string>
#include <unistd.h>

#include "C3/FusionPlanner.h"
#include "C3/MachineFingerprint.h"

using namespace ct::c3;

namespace {
std::string tmpPath() { return std::string("c3_fingerprint_test_") + std::to_string(::getpid()) + ".tmp"; }
}

TEST(MachineFingerprint, RoundtripSaveLoadO1Read) {
    const std::string path = tmpPath();
    FingerprintData d;
    d.machine_label = "test-machine";
    d.calibrated_at = "2026-09-07T00:00";
    d.method_version = "v1";
    d.bandwidth_gbps = 123.5;
    d.launch_us = 3.0;
    d.launch_unit_bytes = (uint64_t)(3.0 * 123.5 * 1000.0);

    ASSERT_TRUE(MachineFingerprint::save(path, d));

    // 加载到同一单例(用显式 load)
    ASSERT_TRUE(MachineFingerprint::instance().load(path));
    EXPECT_TRUE(MachineFingerprint::instance().loaded());
    // O(1) getter 反映文件值
    EXPECT_EQ(MachineFingerprint::instance().launchUnitBytes(), d.launch_unit_bytes);
    EXPECT_DOUBLE_EQ(MachineFingerprint::instance().launchUs(), 3.0);
    EXPECT_DOUBLE_EQ(MachineFingerprint::instance().bandwidthGbps(), 123.5);
    EXPECT_EQ(MachineFingerprint::instance().data().machine_label, "test-machine");

    std::remove(path.c_str());
}

TEST(MachineFingerprint, CostGateBridgeUsesFingerprint) {
    const std::string path = tmpPath();
    FingerprintData d;
    d.machine_label = "x";
    d.bandwidth_gbps = 300.0;
    d.launch_us = 4.0;
    d.launch_unit_bytes = (uint64_t)(4.0 * 300.0 * 1000.0); // 1.2MB
    ASSERT_TRUE(MachineFingerprint::save(path, d));
    ASSERT_TRUE(MachineFingerprint::instance().load(path));

    // fromMachineDefaults 应取指纹实测 launch 项, 而非默认 400KB
    RegionFusionPolicy p = RegionFusionPolicy::fromMachineDefaults();
    EXPECT_EQ(p.launch_unit_bytes, d.launch_unit_bytes);
    EXPECT_NE(p.launch_unit_bytes, 400u * 1024u);

    std::remove(path.c_str());
}

TEST(MachineFingerprint, MissingFileFallsBackConservative) {
    // 不存在路径 → load 返回 false, 不抛异常
    EXPECT_FALSE(MachineFingerprint::instance().load("/nonexistent/c3.fingerprint"));
}
