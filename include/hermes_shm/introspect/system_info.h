/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Distributed under BSD 3-Clause license.                                   *
 * Copyright by The HDF Group.                                               *
 * Copyright by the Illinois Institute of Technology.                        *
 * All rights reserved.                                                      *
 *                                                                           *
 * This file is part of Hermes. The full Hermes copyright notice, including  *
 * terms governing use, modification, and redistribution, is contained in    *
 * the COPYING file, which can be found at the top directory. If you do not  *
 * have access to the file, you may request a copy from help@hdfgroup.org.   *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef HERMES_SYSINFO_INFO_H_
#define HERMES_SYSINFO_INFO_H_

#include <unistd.h>
#include <sys/sysinfo.h>
#include "hermes_shm/util/singleton/_singleton.h"
#include "hermes_shm/util/formatter.h"
#include <iostream>
#include <fstream>

#define HERMES_SYSTEM_INFO \
  hshm::LockfreeSingleton<hshm::SystemInfo>::GetInstance()
#define HERMES_SYSTEM_INFO_T hshm::SystemInfo*

namespace hshm {

struct SystemInfo {
  int pid_;
  int ncpu_;
  int page_size_;
  int uid_;
  int gid_;
  size_t ram_size_;
#ifdef HSHM_IS_HOST
  std::vector<size_t> cur_cpu_freq_;
#endif

  HSHM_CROSS_FUN
  void RefreshInfo() {
#ifdef HSHM_IS_HOST
    pid_ = getpid();
    ncpu_ = get_nprocs_conf();
    page_size_ = getpagesize();
    struct sysinfo info;
    sysinfo(&info);
    uid_ = getuid();
    gid_ = getgid();
    ram_size_ = info.totalram;
    cur_cpu_freq_.resize(ncpu_);
    RefreshCpuFreqKhz();
#endif
  }

  void RefreshCpuFreqKhz() {
    #ifdef HSHM_IS_HOST
    for (int i = 0; i < ncpu_; ++i) {
      cur_cpu_freq_[i] = GetCpuFreqKhz(i);
    }
    #endif
  }

  size_t GetCpuFreqKhz(int cpu) {
    #ifdef HSHM_IS_HOST
    // Read /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_cur_freq
    std::string cpu_str = hshm::Formatter::format(
        "/sys/devices/system/cpu/cpu{}/cpufreq/cpuinfo_cur_freq",
        cpu);
    std::ifstream cpu_file(cpu_str);
    size_t freq_khz;
    cpu_file >> freq_khz;
    return freq_khz;
#else
    return 0;
    #endif
  }

  size_t GetCpuMaxFreqKhz(int cpu) {
    #ifdef HSHM_IS_HOST
    // Read /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_cur_freq
    std::string cpu_str = hshm::Formatter::format(
        "/sys/devices/system/cpu/cpu{}/cpufreq/cpuinfo_max_freq",
        cpu);
    std::ifstream cpu_file(cpu_str);
    size_t freq_khz;
    cpu_file >> freq_khz;
    return freq_khz;
#else
    return 0;
    #endif
  }

  size_t GetCpuMinFreqKhz(int cpu) {
    #ifdef HSHM_IS_HOST
    // Read /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_cur_freq
    std::string cpu_str = hshm::Formatter::format(
        "/sys/devices/system/cpu/cpu{}/cpufreq/cpuinfo_min_freq",
        cpu);
    std::ifstream cpu_file(cpu_str);
    size_t freq_khz;
    cpu_file >> freq_khz;
    return freq_khz;
#else
    return 0;
    #endif
  }

  size_t GetCpuMinFreqMhz(int cpu) {
    return GetCpuMinFreqKhz(cpu) / 1000;
  }

  size_t GetCpuMaxFreqMhz(int cpu) {
    return GetCpuMaxFreqKhz(cpu) / 1000;
  }

  void SetCpuFreqMhz(int cpu, size_t cpu_freq_mhz) {
    SetCpuFreqKhz(cpu, cpu_freq_mhz * 1000);
  }

  void SetCpuFreqKhz(int cpu, size_t cpu_freq_khz) {
    SetCpuMinFreqKhz(cpu, cpu_freq_khz);
    SetCpuMaxFreqKhz(cpu, cpu_freq_khz);
  }

  void SetCpuMinFreqKhz(int cpu, size_t cpu_freq_khz) {
    std::string cpu_str = hshm::Formatter::format(
        "/sys/devices/system/cpu/cpu{}/cpufreq/scaling_min_freq",
        cpu);
    std::ofstream min_freq_file(cpu_str);
    min_freq_file << cpu_freq_khz;
  }

  void SetCpuMaxFreqKhz(int cpu, size_t cpu_freq_khz) {
    std::string cpu_str = hshm::Formatter::format(
        "/sys/devices/system/cpu/cpu{}/cpufreq/scaling_max_freq",
        cpu);
    std::ofstream max_freq_file(cpu_str);
    max_freq_file << cpu_freq_khz;
  }
};

}  // namespace hshm

#endif  // HERMES_SYSINFO_INFO_H_
