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
#include "hermes_shm/util/singleton/_global_singleton.h"
#include "hermes_shm/util/formatter.h"
#include <iostream>
#include <fstream>

#define HERMES_SYSTEM_INFO \
  hshm::GlobalSingleton<hshm::SystemInfo>::GetInstance()
#define HERMES_SYSTEM_INFO_T hshm::SystemInfo*

namespace hshm {

struct SystemInfo {
  int pid_;
  int ncpu_;
  int page_size_;
  int uid_;
  int gid_;
  size_t ram_size_;
  std::vector<size_t> cur_cpu_freq_;

  SystemInfo() {
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
  }

  size_t RefreshCpuFreqKhz() {
    for (int i = 0; i < ncpu_; ++i) {
      cur_cpu_freq_[i] = GetCpuFreqKhz(i);
    }
  }

  size_t GetCpuFreqKhz(int cpu) {
    // Read /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_cur_freq
    std::string cpu_str = hshm::Formatter::format(
        "/sys/devices/system/cpu/cpu{}/cpufreq/cpuinfo_cur_freq",
        cpu);
    std::ifstream cpu_file(cpu_str);
    size_t freq_khz;
    cpu_file >> freq_khz;
    return freq_khz;
  }

  void SetCpuFreqMhz(int cpu, size_t cpu_freq_khz) {
    SetCpuMinFreqMhz(cpu, cpu_freq_khz);
    SetCpuMaxFreqMhz(cpu, cpu_freq_khz);
  }

  void SetCpuMinFreqMhz(int cpu, size_t cpu_freq_khz) {
    std::string cpu_str = hshm::Formatter::format(
        "/sys/devices/system/cpu/cpu{}/cpufreq/scaling_min_freq",
        cpu);
    std::ofstream min_freq_file(cpu_str);
    min_freq_file << cpu_freq_khz;
  }

  void SetCpuMaxFreqMhz(int cpu, size_t cpu_freq_khz) {
    std::string cpu_str = hshm::Formatter::format(
        "/sys/devices/system/cpu/cpu{}/cpufreq/scaling_max_freq",
        cpu);
    std::ofstream max_freq_file(cpu_str);
    max_freq_file << cpu_freq_khz;
  }
};

}  // namespace hshm

#endif  // HERMES_SYSINFO_INFO_H_
