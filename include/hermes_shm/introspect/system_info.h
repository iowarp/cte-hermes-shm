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

#ifdef HERMES_ENABLE_PROCFS_SYSINFO
#include <sys/sysinfo.h>
#include <unistd.h>
#elifdef HERMES_ENABLE_WINDOWS_SYSINFO
#include <windows.h>
#else
#error \
    "Must define either HERMES_ENABLE_PROCFS_SYSINFO or HERMES_ENABLE_WINDOWS_SYSINFO"
#endif

#include <fstream>
#include <iostream>

#include "hermes_shm/util/formatter.h"
#include "hermes_shm/util/singleton/_singleton.h"

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
    pid_ = GetPid();
    ncpu_ = GetCpuCount();
    page_size_ = getpagesize();
    uid_ = GetUid();
    gid_ = GetGid();
    ram_size_ = GetRamCapacity();
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
#ifdef HERMES_ENABLE_PROCFS_SYSINFO
    // Read /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_cur_freq
    std::string cpu_str = hshm::Formatter::format(
        "/sys/devices/system/cpu/cpu{}/cpufreq/cpuinfo_cur_freq", cpu);
    std::ifstream cpu_file(cpu_str);
    size_t freq_khz;
    cpu_file >> freq_khz;
    return freq_khz;
#elifdef HERMES_ENABLE_WINDOWS_SYSINFO
    return 0;
#endif
#else
    return 0;
#endif
  }

  size_t GetCpuMaxFreqKhz(int cpu) {
#ifdef HSHM_IS_HOST
#ifdef HERMES_ENABLE_PROCFS_SYSINFO
    // Read /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_cur_freq
    std::string cpu_str = hshm::Formatter::format(
        "/sys/devices/system/cpu/cpu{}/cpufreq/cpuinfo_max_freq", cpu);
    std::ifstream cpu_file(cpu_str);
    size_t freq_khz;
    cpu_file >> freq_khz;
    return freq_khz;
#elifdef HERMES_ENABLE_WINDOWS_SYSINFO
    return 0;
#endif
#else
    return 0;
#endif
  }

  size_t GetCpuMinFreqKhz(int cpu) {
#ifdef HSHM_IS_HOST
#ifdef HERMES_ENABLE_PROCFS_SYSINFO
    // Read /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_cur_freq
    std::string cpu_str = hshm::Formatter::format(
        "/sys/devices/system/cpu/cpu{}/cpufreq/cpuinfo_min_freq", cpu);
    std::ifstream cpu_file(cpu_str);
    size_t freq_khz;
    cpu_file >> freq_khz;
    return freq_khz;
#elifdef HERMES_ENABLE_WINDOWS_SYSINFO
    return 0;
#endif
#else
    return 0;
#endif
  }

  size_t GetCpuMinFreqMhz(int cpu) { return GetCpuMinFreqKhz(cpu) / 1000; }

  size_t GetCpuMaxFreqMhz(int cpu) { return GetCpuMaxFreqKhz(cpu) / 1000; }

  void SetCpuFreqMhz(int cpu, size_t cpu_freq_mhz) {
    SetCpuFreqKhz(cpu, cpu_freq_mhz * 1000);
  }

  void SetCpuFreqKhz(int cpu, size_t cpu_freq_khz) {
    SetCpuMinFreqKhz(cpu, cpu_freq_khz);
    SetCpuMaxFreqKhz(cpu, cpu_freq_khz);
  }

  void SetCpuMinFreqKhz(int cpu, size_t cpu_freq_khz) {
#ifdef HERMES_ENABLE_PROCFS_SYSINFO
    std::string cpu_str = hshm::Formatter::format(
        "/sys/devices/system/cpu/cpu{}/cpufreq/scaling_min_freq", cpu);
    std::ofstream min_freq_file(cpu_str);
    min_freq_file << cpu_freq_khz;
#endif
  }

  void SetCpuMaxFreqKhz(int cpu, size_t cpu_freq_khz) {
#ifdef HERMES_ENABLE_PROCFS_SYSINFO
    std::string cpu_str = hshm::Formatter::format(
        "/sys/devices/system/cpu/cpu{}/cpufreq/scaling_max_freq", cpu);
    std::ofstream max_freq_file(cpu_str);
    max_freq_file << cpu_freq_khz;
#endif
  }

  static int GetCpuCount() {
#ifdef HERMES_ENABLE_PROCFS_SYSINFO
    return get_nprocs_conf();
#elifdef HERMES_ENABLE_WINDOWS_SYSINFO
    SYSTEM_INFO sys_info;
    GetSystemInfo(&sys_info);
    return sys_info.dwNumberOfProcessors;
#endif
  }

  static int GetPageSize() {
#ifdef HERMES_ENABLE_PROCFS_SYSINFO
    return getpagesize();
#elifdef HERMES_ENABLE_WINDOWS_SYSINFO
    SYSTEM_INFO sys_info;
    GetSystemInfo(&sys_info);
    return sys_info.dwPageSize;
#endif
  }

  static int GetTid() {
#ifdef HERMES_ENABLE_PROCFS_SYSINFO
#ifdef SYS_gettid
    return (pid_t)syscall(SYS_gettid);
#else
#warning "GetTid is not defined"
    return GetPid();
#endif
#elifdef HERMES_ENABLE_WINDOWS_SYSINFO
    return GetCurrentThreadId();
#endif
  }

  static int GetPid() {
#ifdef HERMES_ENABLE_PROCFS_SYSINFO
#ifdef SYS_getpid
    return (pid_t)syscall(SYS_getpid);
#else
#warning "GetPid is not defined"
    return 0;
#endif
#elifdef HERMES_ENABLE_WINDOWS_SYSINFO
    return GetCurrentProcessId();
#endif
  }

  static int GetUid() {
#ifdef HERMES_ENABLE_PROCFS_SYSINFO
    return getuid();
#elifdef HERMES_ENABLE_WINDOWS_SYSINFO
    return 0;
#endif
  };

  static size_t GetRamCapacity() {
#ifdef HERMES_ENABLE_PROCFS_SYSINFO
    struct sysinfo info;
    sysinfo(&info);
    return info.totalram;
#elifdef HERMES_ENABLE_WINDOWS_SYSINFO
    MEMORYSTATUSEX mem_info;
    mem_info.dwLength = sizeof(mem_info);
    GlobalMemoryStatusEx(&mem_info);
    return mem_info.ullTotalPhys;
#endif
  }

}  // namespace hshm

#endif  // HERMES_SYSINFO_INFO_H_
