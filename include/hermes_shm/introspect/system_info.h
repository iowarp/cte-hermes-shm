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

#ifndef HSHM_SYSINFO_INFO_H_
#define HSHM_SYSINFO_INFO_H_

#include "hermes_shm/constants/macros.h"
#if HSHM_ENABLE_PROCFS_SYSINFO
#ifdef __linux__
#include <sys/sysinfo.h>
#endif
#include <unistd.h>
#endif

#include <fstream>
#include <iostream>

#include "hermes_shm/thread/thread_model/thread_model.h"
#include "hermes_shm/util/formatter.h"
#include "hermes_shm/util/singleton.h"

#define HSHM_SYSTEM_INFO \
  hshm::LockfreeCrossSingleton<hshm::SystemInfo>::GetInstance()
#define HSHM_SYSTEM_INFO_T hshm::SystemInfo *

namespace hshm {

/** Dynamically load shared libraries */
struct SharedLibrary {
  void *handle_;

  SharedLibrary() = default;
  HSHM_DLL SharedLibrary(const std::string &name);
  HSHM_DLL ~SharedLibrary();

  // Delete copy operations
  SharedLibrary(const SharedLibrary &) = delete;
  SharedLibrary &operator=(const SharedLibrary &) = delete;

  // Move operations
  HSHM_DLL SharedLibrary(SharedLibrary &&other) noexcept;
  HSHM_DLL SharedLibrary &operator=(SharedLibrary &&other) noexcept;

  HSHM_DLL void Load(const std::string &name);
  HSHM_DLL void *GetSymbol(const std::string &name);
  HSHM_DLL std::string GetError() const;

  bool IsNull() { return handle_ == nullptr; }
};

/** File wrapper */
union File {
  int posix_fd_;
  HANDLE windows_fd_;
};

/** A unification of certain OS system calls */
class SystemInfo {
 public:
  int pid_;
  int ncpu_;
  int page_size_;
  int uid_;
  int gid_;
  size_t ram_size_;
#if HSHM_IS_HOST
  std::vector<size_t> cur_cpu_freq_;
#endif

 public:
  HSHM_CROSS_FUN
  SystemInfo() { RefreshInfo(); }

  HSHM_CROSS_FUN
  void RefreshInfo() {
#if HSHM_IS_HOST
    pid_ = GetPid();
    ncpu_ = GetCpuCount();
    page_size_ = GetPageSize();
    uid_ = GetUid();
    gid_ = GetGid();
    ram_size_ = GetRamCapacity();
    cur_cpu_freq_.resize(ncpu_);
    RefreshCpuFreqKhz();
#endif
  }

  HSHM_DLL void RefreshCpuFreqKhz();

  HSHM_DLL size_t GetCpuFreqKhz(int cpu);

  HSHM_DLL size_t GetCpuMaxFreqKhz(int cpu);

  HSHM_DLL size_t GetCpuMinFreqKhz(int cpu);

  HSHM_DLL size_t GetCpuMinFreqMhz(int cpu);

  HSHM_DLL size_t GetCpuMaxFreqMhz(int cpu);

  HSHM_DLL void SetCpuFreqMhz(int cpu, size_t cpu_freq_mhz);

  HSHM_DLL void SetCpuFreqKhz(int cpu, size_t cpu_freq_khz);

  HSHM_DLL void SetCpuMinFreqKhz(int cpu, size_t cpu_freq_khz);

  HSHM_DLL void SetCpuMaxFreqKhz(int cpu, size_t cpu_freq_khz);

  HSHM_DLL static int GetCpuCount();

  HSHM_DLL static int GetPageSize();

  HSHM_DLL static int GetTid();

  HSHM_DLL static int GetPid();

  HSHM_DLL static int GetUid();

  HSHM_DLL static int GetGid();

  HSHM_DLL static size_t GetRamCapacity();

  HSHM_DLL static void YieldThread();

  HSHM_DLL static bool CreateTls(ThreadLocalKey &key, void *data);

  HSHM_DLL static bool SetTls(const ThreadLocalKey &key, void *data);

  HSHM_DLL static void *GetTls(const ThreadLocalKey &key);

  HSHM_DLL static bool CreateNewSharedMemory(File &fd, const std::string &name,
                                             size_t size);

  HSHM_DLL static bool OpenSharedMemory(File &fd, const std::string &name);

  HSHM_DLL static void CloseSharedMemory(File &file);

  HSHM_DLL static void DestroySharedMemory(const std::string &name);

  HSHM_DLL static void *MapPrivateMemory(size_t size);

  HSHM_DLL static void *MapSharedMemory(const File &fd, size_t size, i64 off);

  HSHM_DLL static void UnmapMemory(void *ptr, size_t size);

  HSHM_DLL static void *AlignedAlloc(size_t alignment, size_t size);

  HSHM_DLL static std::string Getenv(
      const char *name, size_t max_size = hshm::Unit<size_t>::Megabytes(1));

  static std::string Getenv(
      const std::string &name,
      size_t max_size = hshm::Unit<size_t>::Megabytes(1)) {
    return Getenv(name.c_str(), max_size);
  }

  HSHM_DLL static void Setenv(const char *name, const std::string &value,
                              int overwrite);

  HSHM_DLL static void Unsetenv(const char *name);
};

}  // namespace hshm

#undef WIN32_LEAN_AND_MEAN

#endif  // HSHM_SYSINFO_INFO_H_
