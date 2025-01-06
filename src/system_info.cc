#define HSHM_IS_COMPILING_SINGLETONS
#define HSHM_IS_COMPILING

#include "hermes_shm/introspect/system_info.h"

#include <cstdlib>

#include "hermes_shm/constants/macros.h"
#if defined(HERMES_ENABLE_PROCFS_SYSINFO)
#include <sys/sysinfo.h>
#include <unistd.h>
#elif defined(HERMES_ENABLE_WINDOWS_SYSINFO)
#include <windows.h>
#else
#error \
    "Must define either HERMES_ENABLE_PROCFS_SYSINFO or HERMES_ENABLE_WINDOWS_SYSINFO"
#endif

namespace hshm {

HSHM_CROSS_FUN
void SystemInfo::RefreshInfo() {
#ifdef HSHM_IS_HOST
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

void SystemInfo::RefreshCpuFreqKhz() {
#ifdef HSHM_IS_HOST
  for (int i = 0; i < ncpu_; ++i) {
    cur_cpu_freq_[i] = GetCpuFreqKhz(i);
  }
#endif
}

size_t SystemInfo::GetCpuFreqKhz(int cpu) {
#ifdef HSHM_IS_HOST
#if defined(HERMES_ENABLE_PROCFS_SYSINFO)
  // Read /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_cur_freq
  std::string cpu_str = hshm::Formatter::format(
      "/sys/devices/system/cpu/cpu{}/cpufreq/cpuinfo_cur_freq", cpu);
  std::ifstream cpu_file(cpu_str);
  size_t freq_khz;
  cpu_file >> freq_khz;
  return freq_khz;
#elif defined(HERMES_ENABLE_WINDOWS_SYSINFO)
  return 0;
#endif
#else
  return 0;
#endif
}

size_t SystemInfo::GetCpuMaxFreqKhz(int cpu) {
#ifdef HSHM_IS_HOST
#if defined(HERMES_ENABLE_PROCFS_SYSINFO)
  // Read /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_cur_freq
  std::string cpu_str = hshm::Formatter::format(
      "/sys/devices/system/cpu/cpu{}/cpufreq/cpuinfo_max_freq", cpu);
  std::ifstream cpu_file(cpu_str);
  size_t freq_khz;
  cpu_file >> freq_khz;
  return freq_khz;
#elif defined(HERMES_ENABLE_WINDOWS_SYSINFO)
  return 0;
#endif
#else
  return 0;
#endif
}

size_t SystemInfo::GetCpuMinFreqKhz(int cpu) {
#ifdef HSHM_IS_HOST
#if defined(HERMES_ENABLE_PROCFS_SYSINFO)
  // Read /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_cur_freq
  std::string cpu_str = hshm::Formatter::format(
      "/sys/devices/system/cpu/cpu{}/cpufreq/cpuinfo_min_freq", cpu);
  std::ifstream cpu_file(cpu_str);
  size_t freq_khz;
  cpu_file >> freq_khz;
  return freq_khz;
#elif defined(HERMES_ENABLE_WINDOWS_SYSINFO)
  return 0;
#endif
#else
  return 0;
#endif
}

size_t SystemInfo::GetCpuMinFreqMhz(int cpu) {
  return GetCpuMinFreqKhz(cpu) / 1000;
}

size_t SystemInfo::GetCpuMaxFreqMhz(int cpu) {
  return GetCpuMaxFreqKhz(cpu) / 1000;
}

void SystemInfo::SetCpuFreqMhz(int cpu, size_t cpu_freq_mhz) {
  SetCpuFreqKhz(cpu, cpu_freq_mhz * 1000);
}

void SystemInfo::SetCpuFreqKhz(int cpu, size_t cpu_freq_khz) {
  SetCpuMinFreqKhz(cpu, cpu_freq_khz);
  SetCpuMaxFreqKhz(cpu, cpu_freq_khz);
}

void SystemInfo::SetCpuMinFreqKhz(int cpu, size_t cpu_freq_khz) {
#if defined(HERMES_ENABLE_PROCFS_SYSINFO)
  std::string cpu_str = hshm::Formatter::format(
      "/sys/devices/system/cpu/cpu{}/cpufreq/scaling_min_freq", cpu);
  std::ofstream min_freq_file(cpu_str);
  min_freq_file << cpu_freq_khz;
#endif
}

void SystemInfo::SetCpuMaxFreqKhz(int cpu, size_t cpu_freq_khz) {
#if defined(HERMES_ENABLE_PROCFS_SYSINFO)
  std::string cpu_str = hshm::Formatter::format(
      "/sys/devices/system/cpu/cpu{}/cpufreq/scaling_max_freq", cpu);
  std::ofstream max_freq_file(cpu_str);
  max_freq_file << cpu_freq_khz;
#endif
}

int SystemInfo::GetCpuCount() {
#if defined(HERMES_ENABLE_PROCFS_SYSINFO)
  return get_nprocs_conf();
#elif defined(HERMES_ENABLE_WINDOWS_SYSINFO)
  SYSTEM_INFO sys_info;
  GetSystemInfo(&sys_info);
  return sys_info.dwNumberOfProcessors;
#endif
}

int SystemInfo::GetPageSize() {
#if defined(HERMES_ENABLE_PROCFS_SYSINFO)
  return getpagesize();
#elif defined(HERMES_ENABLE_WINDOWS_SYSINFO)
  SYSTEM_INFO sys_info;
  GetSystemInfo(&sys_info);
  return sys_info.dwPageSize;
#endif
}

int SystemInfo::GetTid() {
#if defined(HERMES_ENABLE_PROCFS_SYSINFO)
#ifdef SYS_gettid
  return (pid_t)syscall(SYS_gettid);
#else
#warning "GetTid is not defined"
  return GetPid();
#endif
#elif defined(HERMES_ENABLE_WINDOWS_SYSINFO)
  return GetCurrentThreadId();
#endif
}

int SystemInfo::GetPid() {
#if defined(HERMES_ENABLE_PROCFS_SYSINFO)
#ifdef SYS_getpid
  return (pid_t)syscall(SYS_getpid);
#else
#warning "GetPid is not defined"
  return 0;
#endif
#elif defined(HERMES_ENABLE_WINDOWS_SYSINFO)
  return GetCurrentProcessId();
#endif
}

int SystemInfo::GetUid() {
#if defined(HERMES_ENABLE_PROCFS_SYSINFO)
  return getuid();
#elif defined(HERMES_ENABLE_WINDOWS_SYSINFO)
  return 0;
#endif
};

int SystemInfo::GetGid() {
#if defined(HERMES_ENABLE_PROCFS_SYSINFO)
  return getgid();
#elif defined(HERMES_ENABLE_WINDOWS_SYSINFO)
  return 0;
#endif
};

size_t SystemInfo::GetRamCapacity() {
#if defined(HERMES_ENABLE_PROCFS_SYSINFO)
  struct sysinfo info;
  sysinfo(&info);
  return info.totalram;
#elif defined(HERMES_ENABLE_WINDOWS_SYSINFO)
  MEMORYSTATUSEX mem_info;
  mem_info.dwLength = sizeof(mem_info);
  GlobalMemoryStatusEx(&mem_info);
  return (size_t)mem_info.ullTotalPhys;
#endif
}

void SystemInfo::YieldThread() {
#if defined(HERMES_ENABLE_PROCFS_SYSINFO)
  sched_yield();
#elif defined(HERMES_ENABLE_WINDOWS_SYSINFO)
  Yield();
#endif
}

bool SystemInfo::CreateTls(ThreadLocalKey &key, void *data) {
#ifdef HERMES_ENABLE_PROCFS_SYSINFO
  key.posix_key_ = pthread_key_create(&key.posix_key_, nullptr);
  return key.posix_key_ == 0;
#elif defined(HERMES_ENABLE_WINDOWS_SYSINFO)
  key.windows_key_ = TlsAlloc();
  if (key.windows_key_ == TLS_OUT_OF_INDEXES) {
    return false;
  }
  return TlsSetValue(key.windows_key_, data);
#endif
}

bool SystemInfo::SetTls(const ThreadLocalKey &key, void *data) {
#ifdef HERMES_ENABLE_PROCFS_SYSINFO
  return pthread_setspecific(key.posix_key_, data) == 0;
#elif defined(HERMES_ENABLE_WINDOWS_SYSINFO)
  return TlsSetValue(key.windows_key_, data);
#endif
}

void *SystemInfo::GetTls(const ThreadLocalKey &key) {
#ifdef HERMES_ENABLE_PROCFS_SYSINFO
  return pthread_getspecific(key.posix_key_);
#elif defined(HERMES_ENABLE_WINDOWS_SYSINFO)
  return TlsGetValue(key.windows_key_);
#endif
}

bool SystemInfo::CreateNewSharedMemory(File &fd, const std::string &name,
                                       size_t size) {
#if defined(HERMES_ENABLE_PROCFS_SYSINFO)
  fd.posix_fd_ = shm_open(name.c_str(), O_CREAT | O_RDWR, 0666);
  if (fd.posix_fd_ < 0) {
    return false;
  }
  int ret = ftruncate(fd.posix_fd_, size);
  if (ret < 0) {
    close(fd.posix_fd_);
    return false;
  }
#elif defined(HERMES_ENABLE_WINDOWS_SYSINFO)
  fd.windows_fd_ =
      CreateFileMapping(INVALID_HANDLE_VALUE,  // use paging file
                        nullptr,               // default security
                        PAGE_READWRITE,        // read/write access
                        0,         // maximum object size (high-order DWORD)
                        size,      // maximum object size (low-order DWORD)
                        nullptr);  // name of mapping object
  return fd.windows_fd_ != nullptr;
#endif
}

bool SystemInfo::OpenSharedMemory(File &fd, const std::string &name) {
#if defined(HERMES_ENABLE_PROCFS_SYSINFO)
  fd.posix_fd_ = shm_open(name.c_str(), O_RDWR, 0666);
  return fd.posix_fd_ >= 0;
#elif defined(HERMES_ENABLE_WINDOWS_SYSINFO)
  fd.windows_fd_ = OpenFileMapping(FILE_MAP_ALL_ACCESS, FALSE, name.c_str());
  return fd.windows_fd_ != nullptr;
#endif
}

void SystemInfo::CloseSharedMemory(File &file) {
#if defined(HERMES_ENABLE_PROCFS_SYSINFO)
  close(file.posix_fd_);
#elif defined(HERMES_ENABLE_WINDOWS_SYSINFO)
  CloseHandle(file.windows_fd_);
#endif
}

void SystemInfo::DestroySharedMemory(const std::string &name) {
#if defined(HERMES_ENABLE_PROCFS_SYSINFO)
  shm_unlink(name.c_str());
#elif defined(HERMES_ENABLE_WINDOWS_SYSINFO)
#endif
}

void *SystemInfo::MapPrivateMemory(size_t size) {
#if defined(HERMES_ENABLE_PROCFS_SYSINFO)
  return mmap64(nullptr, size, PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  return ptr;
#elif defined(HERMES_ENABLE_WINDOWS_SYSINFO)
  return VirtualAlloc(nullptr, size, MEM_COMMIT, PAGE_READWRITE);
#endif
}

void *SystemInfo::MapSharedMemory(const File &fd, size_t size, i64 off) {
#if defined(HERMES_ENABLE_PROCFS_SYSINFO)
  return mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS,
              -1, 0);
#elif defined(HERMES_ENABLE_WINDOWS_SYSINFO)
  return MapViewOfFile(fd.windows_fd_,       // handle to map object
                       FILE_MAP_ALL_ACCESS,  // read/write permission
                       0,                    // file offset high
                       (DWORD)off,           // file offset low
                       size);                // number of bytes to map
#endif
}

void SystemInfo::UnmapMemory(void *ptr, size_t size) {
#if defined(HERMES_ENABLE_PROCFS_SYSINFO)
  munmap(ptr, size);
#elif defined(HERMES_ENABLE_WINDOWS_SYSINFO)
  VirtualFree(ptr, size, MEM_RELEASE);
#endif
}

void *SystemInfo::AlignedAlloc(size_t alignment, size_t size) {
#if defined(HERMES_ENABLE_PROCFS_SYSINFO)
  return aligned_alloc(alignment, size);
#elif defined(HERMES_ENABLE_WINDOWS_SYSINFO)
  return _aligned_malloc(size, alignment);
#endif
}

std::string SystemInfo::getenv(const char *name, size_t max_size) {
#if defined(HERMES_ENABLE_PROCFS_SYSINFO)
  char *var = std::getenv(name);
  if (var == nullptr) {
    return "";
  }
  return std::string(var);
#elif defined(HERMES_ENABLE_WINDOWS_SYSINFO)
  std::string var;
  var.resize(max_size);
  GetEnvironmentVariable(name, var.data(), var.size());
  return var;
#endif
}

void SystemInfo::setenv(const char *name, const std::string &value,
                        int overwrite) {
#if defined(HERMES_ENABLE_PROCFS_SYSINFO)
  std::setenv(name, value, overwrite);
#elif defined(HERMES_ENABLE_WINDOWS_SYSINFO)
  SetEnvironmentVariable(name, value.c_str());
#endif
}

void SystemInfo::unsetenv(const char *name) {
#if defined(HERMES_ENABLE_PROCFS_SYSINFO)
  std::unsetenv(name);
#elif defined(HERMES_ENABLE_WINDOWS_SYSINFO)
  SetEnvironmentVariable(name, nullptr);
#endif
}

}  // namespace hshm