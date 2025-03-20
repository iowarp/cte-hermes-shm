#ifndef HSHM_SHM_THREAD_THREAD_MODEL_WINDOWS_H_
#define HSHM_SHM_THREAD_THREAD_MODEL_WINDOWS_H_

#include <thread>

#include "hermes_shm/introspect/system_info.h"
#include "thread_model.h"

namespace hshm::thread {

/** Represents the generic operations of a thread */
class WindowsThread : public ThreadModel {
 public:
  /** Initializer */
  HSHM_INLINE_CROSS_FUN
  WindowsThread() : ThreadModel(ThreadType::kWindows) {}

  /** Sleep thread for a period of time */
  HSHM_CROSS_FUN
  void SleepForUs(size_t us) {
    std::this_thread::sleep_for(std::chrono::microseconds(us));
  }

  /** Yield thread time slice */
  HSHM_CROSS_FUN
  void Yield() { std::this_thread::yield(); }

  /** Create thread-local storage */
  template <typename TLS>
  HSHM_CROSS_FUN bool CreateTls(ThreadLocalKey &key, TLS *data) {
#ifdef HSHM_IS_HOST
    return SystemInfo::CreateTls(key, (void *)data);
#else
    return false;
#endif
  }

  /** Create thread-local storage */
  template <typename TLS>
  HSHM_CROSS_FUN bool SetTls(ThreadLocalKey &key, TLS *data) {
#ifdef HSHM_IS_HOST
    return SystemInfo::SetTls(key, (void *)data);
#else
    return false;
#endif
  }

  /** Get thread-local storage */
  template <typename TLS>
  HSHM_CROSS_FUN TLS *GetTls(const ThreadLocalKey &key) {
#ifdef HSHM_IS_HOST
    return static_cast<TLS *>(SystemInfo::GetTls(key));
#else
    return nullptr;
#endif
  }

  /** Get the TID of the current thread */
  HSHM_CROSS_FUN
  ThreadId GetTid() { return ThreadId(SystemInfo::GetTid()); }

  /** Get the thread model type */
  HSHM_INLINE_CROSS_FUN
  ThreadType GetType() { return type_; }
};

}  // namespace hshm::thread

#endif  // HSHM_SHM_THREAD_THREAD_MODEL_WINDOWS_H_
