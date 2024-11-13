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

#ifndef HERMES_THREAD_PTHREAD_H_
#define HERMES_THREAD_PTHREAD_H_

#include "thread_model.h"
#include <errno.h>
#include "hermes_shm/util/errors.h"
#include <omp.h>
#include "hermes_shm/introspect/system_info.h"

namespace hshm::thread {

class Pthread : public ThreadModel {
 public:
  /** Default constructor */
  HSHM_INLINE_CROSS_FUN
  Pthread() : ThreadModel(ThreadType::kPthread) {}

  /** Virtual destructor */
  virtual ~Pthread() = default;

  /** Yield the thread for a period of time */
  HSHM_CROSS_FUN
  void SleepForUs(size_t us) override {
#ifdef HSHM_IS_HOST
    usleep(us);
#endif
  }

  /** Yield thread time slice */
  HSHM_CROSS_FUN
  void Yield() override {
#ifdef HSHM_IS_HOST
    sched_yield();
#endif
  }

  /** Create thread-local storage */
  template<typename TLS>
  HSHM_CROSS_FUN
  bool CreateTls(ThreadLocalKey &key, TLS *data) {
#ifdef HSHM_IS_HOST
    int ret = pthread_key_create(&key.pthread_key_,
                                 ThreadLocalData::destroy_wrap<TLS>);
    if (ret != 0) {
      return false;
    }
    return SetTls(key, data);
#else
    return false;
#endif
  }

  /** Create thread-local storage */
  template<typename TLS>
  HSHM_CROSS_FUN
  bool SetTls(ThreadLocalKey &key, TLS *data) {
#ifdef HSHM_IS_HOST
    pthread_setspecific(key.pthread_key_, data);
    return true;
#else
    return false;
#endif
  }

  /** Get thread-local storage */
  template<typename TLS>
  HSHM_CROSS_FUN
  TLS* GetTls(const ThreadLocalKey &key) {
#ifdef HSHM_IS_HOST
    TLS *data = (TLS*)pthread_getspecific(key.pthread_key_);
    return data;
#else
    return nullptr;
#endif
  }

  /** Get the TID of the current thread */
  ThreadId GetTid() override {
    return ThreadId{(hshm::u64)omp_get_thread_num()};
    // return static_cast<ThreadId>(pthread_self());
  }
};

}  // namespace hshm::thread

#endif  // HERMES_THREAD_PTHREAD_H_
