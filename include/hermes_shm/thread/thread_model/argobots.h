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

#ifndef HERMES_SHM_INCLUDE_HERMES_SHM_THREAD_THALLIUM_H_
#define HERMES_SHM_INCLUDE_HERMES_SHM_THREAD_THALLIUM_H_

#include <errno.h>
#include <omp.h>

#include <thallium.hpp>

#include "hermes_shm/introspect/system_info.h"
#include "hermes_shm/util/errors.h"
#include "thread_model.h"

namespace hshm::thread {

class Argobots : public ThreadModel {
 public:
  /** Default constructor */
  HSHM_INLINE_CROSS_FUN
  Argobots() : ThreadModel(ThreadType::kArgobots) {}

  /** Virtual destructor */
  HSHM_CROSS_FUN
  virtual ~Argobots() = default;

  /** Yield the current thread for a period of time */
  HSHM_CROSS_FUN
  void SleepForUs(size_t us) override {
    /**
     * TODO(llogan): make this API flexible enough to support argobots fully
     * tl::thread::self().sleep(*HERMES->rpc_.server_engine_,
                               HERMES->server_config_.borg_.blob_reorg_period_);
     */
#ifdef HSHM_IS_HOST
    usleep(us);
#endif
  }

  /** Yield thread time slice */
  HSHM_CROSS_FUN
  void Yield() override {
#ifdef HSHM_IS_HOST
    ABT_thread_yield();
#endif
  }

  /** Create thread-local storage */
  template <typename TLS>
  HSHM_CROSS_FUN bool CreateTls(ThreadLocalKey &key, TLS *data) {
#ifdef HSHM_IS_HOST
    int ret = ABT_key_create(ThreadLocalData::template destroy_wrap<TLS>,
                             &key.argobots_key_);
    if (ret != ABT_SUCCESS) {
      return false;
    }
    return SetTls(key, data);
#else
    return false;
#endif
  }

  /** Create thread-local storage */
  template <typename TLS>
  HSHM_CROSS_FUN bool SetTls(ThreadLocalKey &key, TLS *data) {
#ifdef HSHM_IS_HOST
    int ret = ABT_key_set(key.argobots_key_, data);
    return ret == ABT_SUCCESS;
#else
    return false;
#endif
  }

  /** Get thread-local storage */
  template <typename TLS>
  HSHM_CROSS_FUN TLS *GetTls(const ThreadLocalKey &key) {
#ifdef HSHM_IS_HOST
    TLS *data;
    ABT_key_get(key.argobots_key_, (void **)&data);
    return (TLS *)data;
#else
    return nullptr;
#endif
  }

  /** Get the TID of the current thread */
  HSHM_CROSS_FUN
  ThreadId GetTid() override {
#ifdef HSHM_IS_HOST
    ABT_thread thread;
    ABT_thread_id tid;
    ABT_thread_self(&thread);
    ABT_thread_get_id(thread, &tid);
    return ThreadId{tid};
#else
    return ThreadId{0};
#endif
  }
};

}  // namespace hshm::thread

#endif  // HERMES_SHM_INCLUDE_HERMES_SHM_THREAD_THALLIUM_H_
