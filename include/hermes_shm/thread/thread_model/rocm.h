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

#ifndef HERMES_SHM_INCLUDE_HERMES_SHM_THREAD_ROCM_H__
#define HERMES_SHM_INCLUDE_HERMES_SHM_THREAD_ROCM_H__

#include <errno.h>

#include "hermes_shm/introspect/system_info.h"
#include "hermes_shm/util/errors.h"
#include "thread_model.h"

namespace hshm::thread {

class Rocm : public ThreadModel {
 public:
  /** Default constructor */
  HSHM_INLINE_CROSS_FUN
  Rocm() : ThreadModel(ThreadType::kRocm) {}

  /** Virtual destructor */
  HSHM_CROSS_FUN
  virtual ~Rocm() = default;

  /** Yield the current thread for a period of time */
  HSHM_CROSS_FUN
  void SleepForUs(size_t us) {}

  /** Yield thread time slice */
  HSHM_CROSS_FUN
  void Yield() {}

  /** Create thread-local storage */
  template <typename TLS>
  HSHM_CROSS_FUN bool CreateTls(ThreadLocalKey &key, TLS *data) {
    return false;
  }

  /** Get thread-local storage */
  template <typename TLS>
  HSHM_CROSS_FUN TLS *GetTls(const ThreadLocalKey &key) {
    return nullptr;
  }

  /** Create thread-local storage */
  template <typename TLS>
  HSHM_CROSS_FUN bool SetTls(ThreadLocalKey &key, TLS *data) {
    return false;
  }

  /** Get the TID of the current thread */
  HSHM_CROSS_FUN
  ThreadId GetTid() { return ThreadId::GetNull(); }
};

}  // namespace hshm::thread

#endif  // HERMES_SHM_INCLUDE_HERMES_SHM_THREAD_ROCM_H__
