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


#ifndef HERMES_THREAD_THREAD_MANAGER_H_
#define HERMES_THREAD_THREAD_MANAGER_H_

#include "thread.h"
#include "thread_factory.h"
#include <hermes_shm/constants/data_structure_singleton_macros.h>
#include <hermes_shm/introspect/system_info.h>
#include <mutex>

#define US_TO_CLOCKS(x) (x * 56)

namespace hshm {

union NodeThreadId;

class ThreadManager {
 public:
  ThreadType type_; /**< The type of threads used in this program */
  std::unique_ptr<Thread>
    thread_static_; /**< Functions static to all threads */
  std::mutex lock_; /**< Synchronize */

  /** Default constructor */
  ThreadManager() {
    SetThreadType(ThreadType::kPthread);
  }

  /** Set the threading model of this application */
  void SetThreadType(ThreadType type) {
    lock_.lock();
    if (type_ == type) {
      return;
    }
    type_ = type;
    thread_static_ = ThreadFactory<int>(type).Get();
    lock_.unlock();
  }

  /** Sleep for a period of time (microseconds) */
  void SleepForUs(size_t us) {
    thread_static_->SleepForUs(us);
  }

  /** Call Yield */
  void Yield() {
    thread_static_->Yield();
  }

  /** Call GetTid */
  tid_t GetTid() {
    return thread_static_->GetTid();
  }
};

/** A unique identifier of this thread across processes */
union NodeThreadId {
  struct {
    uint32_t tid_;
    uint32_t pid_;
  } bits_;
  uint64_t as_int_;

  /** Default constructor */
  NodeThreadId() {
    bits_.tid_ = HSHM_THREAD_MANAGER->GetTid();
    bits_.pid_ = HERMES_SYSTEM_INFO->pid_;
  }

  /** Hash function */
  uint32_t hash() {
    return as_int_;
  }
};

}  // namespace hshm

#endif  // HERMES_THREAD_THREAD_MANAGER_H_
