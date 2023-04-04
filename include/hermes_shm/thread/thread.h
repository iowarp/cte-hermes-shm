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

#ifndef HERMES_THREAD_THREAD_H_
#define HERMES_THREAD_THREAD_H_

#include <vector>
#include <cstdint>
#include <memory>
#include <atomic>
#include "hermes_shm/types/bitfield.h"

namespace hshm {

typedef uint32_t tid_t;

/** Available threads that are mapped */
enum class ThreadType {
  kPthread
};

/** A bitfield representing CPU affinity */
typedef big_bitfield<CPU_SETSIZE> cpu_bitfield;

/** Represents the generic operations of a thread */
class Thread {
 public:
  /** Virtual destructor */
  virtual ~Thread() = default;

  /** Pause a thread */
  virtual void Pause() = 0;

  /** Resume a thread */
  virtual void Resume() = 0;

  /** Join the thread */
  virtual void Join() = 0;

  /** Set thread affinity to a single CPU */
  void SetAffinity(int cpu_id) {
    cpu_bitfield mask;
    mask.SetBits(cpu_id, 1);
    SetAffinity(mask);
  }

  /** Set thread affinity to the mask */
  virtual void SetAffinity(const cpu_bitfield &mask) = 0;

  /** Get thread affinity according to the mask */
  virtual void GetAffinity(cpu_bitfield &mask) = 0;

  /** Sleep thread for a period of time */
  virtual void SleepForUs(size_t us) = 0;

  /** Yield thread time slice */
  virtual void Yield() = 0;

  /** Get the TID of the current thread */
  virtual tid_t GetTid() = 0;
};

}  // namespace hshm

#endif  // HERMES_THREAD_THREAD_H_
