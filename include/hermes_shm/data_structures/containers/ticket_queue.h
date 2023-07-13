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

#ifndef HERMES_SHM_INCLUDE_HERMES_SHM_DATA_STRUCTURES_TICKET_QUEUE_H_
#define HERMES_SHM_INCLUDE_HERMES_SHM_DATA_STRUCTURES_TICKET_QUEUE_H_

#include "hermes_shm/data_structures/ipc/internal/shm_internal.h"
#include "hermes_shm/thread/lock.h"
#include "spsc_queue.h"
#include "hermes_shm/types/qtok.h"

namespace hshm {

/**
 * A MPMC queue for allocating tickets. Handles concurrency
 * without blocking.
 * */
template<typename T>
class ticket_queue {
 public:
  spsc_queue<T> queue_;
  hshm::Mutex lock_;

 public:
  /**====================================
   * Default Constructor
   * ===================================*/

  /** Default constructor. */
  explicit ticket_queue(size_t depth = 1024) : queue_(depth) {}

  /** Copy constructor */
  ticket_queue(const ticket_queue &other) {
    queue_ = other.queue_;
  }

  /** Copy assignment operator */
  ticket_queue &operator=(const ticket_queue &other) {
    queue_ = other.queue_;
    return *this;
  }

  /** Move constructor */
  ticket_queue(ticket_queue &&other) noexcept {
    queue_ = std::move(other.queue_);
  }

  /** Move assignment operator */
  ticket_queue &operator=(ticket_queue &&other) noexcept {
    queue_ = std::move(other.queue_);
    return *this;
  }

  /**====================================
   * ticket Queue Methods
   * ===================================*/

  /** Construct an element at \a pos position in the queue */
  template<typename ...Args>
  HSHM_ALWAYS_INLINE qtok_t emplace(T &tkt) {
    lock_.Lock(0);
    auto qtok = queue_.emplace(tkt);
    lock_.Unlock();
    return qtok;
  }

 public:
  /** Pop an element from the queue */
  HSHM_ALWAYS_INLINE qtok_t pop(T &tkt) {
    lock_.Lock(0);
    auto qtok = queue_.pop(tkt);
    lock_.Unlock();
    return qtok;
  }
};

}  // namespace hshm

#endif  // HERMES_SHM_INCLUDE_HERMES_SHM_DATA_STRUCTURES_TICKET_QUEUE_H_
