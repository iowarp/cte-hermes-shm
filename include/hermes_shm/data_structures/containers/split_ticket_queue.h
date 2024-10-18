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

#ifndef HERMES_SHM__DATA_STRUCTURES_SPLIT_TICKET_QUEUE_H_
#define HERMES_SHM__DATA_STRUCTURES_SPLIT_TICKET_QUEUE_H_

#include "hermes_shm/data_structures/ipc/internal/shm_internal.h"
#include "hermes_shm/thread/lock.h"
#include "ticket_queue.h"
#include <vector>

namespace hshm {

/**
 * A MPMC queue for allocating tickets. Handles concurrency
 * without blocking.
 * */
template<typename T>
class split_ticket_queue {
 public:
  std::vector<ticket_queue<T>> splits_;
  std::atomic<uint16_t> rr_tail_, rr_head_;

 public:
  /**====================================
   * Default Constructor
   * ===================================*/

  /** Default constructor */
  explicit split_ticket_queue(size_t depth_per_split = 1024,
                              size_t split = 0) {
    if (split == 0) {
      split = HERMES_SYSTEM_INFO->ncpu_;
    }
    splits_.reserve(split);
    for (size_t i = 0; i < split; ++i) {
      splits_.emplace_back(depth_per_split);
    }
  }

  /** Copy constructor */
  split_ticket_queue(const split_ticket_queue<T> &other) {
    splits_ = other.splits_;
    rr_tail_ = other.rr_tail_.load();
    rr_head_ = other.rr_head_.load();
  }

  /** Copy assignment operator */
  split_ticket_queue &operator=(const split_ticket_queue<T> &other) {
    splits_ = other.splits_;
    rr_tail_ = other.rr_tail_.load();
    rr_head_ = other.rr_head_.load();
    return *this;
  }

  /** Move constructor */
  split_ticket_queue(split_ticket_queue<T> &&other) {
    splits_ = std::move(other.splits_);
    rr_tail_ = other.rr_tail_.load();
    rr_head_ = other.rr_head_.load();
  }

  /** Move assignment operator */
  split_ticket_queue &operator=(split_ticket_queue<T> &&other) {
    splits_ = std::move(other.splits_);
    rr_tail_ = other.rr_tail_.load();
    rr_head_ = other.rr_head_.load();
    return *this;
  }

  /**====================================
   * ticket Queue Methods
   * ===================================*/

  /** Construct an element at \a pos position in the queue */
  template<typename ...Args> qtok_t emplace(T &tkt) {
    uint16_t rr = rr_tail_.fetch_add(1);
    size_t num_splits = splits_.size();
    uint16_t qid_start = rr % num_splits;
    for (size_t i = 0; i < num_splits; ++i) {
      uint32_t qid = (qid_start + i) % num_splits;
      ticket_queue<T> &queue = (splits_)[qid];
      qtok_t qtok = queue.emplace(tkt);
      if (!qtok.IsNull()) {
        return qtok;
      }
    }
    return qtok_t::GetNull();
  }

 public:
  /** Pop an element from the queue */ qtok_t pop(T &tkt) {
    uint16_t rr = rr_head_.fetch_add(1);
    size_t num_splits = splits_.size();
    uint16_t qid_start = rr % num_splits;
    for (size_t i = 0; i < num_splits; ++i) {
      uint32_t qid = (qid_start + i) % num_splits;
      ticket_queue<T> &queue = (splits_)[qid];
      qtok_t qtok = queue.pop(tkt);
      if (!qtok.IsNull()) {
        return qtok;
      }
    }
    return qtok_t::GetNull();
  }
};

}  // namespace hshm

#endif  // HERMES_SHM__DATA_STRUCTURES_SPLIT_TICKET_QUEUE_H_
