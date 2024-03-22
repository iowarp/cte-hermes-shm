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

#ifndef HERMES_SHM_INCLUDE_HERMES_SHM_DATA_STRUCTURES_spsc_queue_H_
#define HERMES_SHM_INCLUDE_HERMES_SHM_DATA_STRUCTURES_spsc_queue_H_

#include "hermes_shm/data_structures/ipc/internal/shm_internal.h"
#include "hermes_shm/util/auto_trace.h"
#include "hermes_shm/thread/lock.h"
#include "hermes_shm/types/qtok.h"
#include <vector>

namespace hshm {

/** Forward declaration of spsc_queue */
template<typename T>
class spsc_queue;

/**
 * A queue optimized for multiple producers (emplace) with a single
 * consumer (pop).
 * */
template<typename T>
class spsc_queue {
 public:
  std::vector<T> queue_;
  _qtok_t head_, tail_;

 public:
  /**====================================
   * spsc Constructors
   * ===================================*/

  /** Default constructor */
  spsc_queue() : head_(0), tail_(0) {}

  /** Emplace constructor */
  explicit spsc_queue(size_t max_size) : head_(0), tail_(0) {
    queue_.resize(max_size);
  }

  /** Copy constructor */
  spsc_queue(const spsc_queue &other) {
    queue_ = other.queue_;
    head_ = other.head_;
    tail_ = other.tail_;
  }

  /** Copy assignment operator */
  spsc_queue &operator=(const spsc_queue &other) {
    queue_ = other.queue_;
    head_ = other.head_;
    tail_ = other.tail_;
    return *this;
  }

  /** Move constructor */
  spsc_queue(spsc_queue &&other) noexcept {
    queue_ = std::move(other.queue_);
    head_ = other.head_;
    tail_ = other.tail_;
  }

  /** Move assignment operator */
  spsc_queue &operator=(spsc_queue &&other) noexcept {
    queue_ = std::move(other.queue_);
    head_ = other.head_;
    tail_ = other.tail_;
    return *this;
  }

  /**====================================
   * spsc Queue Methods
   * ===================================*/

  /** Resize SPSC queue (not thread safe) */
  void Resize(size_t max_size) {
    spsc_queue new_queue(max_size);
    T val;
    while (!pop(val).IsNull()) {
      new_queue.emplace(val);
    }
    (*this) = std::move(new_queue);
  }

  /** Construct an element at \a pos position in the list */
  template<typename ...Args>
  qtok_t emplace(Args&&... args) {
    // Don't emplace if there is no space
    _qtok_t entry_tok = tail_;
    size_t size = tail_ - head_;
    if (size >= queue_.size()) {
      return qtok_t::GetNull();
    }

    // Do the emplace
    _qtok_t idx = entry_tok % queue_.size();
    hipc::Allocator::ConstructObj(queue_[idx], std::forward<Args>(args)...);
    tail_ += 1;
    return qtok_t(entry_tok);
  }

  /** Consumer pops the head object */
  qtok_t pop(T &val) {
    // Don't pop if there's no entries
    _qtok_t head = head_;
    _qtok_t tail = tail_;
    if (head >= tail) {
      return qtok_t::GetNull();
    }

    // Pop the element
    _qtok_t idx = head % queue_.size();
    T &entry = queue_[idx];
    (val) = std::move(entry);
    head_ += 1;
    return qtok_t(head);
  }

  /** Get current size of SPSC queue */
  size_t size() {
    return tail_ - head_;
  }
};

}  // namespace hshm

#endif  // HERMES_SHM_INCLUDE_HERMES_SHM_DATA_STRUCTURES_spsc_queue_H_
