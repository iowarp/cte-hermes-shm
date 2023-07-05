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

#ifndef HERMES_SHM_INCLUDE_HERMES_SHM_DATA_STRUCTURES_IPC_spsc_queue_H_
#define HERMES_SHM_INCLUDE_HERMES_SHM_DATA_STRUCTURES_IPC_spsc_queue_H_

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
    queue_.reserve(max_size);
  }

  /**====================================
   * spsc Queue Methods
   * ===================================*/

  /** Resize SPSC queue (not thread safe) */
  void Resize(size_t max_size) {
    queue_.reserve(max_size);
  }

  /** Construct an element at \a pos position in the list */
  template<typename ...Args>
  qtok_t emplace(Args&&... args) {
    // Don't emplace if there is no space
    _qtok_t entry_tok = tail_;
    size_t size = tail_ - head_;
    if (size >= queue_.capacity()) {
      return qtok_t::GetNull();
    }

    // Do the emplace
    _qtok_t idx = entry_tok % queue_.capacity();
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
    _qtok_t idx = head % queue_.capacity();
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

#endif  // HERMES_SHM_INCLUDE_HERMES_SHM_DATA_STRUCTURES_IPC_spsc_queue_H_
