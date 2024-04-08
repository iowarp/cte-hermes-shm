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

#ifndef HERMES_SHM_SHM_DATA_STRUCTURES_CONTAINERS_MPSC_H_
#define HERMES_SHM_SHM_DATA_STRUCTURES_CONTAINERS_MPSC_H_

#include <vector>
#include "hermes_shm/types/qtok.h"

namespace hshm {

/** Forward declaration of mpsc_queue */
template<typename T>
class mpsc_queue;

/**
 * MACROS used to simplify the mpsc_queue namespace
 * Used as inputs to the SHM_CONTAINER_TEMPLATE
 * */
#define CLASS_NAME mpsc_queue
#define TYPED_CLASS mpsc_queue<T>
#define TYPED_HEADER ShmHeader<mpsc_queue<T>>

/**
 * A queue optimized for multiple producers (emplace) with a single
 * consumer (pop).
 * */
template<typename T>
class mpsc_queue {
 public:
  std::vector<std::pair<bitfield32_t, T>> queue_;
  std::atomic<_qtok_t> tail_;
  std::atomic<_qtok_t> head_;
  bitfield32_t flags_;

 public:
  /**====================================
   * Default Constructor
   * ===================================*/

  /** SHM constructor. Default. */
  explicit mpsc_queue(size_t depth = 1024) {
    queue_.resize(depth);
    flags_.Clear();
    SetNull();
  }

  /**====================================
   * Copy Constructors
   * ===================================*/

  /** SHM copy constructor */
  explicit mpsc_queue(const mpsc_queue &other) {
    head_ = other.head_.load();
    tail_ = other.tail_.load();
    queue_ = other.queue_;
  }

  /** SHM copy assignment operator */
  mpsc_queue& operator=(const mpsc_queue &other) {
    if (this != &other) {
      head_ = other.head_.load();
      tail_ = other.tail_.load();
      queue_ = other.queue_;
    }
    return *this;
  }

  /**====================================
   * Move Constructors
   * ===================================*/

  /** SHM move constructor. */
  mpsc_queue(mpsc_queue &&other) noexcept {
    head_ = other.head_.load();
    tail_ = other.tail_.load();
    queue_ = std::move(other.queue_);
    other.SetNull();
  }

  /** SHM move assignment operator. */
  mpsc_queue& operator=(mpsc_queue &&other) noexcept {
    if (this != &other) {
      head_ = other.head_.load();
      tail_ = other.tail_.load();
      queue_ = std::move(other.queue_);
      other.SetNull();
    }
    return *this;
  }

  /**====================================
   * Destructor
   * ===================================*/

  /** Check if the list is empty */
  bool IsNull() const {
    return queue_.IsNull();
  }

  /** Sets this list as empty */
  void SetNull() {
    head_ = 0;
    tail_ = 0;
  }

  /**====================================
   * MPSC Queue Methods
   * ===================================*/

  /** Construct an element at \a pos position in the list */
  template<typename ...Args>
  qtok_t emplace(Args&&... args) {
    // Allocate a slot in the queue
    // The slot is marked NULL, so pop won't do anything if context switch
    _qtok_t head = head_.load();
    _qtok_t tail = tail_.fetch_add(1);
    size_t size = tail - head + 1;

    // Check if there's space in the queue.
    if (size > queue_.size()) {
      while (true) {
        head = head_.load();
        size = tail - head + 1;
        if (size <= queue_.size()) {
          break;
        }
        HERMES_THREAD_MODEL->Yield();
      }
    }

    // Emplace into queue at our slot
    uint32_t idx = tail % queue_.size();
    auto iter = queue_.begin() + idx;
    hipc::Allocator::DestructObj(*iter);
    hipc::Allocator::ConstructObj(
        iter->second, std::forward<Args>(args)...);

    // Let pop know that the data is fully prepared
    std::pair<bitfield32_t, T> &entry = (*iter);
    entry.first.SetBits(1);
    return qtok_t(tail);
  }

 public:
  /** Consumer pops the head object */
  qtok_t pop(T &val) {
    // Don't pop if there's no entries
    _qtok_t head = head_.load();
    _qtok_t tail = tail_.load();
    if (head >= tail) {
      return qtok_t::GetNull();
    }

    // Pop the element, but only if it's marked valid
    _qtok_t idx = head % queue_.size();
    std::pair<bitfield32_t, T> &entry = queue_[idx];
    if (entry.first.Any(1)) {
      val = std::move(entry.second);
      entry.first.Clear();
      head_.fetch_add(1);
      return qtok_t(head);
    } else {
      return qtok_t::GetNull();
    }
  }

  /** Consumer pops the head object */
  qtok_t pop() {
    // Don't pop if there's no entries
    _qtok_t head = head_.load();
    _qtok_t tail = tail_.load();
    if (head >= tail) {
      return qtok_t::GetNull();
    }

    // Pop the element, but only if it's marked valid
    _qtok_t idx = head % queue_.size();
    std::pair<bitfield32_t, T> &entry = queue_[idx];
    if (entry.first.Any(1)) {
      entry.first.Clear();
      head_.fetch_add(1);
      return qtok_t(head);
    } else {
      return qtok_t::GetNull();
    }
  }

  /** Consumer peeks an object */
  qtok_t peek(T *&val, int off = 0) {
    // Don't pop if there's no entries
    _qtok_t head = head_.load() + off;
    _qtok_t tail = tail_.load();
    if (head >= tail) {
      return qtok_t::GetNull();
    }

    // Pop the element, but only if it's marked valid
    _qtok_t idx = (head) % queue_.size();
    std::pair<bitfield32_t, T> &entry = queue_[idx];
    if (entry.first.Any(1)) {
      val = &entry.second;
      return qtok_t(head);
    } else {
      return qtok_t::GetNull();
    }
  }

  /** Consumer peeks an object */
  qtok_t peek(std::pair<bitfield32_t, T> *&val, int off = 0) {
    // Don't pop if there's no entries
    _qtok_t head = head_.load() + off;
    _qtok_t tail = tail_.load();
    if (head >= tail) {
      return qtok_t::GetNull();
    }

    // Pop the element, but only if it's marked valid
    _qtok_t idx = (head) % queue_.size();
    std::pair<bitfield32_t, T> &entry = queue_[idx];
    if (entry.first.Any(1)) {
      val = &entry;
      return qtok_t(head);
    } else {
      return qtok_t::GetNull();
    }
  }

  /** Get size at this moment */
  size_t GetSize() {
    size_t tail = tail_.load();
    size_t head = head_.load();
    if (tail < head) {
      return 0;
    }
    return tail - head;
  }
};

}  // namespace hshm::ipc

#undef CLASS_NAME
#undef TYPED_CLASS
#undef TYPED_HEADER

#endif  // HERMES_SHM_SHM_DATA_STRUCTURES_CONTAINERS_MPSC_H_
