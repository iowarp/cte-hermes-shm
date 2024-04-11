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
#include "hermes_shm/data_structures/ipc/mpsc_queue.h"

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
  hipc::mptr<hipc::mpsc_queue<T>> queue_;

 public:
  /**====================================
   * Default Constructor
   * ===================================*/

  /** SHM constructor. Default. */
  explicit mpsc_queue(size_t depth = 1024) {
    queue_ = hipc::make_mptr<hipc::mpsc_queue<T>>(depth);
  }

  /**====================================
   * Copy Constructors
   * ===================================*/

  /** SHM copy constructor */
  explicit mpsc_queue(const mpsc_queue &other) {
    queue_ = other.queue_;
  }

  /** SHM copy assignment operator */
  mpsc_queue& operator=(const mpsc_queue &other) {
    if (this != &other) {
      queue_ = other.queue_;
    }
    return *this;
  }

  /**====================================
   * Move Constructors
   * ===================================*/

  /** SHM move constructor. */
  mpsc_queue(mpsc_queue &&other) noexcept {
    queue_ = std::move(other.queue_);
    other.SetNull();
  }

  /** SHM move assignment operator. */
  mpsc_queue& operator=(mpsc_queue &&other) noexcept {
    if (this != &other) {
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
    return queue_->IsNull();
  }

  /** Sets this list as empty */
  void SetNull() {
    queue_->SetNull();
  }

  /**====================================
   * MPSC Queue Methods
   * ===================================*/

  /** Construct an element at \a pos position in the list */
  template<typename ...Args>
  qtok_t emplace(Args&&... args) {
    return queue_->emplace(std::forward<Args>(args)...);
  }

 public:
  /** Consumer pops the head object */
  qtok_t pop(T &val) {
    return queue_->pop(val);
  }

  /** Consumer pops the head object */
  qtok_t pop() {
    return queue_->pop();
  }

  /** Consumer peeks an object */
  qtok_t peek(T *&val, int off = 0) {
    return queue_->peek(val, off);
  }

  /** Consumer peeks an object */
  qtok_t peek(std::pair<bitfield32_t, T> *&val, int off = 0) {
    return queue_->peek(val, off);
  }

  /** Get size at this moment */
  size_t GetSize() {
    return queue_->GetSize();
  }
};

}  // namespace hshm

#undef CLASS_NAME
#undef TYPED_CLASS
#undef TYPED_HEADER

#endif  // HERMES_SHM_SHM_DATA_STRUCTURES_CONTAINERS_MPSC_H_
