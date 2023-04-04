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

#ifndef HERMES_SHM_INCLUDE_HERMES_SHM_DATA_STRUCTURES_IPC_mpsc_queue_H_
#define HERMES_SHM_INCLUDE_HERMES_SHM_DATA_STRUCTURES_IPC_mpsc_queue_H_

#include "hermes_shm/data_structures/ipc/internal/shm_internal.h"
#include "hermes_shm/thread/lock.h"
#include "vector.h"
#include "pair.h"

namespace hshm::ipc {

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
 * The mpsc_queue shared-memory header
 * */
template<typename T>
struct ShmHeader<mpsc_queue<T>> {
  SHM_CONTAINER_HEADER_TEMPLATE(ShmHeader)
  ShmArchive<vector<pair<bitfield32_t, T>>> queue_;
  std::atomic<uint64_t> tail_;
  std::atomic<uint64_t> head_;
  RwLock lock_;

  /** Strong copy operation */
  void strong_copy(const ShmHeader &other) {
    head_ = other.head_.load();
    tail_ = other.tail_.load();
  }
};

/**
 * A queue optimized for multiple producers (emplace) with a single
 * consumer (pop).
 * */
template<typename T>
class mpsc_queue : public ShmContainer {
 public:
  SHM_CONTAINER_TEMPLATE((CLASS_NAME), (TYPED_CLASS), (TYPED_HEADER))
  Ref<vector<pair<bitfield32_t, T>>> queue_;

 public:
  /**====================================
   * Default Constructor
   * ===================================*/

  /** SHM constructor. Default. */
  explicit mpsc_queue(TYPED_HEADER *header, Allocator *alloc,
                      size_t depth = 1024) {
    shm_init_header(header, alloc);
    queue_ = make_ref<vector<pair<bitfield32_t, T>>>(header_->queue_,
                                                     alloc_,
                                                     depth);
    SetNull();
  }

  /**====================================
   * Copy Constructors
   * ===================================*/

  /** SHM copy constructor */
  explicit mpsc_queue(TYPED_HEADER *header, Allocator *alloc,
                     const mpsc_queue &other) {
    shm_init_header(header, alloc);
    SetNull();
    shm_strong_copy_construct_and_op(other);
  }

  /** SHM copy assignment operator */
  mpsc_queue& operator=(const mpsc_queue &other) {
    if (this != &other) {
      shm_destroy();
      shm_strong_copy_construct_and_op(other);
    }
    return *this;
  }

  /** SHM copy constructor + operator main */
  void shm_strong_copy_construct_and_op(const mpsc_queue &other) {
    (*header_) = *(other.header_);
    (*queue_) = (*other.queue_);
  }

  /**====================================
   * Move Constructors
   * ===================================*/

  /** SHM move constructor. */
  mpsc_queue(TYPED_HEADER *header, Allocator *alloc,
            mpsc_queue &&other) noexcept {
    shm_init_header(header, alloc);
    if (alloc_ == other.alloc_) {
      (*header_) = std::move(*other.header_);
      (*queue_) = std::move(*other.queue_);
      other.SetNull();
    } else {
      shm_strong_copy_construct_and_op(other);
      other.shm_destroy();
    }
  }

  /** SHM move assignment operator. */
  mpsc_queue& operator=(mpsc_queue &&other) noexcept {
    if (this != &other) {
      shm_destroy();
      if (alloc_ == other.alloc_) {
        (*header_) = std::move(*other.header_);
        (*queue_) = std::move(*other.queue_);
        other.SetNull();
      } else {
        shm_strong_copy_construct_and_op(other);
        other.shm_destroy();
      }
    }
    return *this;
  }

  /**====================================
   * Destructor
   * ===================================*/

  /** SHM destructor.  */
  void shm_destroy_main() {
    queue_->shm_destroy();
  }

  /** Check if the list is empty */
  bool IsNull() const {
    return queue_->IsNull();
  }

  /** Sets this list as empty */
  void SetNull() {
    queue_->SetNull();
  }

  /**====================================
   * SHM Deserialization
   * ===================================*/

  /** Load from shared memory */
  void shm_deserialize_main() {
    queue_ = Ref<vector<pair<bitfield32_t, T>>>(header_->queue_,
                                                alloc_);
  }

  /**====================================
   * MPSC Queue Methods
   * ===================================*/

  /** Construct an element at \a pos position in the list */
  template<typename ...Args>
  void emplace(Args&&... args) {
    // Allocate a slot in the queue
    // The slot is marked NULL, so pop won't do anything if context switch
    uint64_t head = header_->head_.load();
    uint64_t tail = header_->tail_.fetch_add(1);
    uint64_t size = tail - head;

    // Check if there's space in the queue. Resize if necessary.
    if (size > queue_->size()) {
      ScopedRwWriteLock resize_lock(header_->lock_);
      if (size > queue_->size()) {
        queue_->resize(size + 64);
      }
    }

    // Emplace into queue at our slot
    ScopedRwReadLock resize_lock(header_->lock_);
    uint32_t idx = tail % queue_->size();
    auto iter = queue_->begin() + idx;
    queue_->emplace(iter,
                    hshm::PiecewiseConstruct(),
                    make_argpack(),
                    make_argpack(std::forward<Args>(args)...));

    // Let pop know that the data is fully prepared
    (*queue_)[idx]->first_->SetBits(1);
  }

  /** Consumer pops the head object */
  bool pop(Ref<T> &val) {
    ScopedRwReadLock resize_lock(header_->lock_);

    // Don't pop if there's no entries
    uint64_t head = header_->head_.load();
    uint64_t tail = header_->tail_.load();
    uint64_t size = tail - head;
    if (size == 0) {
      return false;
    }

    // Pop the element, but only if it's marked valid
    uint64_t idx = head % queue_->size();
    hipc::Ref<std::pair<bitfield32_t, T>> entry = queue_[idx];
    if (entry->first_->Any(1)) {
      (*val) = std::move(*entry->second_);
      entry->first_->Clear();
      header_->head_.fetch_add(1);
      return true;
    } else {
      return false;
    }
  }
};

}  // namespace hshm::ipc

#undef CLASS_NAME
#undef TYPED_CLASS
#undef TYPED_HEADER

#endif  // HERMES_SHM_INCLUDE_HERMES_SHM_DATA_STRUCTURES_IPC_mpsc_queue_H_
