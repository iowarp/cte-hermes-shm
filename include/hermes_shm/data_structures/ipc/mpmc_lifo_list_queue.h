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

#ifndef HERMES_DATA_STRUCTURES__MPMC_LIST_lifo_list_queue_H
#define HERMES_DATA_STRUCTURES__MPMC_LIST_lifo_list_queue_H

#include "hermes_shm/memory/memory.h"
#include "lifo_list_queue.h"

namespace hshm::ipc {

/** forward pointer for mpmc_lifo_list_queue */
template <typename T, HSHM_CLASS_TEMPL_WITH_DEFAULTS>
class mpmc_lifo_list_queue;

/**
 * MACROS used to simplify the mpmc_lifo_list_queue namespace
 * Used as inputs to the HIPC_CONTAINER_TEMPLATE
 * */
#define CLASS_NAME mpmc_lifo_list_queue
#define CLASS_NEW_ARGS T

/**
 * A singly-linked lock-free queue implementation
 * */
template <typename T, HSHM_CLASS_TEMPL>
class mpmc_lifo_list_queue : public ShmContainer {
 public:
  HIPC_CONTAINER_TEMPLATE((CLASS_NAME), (CLASS_NEW_ARGS))
  AtomicOffsetPointer tail_shm_;
  hipc::atomic<size_t> count_;

 public:
  /**====================================
   * Default Constructor
   * ===================================*/

  /** Constructor. Default. */
  HSHM_CROSS_FUN
  mpmc_lifo_list_queue() {
    shm_init(HERMES_MEMORY_MANAGER->GetDefaultAllocator<AllocT>());
  }

  /** Constructor. Int */
  HSHM_CROSS_FUN
  explicit mpmc_lifo_list_queue(size_t depth) {
    shm_init(HERMES_MEMORY_MANAGER->GetDefaultAllocator<AllocT>());
  }

  /** SHM constructor. Default. */
  HSHM_CROSS_FUN
  explicit mpmc_lifo_list_queue(const hipc::CtxAllocator<AllocT> &alloc) {
    shm_init(alloc);
  }

  /** SHM constructor. Int */
  HSHM_CROSS_FUN
  mpmc_lifo_list_queue(const hipc::CtxAllocator<AllocT> &alloc, size_t depth) {
    shm_init(alloc);
  }

  /** SHM constructor. Default. */
  HSHM_CROSS_FUN
  void shm_init(const hipc::CtxAllocator<AllocT> &alloc) {
    init_shm_container(alloc);
    tail_shm_.SetNull();
    count_ = 0;
  }

  /**====================================
   * Copy Constructors
   * ===================================*/

  /** Copy constructor */
  HSHM_CROSS_FUN
  explicit mpmc_lifo_list_queue(const mpmc_lifo_list_queue &other) {
    init_shm_container(HERMES_MEMORY_MANAGER->GetDefaultAllocator<AllocT>());
    shm_strong_copy_op(other);
  }

  /** SHM copy constructor */
  HSHM_CROSS_FUN
  explicit mpmc_lifo_list_queue(const hipc::CtxAllocator<AllocT> &alloc,
                                const mpmc_lifo_list_queue &other) {
    init_shm_container(alloc);
    shm_strong_copy_op(other);
  }

  /** SHM copy assignment operator */
  HSHM_CROSS_FUN
  mpmc_lifo_list_queue &operator=(const mpmc_lifo_list_queue &other) {
    if (this != &other) {
      shm_destroy();
      shm_strong_copy_op(other);
    }
    return *this;
  }

  /** SHM copy constructor + operator */
  HSHM_CROSS_FUN
  void shm_strong_copy_op(const mpmc_lifo_list_queue &other) {
    memcpy((void *)this, (void *)&other, sizeof(*this));
  }

  /**====================================
   * Move Constructors
   * ===================================*/

  /** Move constructor. */
  HSHM_CROSS_FUN
  mpmc_lifo_list_queue(mpmc_lifo_list_queue &&other) noexcept {
    init_shm_container(other.GetAllocator());
    memcpy((void *)this, (void *)&other, sizeof(*this));
    other.SetNull();
  }

  /** SHM move constructor. */
  HSHM_CROSS_FUN
  mpmc_lifo_list_queue(const hipc::CtxAllocator<AllocT> &alloc,
                       mpmc_lifo_list_queue &&other) noexcept {
    init_shm_container(alloc);
    if (GetAllocator() == other.GetAllocator()) {
      memcpy((void *)this, (void *)&other, sizeof(*this));
      other.SetNull();
    } else {
      shm_strong_copy_op(other);
      other.shm_destroy();
    }
  }

  /** SHM move assignment operator. */
  HSHM_CROSS_FUN
  mpmc_lifo_list_queue &operator=(mpmc_lifo_list_queue &&other) noexcept {
    if (this != &other) {
      shm_destroy();
      if (this != &other) {
        memcpy((void *)this, (void *)&other, sizeof(*this));
        other.SetNull();
      } else {
        shm_strong_copy_op(other);
        other.shm_destroy();
      }
    }
    return *this;
  }

  /**====================================
   * Destructor
   * ===================================*/

  /** Check if the mpmc_lifo_list_queue is null */
  HSHM_CROSS_FUN
  bool IsNull() { return false; }

  /** Set the mpmc_lifo_list_queue to null */
  HSHM_CROSS_FUN
  void SetNull() {}

  /** SHM destructor. */
  HSHM_CROSS_FUN
  void shm_destroy_main() { clear(); }

  /**====================================
   * mpmc_lifo_list_queue Methods
   * ===================================*/

  /** Construct an element at \a pos position in the mpmc_lifo_list_queue */
  HSHM_CROSS_FUN
  qtok_t enqueue(const LPointer<T> &entry) {
    bool ret;
    do {
      size_t tail_shm = tail_shm_.load();
      entry.ptr_->next_shm_ = tail_shm;
      ret = tail_shm_.compare_exchange_weak(tail_shm, entry.shm_.off_.load());
    } while (!ret);
    ++count_;
    return qtok_t(1);
  }

  /** Construct an element at \a pos position in the mpmc_lifo_list_queue */
  HSHM_CROSS_FUN
  qtok_t enqueue(T *entry) {
    FullPtr<T> entry_ptr(GetAllocator(), entry);
    return enqueue(entry_ptr);
  }

  /** Emplace. wrapper for enqueue */
  HSHM_CROSS_FUN
  qtok_t emplace(T *entry) { return enqueue(entry); }

  /** Push. wrapper for enqueue */
  HSHM_INLINE_CROSS_FUN
  qtok_t push(T *entry) { return enqueue(entry); }

  /** Dequeue the element */
  HSHM_INLINE_CROSS_FUN
  T *dequeue() {
    T *val;
    if (dequeue(val).IsNull()) {
      return nullptr;
    }
    return val;
  }

  /** Pop the element */
  HSHM_INLINE_CROSS_FUN
  T *pop() { return dequeue(); }

  /** Dequeue the element (FullPtr, qtok_t) */
  HSHM_CROSS_FUN
  qtok_t dequeue(FullPtr<T> &val) {
    size_t cur_size = size();
    if (cur_size == 0) {
      return qtok_t::GetNull();
    }
    bool ret;
    do {
      OffsetPointer tail_shm(tail_shm_.load());
      val.shm_.off_ = tail_shm_.load();
      val.shm_.alloc_id_ = GetAllocator()->GetId();
      val.ptr_ = GetAllocator()->template Convert<T>(tail_shm);
      if (val.IsNull()) {
        return qtok_t::GetNull();
      }
      auto next_tail = val->next_shm_;
      ret = tail_shm_.compare_exchange_weak(tail_shm.off_.ref(),
                                            next_tail.off_.ref());
    } while (!ret);
    --count_;
    return qtok_t(1);
  }

  /** Pop the element (FullPtr, qtok_t) */
  HSHM_INLINE_CROSS_FUN
  qtok_t pop(FullPtr<T> &val) { return dequeue(val); }

  /** Dequeue the element (qtok_t) */
  HSHM_CROSS_FUN
  qtok_t dequeue(T *&val) {
    FullPtr<T> entry;
    qtok_t ret = dequeue(entry);
    val = entry.ptr_;
    return ret;
  }

  /** Pop the element (qtok_t) */
  HSHM_INLINE_CROSS_FUN
  qtok_t pop(T *&val) { return dequeue(val); }

  /** Peek the first element of the queue */
  HSHM_CROSS_FUN
  T *peek() {
    if (size() == 0) {
      return nullptr;
    }
    auto entry = GetAllocator()->template Convert<list_queue_entry>(tail_shm_);
    return reinterpret_cast<T *>(entry);
  }

  /** Destroy all elements in the mpmc_lifo_list_queue */
  HSHM_CROSS_FUN
  void clear() {
    while (size()) {
      dequeue();
    }
  }

  /** Get the number of elements in the mpmc_lifo_list_queue */
  HSHM_CROSS_FUN
  size_t size() const { return count_.load(); }
};

}  // namespace hshm::ipc

namespace hshm {

template <typename T, HSHM_CLASS_TEMPL_WITH_PRIV_DEFAULTS>
using mpmc_lifo_list_queue =
    hshm::ipc::mpmc_lifo_list_queue<T, HSHM_CLASS_TEMPL_ARGS>;

}  // namespace hshm

#undef CLASS_NAME
#undef CLASS_NEW_ARGS

#endif  // HERMES_DATA_STRUCTURES__MPMC_LIST_lifo_list_queue_H
