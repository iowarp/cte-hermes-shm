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

#ifndef HSHM_DATA_STRUCTURES__MPMC_LIST_lifo_list_queue_H
#define HSHM_DATA_STRUCTURES__MPMC_LIST_lifo_list_queue_H

#include "hermes_shm/memory/memory.h"
#include "hermes_shm/util/logging.h"
#include "lifo_list_queue.h"

namespace hshm::ipc {

/** forward pointer for mpsc_lifo_list_queue */
template <typename T, HSHM_CLASS_TEMPL_WITH_DEFAULTS>
class mpsc_lifo_list_queue;

/**
 * MACROS used to simplify the mpsc_lifo_list_queue namespace
 * Used as inputs to the HIPC_CONTAINER_TEMPLATE
 * */
#define CLASS_NAME mpsc_lifo_list_queue
#define CLASS_NEW_ARGS T

/**
 * A singly-linked lock-free queue implementation
 * */
template <typename T, HSHM_CLASS_TEMPL>
class mpsc_lifo_list_queue : public ShmContainer {
 public:
  HIPC_CONTAINER_TEMPLATE((CLASS_NAME), (CLASS_NEW_ARGS))
  AtomicOffsetPointer tail_shm_;
  hipc::atomic<hshm::size_t> count_;

 public:
  /**====================================
   * Default Constructor
   * ===================================*/

  /** Constructor. Default. */
  HSHM_CROSS_FUN
  mpsc_lifo_list_queue() {
    shm_init(HSHM_MEMORY_MANAGER->GetDefaultAllocator<AllocT>());
  }

  /** Constructor. Int */
  HSHM_CROSS_FUN
  explicit mpsc_lifo_list_queue(size_t depth) {
    shm_init(HSHM_MEMORY_MANAGER->GetDefaultAllocator<AllocT>());
  }

  /** SHM constructor. Default. */
  HSHM_CROSS_FUN
  explicit mpsc_lifo_list_queue(const hipc::CtxAllocator<AllocT> &alloc) {
    shm_init(alloc);
  }

  /** SHM constructor. Int */
  HSHM_CROSS_FUN
  mpsc_lifo_list_queue(const hipc::CtxAllocator<AllocT> &alloc, size_t depth) {
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
  explicit mpsc_lifo_list_queue(const mpsc_lifo_list_queue &other) {
    init_shm_container(HSHM_MEMORY_MANAGER->GetDefaultAllocator<AllocT>());
    shm_strong_copy_op(other);
  }

  /** SHM copy constructor */
  HSHM_CROSS_FUN
  explicit mpsc_lifo_list_queue(const hipc::CtxAllocator<AllocT> &alloc,
                                const mpsc_lifo_list_queue &other) {
    init_shm_container(alloc);
    shm_strong_copy_op(other);
  }

  /** SHM copy assignment operator */
  HSHM_CROSS_FUN
  mpsc_lifo_list_queue &operator=(const mpsc_lifo_list_queue &other) {
    if (this != &other) {
      shm_destroy();
      shm_strong_copy_op(other);
    }
    return *this;
  }

  /** SHM copy constructor + operator */
  HSHM_CROSS_FUN
  void shm_strong_copy_op(const mpsc_lifo_list_queue &other) {
    memcpy((void *)this, (void *)&other, sizeof(*this));
  }

  /**====================================
   * Move Constructors
   * ===================================*/

  /** Move constructor. */
  HSHM_CROSS_FUN
  mpsc_lifo_list_queue(mpsc_lifo_list_queue &&other) noexcept {
    init_shm_container(other.GetAllocator());
    memcpy((void *)this, (void *)&other, sizeof(*this));
    other.SetNull();
  }

  /** SHM move constructor. */
  HSHM_CROSS_FUN
  mpsc_lifo_list_queue(const hipc::CtxAllocator<AllocT> &alloc,
                       mpsc_lifo_list_queue &&other) noexcept {
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
  mpsc_lifo_list_queue &operator=(mpsc_lifo_list_queue &&other) noexcept {
    if (this != &other) {
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

  /** Check if the mpsc_lifo_list_queue is null */
  HSHM_CROSS_FUN
  bool IsNull() { return false; }

  /** Set the mpsc_lifo_list_queue to null */
  HSHM_CROSS_FUN
  void SetNull() {}

  /** SHM destructor. */
  HSHM_CROSS_FUN
  void shm_destroy_main() { clear(); }

  /**====================================
   * mpsc_lifo_list_queue Methods
   * ===================================*/

  /** Construct an element at \a pos position in the mpsc_lifo_list_queue */
  HSHM_CROSS_FUN
  qtok_t enqueue(const FullPtr<T> &entry) {
    bool ret;
    do {
      size_t tail_shm = tail_shm_.load();
      entry->next_shm_ = tail_shm;
      ret = tail_shm_.compare_exchange_weak(tail_shm, entry.shm_.off_.load());
    } while (!ret);
    ++count_;
    return qtok_t(1);
  }

  /** Emplace. wrapper for enqueue */
  HSHM_INLINE_CROSS_FUN
  qtok_t emplace(const FullPtr<T> &entry) { return enqueue(entry); }

  /** Push. wrapper for enqueue */
  HSHM_INLINE_CROSS_FUN
  qtok_t push(const FullPtr<T> &entry) { return enqueue(entry); }

  /** Construct an element at \a pos position in the mpsc_lifo_list_queue */
  HSHM_INLINE_CROSS_FUN
  qtok_t enqueue(T *entry) {
    FullPtr<T> entry_ptr(GetAllocator(), entry);
    return enqueue(entry_ptr);
  }

  /** Emplace. wrapper for enqueue */
  HSHM_INLINE_CROSS_FUN
  qtok_t emplace(T *entry) { return enqueue(entry); }

  /** Push. wrapper for enqueue */
  HSHM_INLINE_CROSS_FUN
  qtok_t push(T *entry) { return enqueue(entry); }

  /** Dequeue the element (FullPtr, qtok_t) */
  HSHM_CROSS_FUN
  qtok_t dequeue(FullPtr<T> &val) {
    size_t cur_size = size();
    if (cur_size == 0) {
      return qtok_t::GetNull();
    }
    bool ret;
    auto *alloc = GetAllocator();
    do {
      OffsetPointer tail_shm(tail_shm_.load());
      if (tail_shm.IsNull()) {
        return qtok_t::GetNull();
      }
      val.shm_.off_ = tail_shm.load();
      val.shm_.alloc_id_ = alloc->GetId();
      val.ptr_ = alloc->template Convert<T>(tail_shm);
      hshm::size_t next_tail = val->next_shm_.load();
      ret = tail_shm_.compare_exchange_weak(tail_shm.off_.ref(), next_tail);
    } while (!ret);
    --count_;
    return qtok_t(1);
  }

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

  /** Destroy all elements in the mpsc_lifo_list_queue */
  HSHM_CROSS_FUN
  void clear() {
    while (size()) {
      dequeue();
    }
  }

  /** Get the number of elements in the mpsc_lifo_list_queue */
  HSHM_CROSS_FUN
  size_t size() const { return count_.load(); }
};

}  // namespace hshm::ipc

namespace hshm {

template <typename T, HSHM_CLASS_TEMPL_WITH_PRIV_DEFAULTS>
using mpsc_lifo_list_queue =
    hshm::ipc::mpsc_lifo_list_queue<T, HSHM_CLASS_TEMPL_ARGS>;

}  // namespace hshm

#undef CLASS_NAME
#undef CLASS_NEW_ARGS

#endif  // HSHM_DATA_STRUCTURES__MPMC_LIST_lifo_list_queue_H
