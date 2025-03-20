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

#ifndef HSHM_DATA_STRUCTURES__SPSC_LIST_lifo_list_queue_H
#define HSHM_DATA_STRUCTURES__SPSC_LIST_lifo_list_queue_H

#include "hermes_shm/memory/memory.h"
#include "lifo_list_queue.h"

namespace hshm::ipc {

/** forward pointer for spsc_fifo_list_queue */
template <typename T, HSHM_CLASS_TEMPL_WITH_DEFAULTS>
class spsc_fifo_list_queue;

/**
 * MACROS used to simplify the spsc_fifo_list_queue namespace
 * Used as inputs to the HIPC_CONTAINER_TEMPLATE
 * */
#define CLASS_NAME spsc_fifo_list_queue
#define CLASS_NEW_ARGS T

/**
 * A singly-linked lock-free queue implementation
 * */
template <typename T, HSHM_CLASS_TEMPL>
class spsc_fifo_list_queue : public ShmContainer {
 public:
  HIPC_CONTAINER_TEMPLATE((CLASS_NAME), (CLASS_NEW_ARGS))
  OffsetPointer head_shm_, tail_shm_;
  hipc::nonatomic<size_t> head_, tail_;

 public:
  /**====================================
   * Default Constructor
   * ===================================*/

  /** Constructor. Default. */
  HSHM_CROSS_FUN
  spsc_fifo_list_queue() {
    shm_init(HSHM_MEMORY_MANAGER->GetDefaultAllocator<AllocT>());
  }

  /** Constructor. Int */
  HSHM_CROSS_FUN
  explicit spsc_fifo_list_queue(size_t depth) {
    shm_init(HSHM_MEMORY_MANAGER->GetDefaultAllocator<AllocT>());
  }

  /** SHM constructor. Default. */
  HSHM_CROSS_FUN
  explicit spsc_fifo_list_queue(const hipc::CtxAllocator<AllocT> &alloc) {
    shm_init(alloc);
  }

  /** SHM constructor. Int */
  HSHM_CROSS_FUN
  spsc_fifo_list_queue(const hipc::CtxAllocator<AllocT> &alloc, size_t depth) {
    shm_init(alloc);
  }

  /** SHM constructor. Default. */
  HSHM_CROSS_FUN
  void shm_init(const hipc::CtxAllocator<AllocT> &alloc) {
    init_shm_container(alloc);
    head_ = 0;
    tail_ = 0;
    head_shm_.SetNull();
    tail_shm_.SetNull();
  }

  /**====================================
   * Copy Constructors
   * ===================================*/

  /** Copy constructor */
  HSHM_CROSS_FUN
  explicit spsc_fifo_list_queue(const spsc_fifo_list_queue &other) {
    init_shm_container(HSHM_MEMORY_MANAGER->GetDefaultAllocator<AllocT>());
    shm_strong_copy_op(other);
  }

  /** SHM copy constructor */
  HSHM_CROSS_FUN
  explicit spsc_fifo_list_queue(const hipc::CtxAllocator<AllocT> &alloc,
                                const spsc_fifo_list_queue &other) {
    init_shm_container(alloc);
    shm_strong_copy_op(other);
  }

  /** SHM copy assignment operator */
  HSHM_CROSS_FUN
  spsc_fifo_list_queue &operator=(const spsc_fifo_list_queue &other) {
    if (this != &other) {
      shm_destroy();
      shm_strong_copy_op(other);
    }
    return *this;
  }

  /** SHM copy constructor + operator */
  HSHM_CROSS_FUN
  void shm_strong_copy_op(const spsc_fifo_list_queue &other) {
    memcpy((void *)this, (void *)&other, sizeof(*this));
  }

  /**====================================
   * Move Constructors
   * ===================================*/

  /** Move constructor. */
  HSHM_CROSS_FUN
  spsc_fifo_list_queue(spsc_fifo_list_queue &&other) noexcept {
    init_shm_container(other.GetAllocator());
    memcpy((void *)this, (void *)&other, sizeof(*this));
    other.SetNull();
  }

  /** SHM move constructor. */
  HSHM_CROSS_FUN
  spsc_fifo_list_queue(const hipc::CtxAllocator<AllocT> &alloc,
                       spsc_fifo_list_queue &&other) noexcept {
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
  spsc_fifo_list_queue &operator=(spsc_fifo_list_queue &&other) noexcept {
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

  /** Check if the spsc_fifo_list_queue is null */
  HSHM_CROSS_FUN
  bool IsNull() { return false; }

  /** Set the spsc_fifo_list_queue to null */
  HSHM_CROSS_FUN
  void SetNull() {}

  /** SHM destructor. */
  HSHM_CROSS_FUN
  void shm_destroy_main() { clear(); }

  /**====================================
   * spsc_fifo_list_queue Methods
   * ===================================*/

  /** Construct an element (FullPtr) */
  HSHM_CROSS_FUN
  qtok_t enqueue(const FullPtr<T> &entry) {
    entry->next_shm_ = OffsetPointer::GetNull();
    size_t tail_id = tail_.fetch_add(1);
    if (!head_shm_.IsNull()) {
      auto tail = GetAllocator()->template Convert<T>(tail_shm_);
      tail->next_shm_ = entry.shm_.off_;
    } else {
      head_shm_ = entry.shm_.off_;
    }
    tail_shm_ = entry.shm_.off_;
    return qtok_t(tail_id);
  }

  /** Emplace. wrapper for enqueue (FullPtr) */
  HSHM_CROSS_FUN
  qtok_t emplace(const FullPtr<T> &entry) { return enqueue(entry); }

  /** Push. wrapper for enqueue (FullPtr) */
  HSHM_INLINE_CROSS_FUN
  qtok_t push(const FullPtr<T> &entry) { return enqueue(entry); }

  /** Construct an element at \a pos position in the spsc_fifo_list_queue */
  HSHM_CROSS_FUN
  qtok_t enqueue(T *entry) {
    FullPtr<T> ptr(GetAllocator(), entry);
    return enqueue(ptr);
  }

  /** Emplace. wrapper for enqueue */
  HSHM_CROSS_FUN
  qtok_t emplace(T *entry) { return enqueue(entry); }

  /** Push. wrapper for enqueue */
  HSHM_INLINE_CROSS_FUN
  qtok_t push(T *entry) { return enqueue(entry); }

  /** Dequeue the element (qtok_t) */
  HSHM_INLINE_CROSS_FUN
  T *dequeue() {
    T *val;
    if (dequeue(val).IsNull()) {
      return nullptr;
    }
    return val;
  }

  /** Pop the element (qtok_t) */
  HSHM_INLINE_CROSS_FUN
  T *pop() { return dequeue(); }

  /** Dequeue the element (qtok_t) */
  HSHM_INLINE_CROSS_FUN
  qtok_t dequeue(FullPtr<T> &val) {
    size_t cur_size = size();
    if (cur_size == 0) {
      return qtok_t::GetNull();
    }
    val = FullPtr<T>(GetAllocator(), head_shm_);
    if (val.ptr_->next_shm_.IsNull() && cur_size > 1) {
      return qtok_t::GetNull();
    }
    head_shm_ = val.ptr_->next_shm_;
    size_t head_id = head_.fetch_add(1);
    return qtok_t(head_id);
  }

  /** Pop the element (qtok_t) */
  HSHM_INLINE_CROSS_FUN
  qtok_t pop(FullPtr<T> &val) { return dequeue(val); }

  /** Dequeue the element (qtok_t) */
  HSHM_INLINE_CROSS_FUN
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
    auto entry = GetAllocator()->template Convert<list_queue_entry>(head_shm_);
    return reinterpret_cast<T *>(entry);
  }

  /** Destroy all elements in the spsc_fifo_list_queue */
  HSHM_CROSS_FUN
  void clear() {
    while (size()) {
      dequeue();
    }
  }

  /** Get the number of elements in the spsc_fifo_list_queue */
  HSHM_CROSS_FUN
  size_t size() const {
    size_t head = head_.load();
    size_t tail = tail_.load();
    if (tail < head) {
      return 0;
    }
    return tail - head;
  }
};

}  // namespace hshm::ipc

namespace hshm {

template <typename T, HSHM_CLASS_TEMPL_WITH_PRIV_DEFAULTS>
using spsc_fifo_list_queue =
    hshm::ipc::spsc_fifo_list_queue<T, HSHM_CLASS_TEMPL_ARGS>;

}  // namespace hshm

#undef CLASS_NAME
#undef CLASS_NEW_ARGS

#endif  // HSHM_DATA_STRUCTURES__SPSC_LIST_lifo_list_queue_H
