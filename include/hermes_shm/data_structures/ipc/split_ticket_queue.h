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

#ifndef HSHM_SHM__DATA_STRUCTURES_IPC_SPLIT_TICKET_QUEUE_H_
#define HSHM_SHM__DATA_STRUCTURES_IPC_SPLIT_TICKET_QUEUE_H_

#include "hermes_shm/data_structures/internal/shm_internal.h"
#include "hermes_shm/thread/lock.h"
#include "ticket_queue.h"
#include "vector.h"

namespace hshm::ipc {

/** Forward declaration of split_ticket_queue */
template <typename T, HSHM_CLASS_TEMPL_WITH_DEFAULTS>
class split_ticket_queue;

/**
 * MACROS used to simplify the split_ticket_queue namespace
 * Used as inputs to the HIPC_CONTAINER_TEMPLATE
 * */
#define CLASS_NAME split_ticket_queue
#define CLASS_NEW_ARGS T

/**
 * A MPMC queue for allocating tickets. Handles concurrency
 * without blocking.
 * */
template <typename T, HSHM_CLASS_TEMPL>
class split_ticket_queue : public ShmContainer {
 public:
  HIPC_CONTAINER_TEMPLATE((CLASS_NAME), (CLASS_NEW_ARGS))
  typedef ticket_queue<T, HSHM_CLASS_TEMPL_ARGS> ticket_queue_t;
  typedef vector<ticket_queue_t, HSHM_CLASS_TEMPL_ARGS> vector_t;
  delay_ar<vector_t> splits_;
  hipc::atomic<hshm::min_i32> rr_tail_, rr_head_;

 public:
  /**====================================
   * Default Constructor
   * ===================================*/

  /** Constructor. Default. */
  HSHM_CROSS_FUN
  explicit split_ticket_queue(size_t depth_per_split = 1024, size_t split = 0) {
    shm_init(HSHM_MEMORY_MANAGER->GetDefaultAllocator<AllocT>(),
             depth_per_split, split);
  }

  /** SHM constructor. Default. */
  HSHM_CROSS_FUN
  explicit split_ticket_queue(const hipc::CtxAllocator<AllocT> &alloc,
                              size_t depth_per_split = 1024, size_t split = 0) {
    shm_init(alloc, depth_per_split, split);
  }

  /** SHM constructor */
  void shm_init(const hipc::CtxAllocator<AllocT> &alloc,
                size_t depth_per_split = 1024, size_t split = 0) {
    init_shm_container(alloc);
    if (split == 0) {
      split = HSHM_SYSTEM_INFO->ncpu_;
    }
    splits_.shm_init(GetCtxAllocator(), split, depth_per_split);
    SetNull();
  }

  /**====================================
   * Copy Constructors
   * ===================================*/

  /** Copy constructor */
  HSHM_CROSS_FUN
  explicit split_ticket_queue(const split_ticket_queue &other) {
    init_shm_container(other.GetCtxAllocator());
    SetNull();
    shm_strong_copy_op(other);
  }

  /** SHM copy constructor */
  HSHM_CROSS_FUN
  explicit split_ticket_queue(const hipc::CtxAllocator<AllocT> &alloc,
                              const split_ticket_queue &other) {
    init_shm_container(alloc);
    SetNull();
    shm_strong_copy_op(other);
  }

  /** SHM copy assignment operator */
  HSHM_CROSS_FUN
  split_ticket_queue &operator=(const split_ticket_queue &other) {
    if (this != &other) {
      shm_destroy();
      shm_strong_copy_op(other);
    }
    return *this;
  }

  /** SHM copy constructor + operator main */
  HSHM_CROSS_FUN
  void shm_strong_copy_op(const split_ticket_queue &other) {
    (*splits_) = (*other.splits_);
  }

  /**====================================
   * Move Constructors
   * ===================================*/

  /** Move constructor. */
  HSHM_CROSS_FUN
  split_ticket_queue(split_ticket_queue &&other) noexcept {
    shm_move_op<false>(other.GetCtxAllocator(), std::move(other));
  }

  /** SHM move constructor. */
  HSHM_CROSS_FUN
  split_ticket_queue(const hipc::CtxAllocator<AllocT> &alloc,
                     split_ticket_queue &&other) noexcept {
    shm_move_op<false>(alloc, std::move(other));
  }

  /** SHM move assignment operator. */
  HSHM_CROSS_FUN
  split_ticket_queue &operator=(split_ticket_queue &&other) noexcept {
    if (this != &other) {
      shm_move_op<true>(GetCtxAllocator(), std::move(other));
    }
    return *this;
  }

  /** SHM move assignment operator. */
  template <bool IS_ASSIGN>
  HSHM_CROSS_FUN void shm_move_op(const hipc::CtxAllocator<AllocT> &alloc,
                                  split_ticket_queue &&other) noexcept {
    if constexpr (!IS_ASSIGN) {
      init_shm_container(alloc);
    }
    if (GetAllocator() == other.GetAllocator()) {
      (*splits_) = std::move(*other.splits_);
      other.SetNull();
    } else {
      shm_strong_copy_op(other);
      other.shm_destroy();
    }
  }

  /**====================================
   * Destructor
   * ===================================*/

  /** SHM destructor.  */
  HSHM_CROSS_FUN
  void shm_destroy_main() { (*splits_).shm_destroy(); }

  /** Check if the list is empty */
  HSHM_CROSS_FUN
  bool IsNull() const { return (*splits_).IsNull(); }

  /** Sets this list as empty */
  HSHM_CROSS_FUN
  void SetNull() {
    rr_tail_ = 0;
    rr_head_ = 0;
  }

  /**====================================
   * ticket Queue Methods
   * ===================================*/

  /** Construct an element at \a pos position in the queue */
  template <typename... Args>
  HSHM_CROSS_FUN qtok_t emplace(T &tkt) {
    uint16_t rr = rr_tail_.fetch_add(1);
    auto &splits = (*splits_);
    size_t num_splits = splits.size();
    uint16_t qid_start = rr % num_splits;
    for (size_t i = 0; i < num_splits; ++i) {
      uint32_t qid = (qid_start + i) % num_splits;
      ticket_queue_t &queue = (*splits_)[qid];
      qtok_t qtok = queue.emplace(tkt);
      if (!qtok.IsNull()) {
        return qtok;
      }
    }
    return qtok_t::GetNull();
  }

 public:
  /** Pop an element from the queue */
  HSHM_CROSS_FUN
  qtok_t pop(T &tkt) {
    uint16_t rr = rr_head_.fetch_add(1);
    auto &splits = (*splits_);
    size_t num_splits = splits.size();
    uint16_t qid_start = rr % num_splits;
    for (size_t i = 0; i < num_splits; ++i) {
      uint32_t qid = (qid_start + i) % num_splits;
      ticket_queue_t &queue = (*splits_)[qid];
      qtok_t qtok = queue.pop(tkt);
      if (!qtok.IsNull()) {
        return qtok;
      }
    }
    return qtok_t::GetNull();
  }
};

}  // namespace hshm::ipc

namespace hshm {

template <typename T, HSHM_CLASS_TEMPL_WITH_PRIV_DEFAULTS>
using split_ticket_queue = hipc::split_ticket_queue<T, HSHM_CLASS_TEMPL_ARGS>;

}  // namespace hshm

#undef CLASS_NEW_ARGS
#undef CLASS_NAME

#endif  // HSHM_SHM__DATA_STRUCTURES_IPC_SPLIT_TICKET_QUEUE_H_
