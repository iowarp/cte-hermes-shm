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

#ifndef HSHM_SHM_INCLUDE_HSHM_SHM_DATA_STRUCTURES_IPC_TICKET_QUEUE_H_
#define HSHM_SHM_INCLUDE_HSHM_SHM_DATA_STRUCTURES_IPC_TICKET_QUEUE_H_

#include "hermes_shm/data_structures/internal/shm_internal.h"
#include "hermes_shm/thread/lock.h"
#include "hermes_shm/types/qtok.h"
#include "ring_queue.h"

namespace hshm::ipc {

/** Forward declaration of ticket_queue */
template <typename T, HSHM_CLASS_TEMPL_WITH_DEFAULTS>
class ticket_queue;

/**
 * MACROS used to simplify the ticket_queue namespace
 * Used as inputs to the HIPC_CONTAINER_TEMPLATE
 * */
#define CLASS_NAME ticket_queue
#define CLASS_NEW_ARGS T

/**
 * A MPMC queue for allocating tickets. Handles concurrency
 * without blocking.
 * */
template <typename T, HSHM_CLASS_TEMPL>
class ticket_queue : public ShmContainer {
 public:
  HIPC_CONTAINER_TEMPLATE((CLASS_NAME), (CLASS_NEW_ARGS))
  delay_ar<fixed_spsc_queue<T, hipc::EmptyHeader, HSHM_CLASS_TEMPL_ARGS>> queue_;
  hshm::Mutex lock_;

 public:
  /**====================================
   * Default Constructor
   * ===================================*/

  /** Constructor. Default. */
  HSHM_CROSS_FUN
  explicit ticket_queue(size_t depth = 1024) {
    shm_init(HSHM_MEMORY_MANAGER->GetDefaultAllocator<AllocT>(), depth);
  }

  /** SHM constructor. Default. */
  HSHM_CROSS_FUN
  explicit ticket_queue(const hipc::CtxAllocator<AllocT> &alloc,
                        size_t depth = 1024) {
    shm_init(alloc, depth);
  }

  /** SHM Constructor. */
  void shm_init(const hipc::CtxAllocator<AllocT> &alloc, size_t depth = 1024) {
    init_shm_container(alloc);
    queue_.shm_init(GetCtxAllocator(), depth);
    lock_.Init();
    SetNull();
  }

  /**====================================
   * Copy Constructors
   * ===================================*/

  /** Copy constructor */
  HSHM_CROSS_FUN
  explicit ticket_queue(const ticket_queue &other) {
    init_shm_container(other.GetCtxAllocator());
    SetNull();
    shm_strong_copy_op(other);
  }

  /** SHM copy constructor */
  HSHM_CROSS_FUN
  explicit ticket_queue(const hipc::CtxAllocator<AllocT> &alloc,
                        const ticket_queue &other) {
    init_shm_container(alloc);
    SetNull();
    shm_strong_copy_op(other);
  }

  /** SHM copy assignment operator */
  HSHM_CROSS_FUN
  ticket_queue &operator=(const ticket_queue &other) {
    if (this != &other) {
      shm_destroy();
      shm_strong_copy_op(other);
    }
    return *this;
  }

  /** SHM copy constructor + operator main */
  HSHM_CROSS_FUN
  void shm_strong_copy_op(const ticket_queue &other) {
    (*queue_) = (*other.queue_);
  }

  /**====================================
   * Move Constructors
   * ===================================*/

  /** Move constructor. */
  HSHM_CROSS_FUN
  ticket_queue(ticket_queue &&other) noexcept {
    shm_move_op<false>(other.GetCtxAllocator(), std::move(other));
  }

  /** SHM move constructor. */
  HSHM_CROSS_FUN
  ticket_queue(const hipc::CtxAllocator<AllocT> &alloc,
               ticket_queue &&other) noexcept {
    shm_move_op<false>(alloc, std::move(other));
  }

  /** SHM move assignment operator. */
  HSHM_CROSS_FUN
  ticket_queue &operator=(ticket_queue &&other) noexcept {
    if (this != &other) {
      shm_move_op<true>(GetCtxAllocator(), std::move(other));
    }
    return *this;
  }

  /** SHM move operator. */
  template <bool IS_ASSIGN>
  HSHM_CROSS_FUN void shm_move_op(const hipc::CtxAllocator<AllocT> &alloc,
                                  ticket_queue &&other) noexcept {
    if constexpr (!IS_ASSIGN) {
      init_shm_container(alloc);
    }
    if (GetAllocator() == other.GetAllocator()) {
      (*queue_) = std::move(*other.queue_);
      other.SetNull();
    } else {
      shm_strong_copy_op(other);
      other.shm_destroy();
    }
  }

  /**====================================
   * Destructor
   * ===================================*/

  /** SHM destructor. */
  HSHM_CROSS_FUN
  void shm_destroy_main() { (*queue_).shm_destroy(); }

  /** Check if the list is empty */
  HSHM_CROSS_FUN
  bool IsNull() const { return (*queue_).IsNull(); }

  /** Sets this list as empty */
  HSHM_CROSS_FUN
  void SetNull() {}

  /**====================================
   * ticket Queue Methods
   * ===================================*/

  /** Construct an element at \a pos position in the queue */
  template <typename... Args>
  HSHM_INLINE_CROSS_FUN qtok_t emplace(T &tkt) {
    lock_.Lock(0);
    auto qtok = queue_->emplace(tkt);
    lock_.Unlock();
    return qtok;
  }

 public:
  /** Pop an element from the queue */
  HSHM_INLINE_CROSS_FUN qtok_t pop(T &tkt) {
    lock_.Lock(0);
    auto qtok = queue_->pop(tkt);
    lock_.Unlock();
    return qtok;
  }
};

}  // namespace hshm::ipc

namespace hshm {

template <typename T, HSHM_CLASS_TEMPL_WITH_PRIV_DEFAULTS>
using ticket_queue = hipc::ticket_queue<T, HSHM_CLASS_TEMPL_ARGS>;

}  // namespace hshm

#undef CLASS_NEW_ARGS
#undef CLASS_NAME

#endif  // HSHM_SHM_INCLUDE_HSHM_SHM_DATA_STRUCTURES_IPC_TICKET_QUEUE_H_
