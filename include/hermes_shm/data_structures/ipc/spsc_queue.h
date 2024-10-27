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

#ifndef HERMES_SHM_INCLUDE_HERMES_SHM_DATA_STRUCTURES_IPC_spsc_queue_templ_H_
#define HERMES_SHM_INCLUDE_HERMES_SHM_DATA_STRUCTURES_IPC_spsc_queue_templ_H_

#include "hermes_shm/data_structures/ipc/internal/shm_internal.h"
#include "hermes_shm/util/auto_trace.h"
#include "hermes_shm/thread/lock.h"
#include "vector.h"
#include "pair.h"
#include "hermes_shm/types/qtok.h"

namespace hshm::ipc {

/** Forward declaration of spsc_queue_templ */
template<typename T, bool EXTENSIBLE, typename AllocT = HSHM_DEFAULT_ALLOC>
class spsc_queue_templ;

/**
 * MACROS used to simplify the spsc_queue_templ namespace
 * Used as inputs to the HIPC_CONTAINER_TEMPLATE
 * */
#define CLASS_NAME spsc_queue_templ
#define TYPED_CLASS spsc_queue_templ<T, EXTENSIBLE>

/**
 * A queue optimized for multiple producers (emplace) with a single
 * consumer (pop).
 * */
template<typename T, bool EXTENSIBLE, typename AllocT>
class spsc_queue_templ : public ShmContainer {
 public:
  HIPC_CONTAINER_TEMPLATE((CLASS_NAME), (TYPED_CLASS))
  ShmArchive<vector<T, AllocT>> queue_;
  qtok_id tail_;
  qtok_id head_;

 public:
  /**====================================
   * Default Constructor
   * ===================================*/

  /** SHM constructor. Default. */
  HSHM_CROSS_FUN
  explicit spsc_queue_templ(size_t depth = 1024) {
    shm_init(HERMES_MEMORY_MANAGER->GetDefaultAllocator(), depth);
  }

  /** SHM constructor. Default. */
  HSHM_CROSS_FUN
  explicit spsc_queue_templ(AllocT *alloc, size_t depth = 1024) {
    shm_init(alloc, depth);
  }

  /** SHM constructor. */
  HSHM_CROSS_FUN
  void shm_init(AllocT *alloc, size_t depth = 1024) {
    init_shm_container(alloc);
    HSHM_MAKE_AR(queue_, GetAllocator(), depth)
    SetNull();
  }

  /**====================================
   * Copy Constructors
   * ===================================*/

  /** Copy constructor */
  HSHM_CROSS_FUN
  explicit spsc_queue_templ(const spsc_queue_templ &other) {
    init_shm_container(other.GetAllocator());
    SetNull();
    shm_strong_copy_op(other);
  }

  /** SHM copy constructor */
  HSHM_CROSS_FUN
  explicit spsc_queue_templ(AllocT *alloc,
                            const spsc_queue_templ &other) {
    init_shm_container(alloc);
    SetNull();
    shm_strong_copy_op(other);
  }

  /** SHM copy assignment operator */
  HSHM_CROSS_FUN
  spsc_queue_templ& operator=(const spsc_queue_templ &other) {
    if (this != &other) {
      shm_destroy();
      shm_strong_copy_op(other);
    }
    return *this;
  }

  /** SHM copy constructor + operator main */
  HSHM_CROSS_FUN
  void shm_strong_copy_op(const spsc_queue_templ &other) {
    head_ = other.head_;
    tail_ = other.tail_;
    (*queue_) = (*other.queue_);
  }

  /**====================================
   * Move Constructors
   * ===================================*/

  /** SHM move constructor. */
  HSHM_CROSS_FUN
  spsc_queue_templ(spsc_queue_templ &&other) noexcept {
    shm_move_op<false>(other.GetAllocator(), std::move(other));
  }

  /** SHM move constructor. */
  HSHM_CROSS_FUN
  spsc_queue_templ(AllocT *alloc,
                   spsc_queue_templ &&other) noexcept {
    shm_move_op<false>(alloc, std::move(other));
  }

  /** SHM move assignment operator. */
  HSHM_CROSS_FUN
  spsc_queue_templ& operator=(spsc_queue_templ &&other) noexcept {
    if (this != &other) {
      shm_move_op<true>(GetAllocator(), std::move(other));
    }
    return *this;
  }

  /** SHM move assignment operator. */
  template<bool IS_ASSIGN>
  HSHM_CROSS_FUN
  void shm_move_op(AllocT *alloc, spsc_queue_templ &&other) noexcept {
    if constexpr (IS_ASSIGN) {
      shm_destroy();
    } else {
      init_shm_container(alloc);
    }
    if (GetAllocator() == other.GetAllocator()) {
      head_ = other.head_;
      tail_ = other.tail_;
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

  /** SHM destructor.  */
  HSHM_CROSS_FUN
  void shm_destroy_main() {
    (*queue_).shm_destroy();
  }

  /** Check if the list is empty */
  HSHM_CROSS_FUN
  bool IsNull() const {
    return (*queue_).IsNull();
  }

  /** Sets this list as empty */
  HSHM_CROSS_FUN
  void SetNull() {
    head_ = 0;
    tail_ = 0;
  }

  /**====================================
   * spsc Queue Methods
   * ===================================*/

  /** Construct an element at \a pos position in the list */
  template<typename ...Args>
  HSHM_CROSS_FUN
  qtok_t emplace(Args&&... args) {
    // Don't emplace if there is no space
    qtok_id entry_tok = tail_;
    size_t size = tail_ - head_;
    auto &queue = (*queue_);
    if (size >= queue.size()) {
      return qtok_t::GetNull();
    }

    // Do the emplace
    qtok_id idx = entry_tok % queue.size();
    auto iter = queue.begin() + idx;
    queue.replace(iter, std::forward<Args>(args)...);
    tail_ += 1;
    return qtok_t(entry_tok);
  }

 public:
  /** Consumer pops the head object */
  HSHM_CROSS_FUN
  qtok_t pop(T &val) {
    // Don't pop if there's no entries
    qtok_id head = head_;
    qtok_id tail = tail_;
    if (head >= tail) {
      return qtok_t::GetNull();
    }

    // Pop the element
    auto &queue = (*queue_);
    qtok_id idx = head % queue.size();
    T &entry = queue[idx];
    (val) = std::move(entry);
    head_ += 1;
    return qtok_t(head);
  }
};

template<typename T>
using spsc_queue_ext = spsc_queue_templ<T, true>;

template<typename T>
using spsc_queue = spsc_queue_templ<T, false>;

}  // namespace hshm::ipc

#undef CLASS_NAME
#undef TYPED_CLASS

#endif  // HERMES_SHM_INCLUDE_HERMES_SHM_DATA_STRUCTURES_IPC_spsc_queue_templ_H_
