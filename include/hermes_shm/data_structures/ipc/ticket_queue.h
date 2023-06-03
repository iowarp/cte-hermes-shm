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

#ifndef HERMES_SHM_INCLUDE_HERMES_SHM_DATA_STRUCTURES_IPC_TICKET_QUEUE_H_
#define HERMES_SHM_INCLUDE_HERMES_SHM_DATA_STRUCTURES_IPC_TICKET_QUEUE_H_

#include "hermes_shm/data_structures/ipc/internal/shm_internal.h"
#include "hermes_shm/thread/lock.h"
#include "vector.h"
#include "pair.h"
#include "_queue.h"

namespace hshm::ipc {

/** Forward declaration of ticket_queue_templ */
template<typename T>
class ticket_queue_templ;

/**
 * MACROS used to simplify the ticket_queue_templ namespace
 * Used as inputs to the SHM_CONTAINER_TEMPLATE
 * */
#define CLASS_NAME ticket_queue_templ
#define TYPED_CLASS ticket_queue_templ<T>
#define TYPED_HEADER ShmHeader<ticket_queue_templ<T>>

#define MARK_FIRST_BIT (((T)1) << (8 * sizeof(T) - 1))
#define MARK_TICKET(tkt) ((tkt) | MARK_FIRST_BIT)
#define IS_MARKED(tkt) ((tkt) & MARK_FIRST_BIT)
#define UNMARK_TICKET(tkt) ((tkt) & ~MARK_FIRST_BIT)

struct Conc {
  std::atomic<uint32_t> &conc_;
  uint32_t val_;

  HSHM_ALWAYS_INLINE explicit Conc(std::atomic<uint32_t> &conc)
  : conc_(conc) {
    val_ = conc_.fetch_add(1);
  }

  HSHM_ALWAYS_INLINE ~Conc() {
    conc_.fetch_sub(1);
  }
};

/**
 * A queue optimized for allocating (integer) tickets.
 * The queue has a fixed size.
 * */
template<typename T>
class ticket_queue_templ : public ShmContainer {
 public:
  SHM_CONTAINER_TEMPLATE((CLASS_NAME), (TYPED_CLASS))
  ShmArchive<vector<T>> queue_;
  std::atomic<_qtok_t> head_, tail_;
  // std::atomic<_qtok_t> head_max_, tail_min_;

 public:
  /**====================================
   * Default Constructor
   * ===================================*/

  /** SHM constructor. Default. */
  explicit ticket_queue_templ(Allocator *alloc,
                              size_t depth = 1024) {
    shm_init_container(alloc);
    HSHM_MAKE_AR(queue_, GetAllocator(), depth);
    SetNull();
  }

  /**====================================
   * Copy Constructors
   * ===================================*/

  /** SHM copy constructor */
  explicit ticket_queue_templ(Allocator *alloc,
                            const ticket_queue_templ &other) {
    shm_init_container(alloc);
    SetNull();
    shm_strong_copy_construct_and_op(other);
  }

  /** SHM copy assignment operator */
  ticket_queue_templ& operator=(const ticket_queue_templ &other) {
    if (this != &other) {
      shm_destroy();
      shm_strong_copy_construct_and_op(other);
    }
    return *this;
  }

  /** SHM copy constructor + operator main */
  void shm_strong_copy_construct_and_op(const ticket_queue_templ &other) {
    head_ = other.head_.load();
    tail_ = other.tail_.load();
    (*queue_) = (*other.queue_);
  }

  /**====================================
   * Move Constructors
   * ===================================*/

  /** SHM move constructor. */
  ticket_queue_templ(Allocator *alloc,
                   ticket_queue_templ &&other) noexcept {
    shm_init_container(alloc);
    if (GetAllocator() == other.GetAllocator()) {
      head_ = other.head_.load();
      tail_ = other.tail_.load();
      (*queue_) = std::move(*other.queue_);
      other.SetNull();
    } else {
      shm_strong_copy_construct_and_op(other);
      other.shm_destroy();
    }
  }

  /** SHM move assignment operator. */
  ticket_queue_templ& operator=(ticket_queue_templ &&other) noexcept {
    if (this != &other) {
      shm_destroy();
      if (GetAllocator() == other.GetAllocator()) {
        head_ = other.head_.load();
        tail_ = other.tail_.load();
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
    (*queue_).shm_destroy();
  }

  /** Check if the list is empty */
  bool IsNull() const {
    return (*queue_).IsNull();
  }

  /** Sets this list as empty */
  void SetNull() {
    head_ = 0;
    tail_ = 0;
    // head_max_ = 0;
    // tail_min_ = 0;
  }

  /**====================================
   * ticket Queue Methods
   * ===================================*/

  /** Construct an element at \a pos position in the queue */
  template<typename ...Args>
  qtok_t emplace(T &tkt) {
    do {
      // Get the current tail
      _qtok_t entry_tok = tail_.load();
      _qtok_t tail = entry_tok + 1;
      _qtok_t head = head_.load();
      size_t size = tail - head;

      // Verify tail exists
      if (size > (*queue_).size()) {
        return qtok_t::GetNull();
      }

      // Claim the tail
      bool ret = tail_.compare_exchange_weak(entry_tok, tail);
      if (!ret) {
        continue;
      }

      // Update the tail
      uint32_t idx = entry_tok % (*queue_).size();
      (*queue_)[idx] = MARK_TICKET(tkt);
      return qtok_t(entry_tok);
    } while (true);
  }

 public:
  /** Pop an element from the queue */
  qtok_t pop(T &tkt) {
    do {
      // Get the current head
      _qtok_t entry_tok = head_.load();
      _qtok_t tail = tail_.load();
      if (entry_tok >= tail) {
        return qtok_t::GetNull();
      }
      _qtok_t head = entry_tok + 1;

      // Verify head is marked
      uint32_t idx = entry_tok % (*queue_).size();
      auto &entry = (*queue_)[idx];
      tkt = entry;
      if (!IS_MARKED(tkt)) {
        return qtok_t::GetNull();
      }

      // Claim the head
      bool ret = head_.compare_exchange_weak(entry_tok, head);
      if (ret) {
        return qtok_t(entry_tok);
      }

      // Update the head
      tkt = entry;
      if (!IS_MARKED(tkt)) {
        return qtok_t::GetNull();
      }
      entry = 0;
      tkt = UNMARK_TICKET(tkt);
    } while (true);
  }
};

template<typename T>
using ticket_queue = ticket_queue_templ<T>;

}  // namespace hshm::ipc

#endif  // HERMES_SHM_INCLUDE_HERMES_SHM_DATA_STRUCTURES_IPC_TICKET_QUEUE_H_
