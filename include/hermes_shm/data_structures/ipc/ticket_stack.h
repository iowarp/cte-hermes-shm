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

#ifndef HERMES_SHM_INCLUDE_HERMES_SHM_DATA_STRUCTURES_IPC_TICKET_STACK_H_
#define HERMES_SHM_INCLUDE_HERMES_SHM_DATA_STRUCTURES_IPC_TICKET_STACK_H_

#include "hermes_shm/data_structures/ipc/internal/shm_internal.h"
#include "hermes_shm/thread/lock.h"
#include "vector.h"
#include "_queue.h"

namespace hshm::ipc {

/** Forward declaration of ticket_stack */
template<typename T>
class ticket_stack;

/**
 * MACROS used to simplify the ticket_stack namespace
 * Used as inputs to the SHM_CONTAINER_TEMPLATE
 * */
#define CLASS_NAME ticket_stack
#define TYPED_CLASS ticket_stack<T>
#define TYPED_HEADER ShmHeader<ticket_stack<T>>

#define MARK_FIRST_BIT (((T)1) << (8 * sizeof(T) - 1))
#define MARK_TICKET(tkt) ((tkt) | MARK_FIRST_BIT)
#define IS_MARKED(tkt) ((tkt) & MARK_FIRST_BIT)
#define UNMARK_TICKET(tkt) ((tkt) & ~MARK_FIRST_BIT)

#include <vector>
struct Record {
  int method_;
  size_t cur_data_;
  size_t new_data_;
  size_t entry_tok_;
  size_t new_tail_;
  bool accept_;
};

struct RecordManager {
  std::vector<Record> records_;
  std::atomic<size_t> cnt_;

  RecordManager() {
    cnt_ = 0;
    records_.resize(1 << 20);
  }

  void emplace(int method, size_t cur_data,
               size_t new_data, size_t new_tail,
               size_t entry_tok, bool accept) {
    if (cnt_ >= records_.size()) {
      HILOG(kInfo, "HERE");
      exit(1);
    }
    if (!accept) {
      return;
    }
    size_t cnt = cnt_.fetch_add(1);
    auto &record = records_[cnt];
    record.method_ = method;
    record.cur_data_ = cur_data;
    record.new_data_ = new_data;
    record.new_tail_ = new_tail;
    record.accept_ = accept;
    record.entry_tok_ = entry_tok;
  }
};

/**
 * A MPMC queue for allocating tickets. Handles concurrency
 * without blocking.
 * */
template<typename T>
class ticket_stack : public ShmContainer {
 public:
  SHM_CONTAINER_TEMPLATE((CLASS_NAME), (TYPED_CLASS))
  ShmArchive<vector<T>> queue_;
  std::atomic<_qtok_t> tail_;
  RecordManager records_;

 public:
  /**====================================
   * Default Constructor
   * ===================================*/

  /** SHM constructor. Default. */
  explicit ticket_stack(Allocator *alloc,
                        size_t depth = 1024) {
    shm_init_container(alloc);
    HSHM_MAKE_AR(queue_, GetAllocator(), depth, 0);
    SetNull();
  }

  /**====================================
   * Copy Constructors
   * ===================================*/

  /** SHM copy constructor */
  explicit ticket_stack(Allocator *alloc,
                        const ticket_stack &other) {
    shm_init_container(alloc);
    SetNull();
    shm_strong_copy_construct_and_op(other);
  }

  /** SHM copy assignment operator */
  ticket_stack& operator=(const ticket_stack &other) {
    if (this != &other) {
      shm_destroy();
      shm_strong_copy_construct_and_op(other);
    }
    return *this;
  }

  /** SHM copy constructor + operator main */
  void shm_strong_copy_construct_and_op(const ticket_stack &other) {
    tail_ = other.tail_.load();
    (*queue_) = (*other.queue_);
  }

  /**====================================
   * Move Constructors
   * ===================================*/

  /** SHM move constructor. */
  ticket_stack(Allocator *alloc,
               ticket_stack &&other) noexcept {
    shm_init_container(alloc);
    if (GetAllocator() == other.GetAllocator()) {
      tail_ = other.tail_.load();
      (*queue_) = std::move(*other.queue_);
      other.SetNull();
    } else {
      shm_strong_copy_construct_and_op(other);
      other.shm_destroy();
    }
  }

  /** SHM move assignment operator. */
  ticket_stack& operator=(ticket_stack &&other) noexcept {
    if (this != &other) {
      shm_destroy();
      if (GetAllocator() == other.GetAllocator()) {
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
    tail_ = 0;
  }

  /**====================================
   * ticket Queue Methods
   * ===================================*/

  /** Construct an element at \a pos position in the queue */
  template<typename ...Args>
  qtok_t emplace(T &tkt) {
    auto &queue = *queue_;
    T* data = (T*)queue.data();
    do {
      // Get the current tail
      _qtok_t entry_tok = tail_.load();
      size_t queue_size = queue.size();
      if (entry_tok >= queue_size) {
        records_.emplace(0, -1, tkt, entry_tok + 1, entry_tok, false);
        return qtok_t::GetNull();
      }
      _qtok_t tail = entry_tok + 1;

      // Verify tail exists
      auto &entry = queue[entry_tok];
      if (IS_MARKED(entry)) {
        records_.emplace(1, entry, tkt, tail, entry_tok, false);
        return qtok_t::GetNull();
      }

      // Claim the tail
      bool ret = tail_.compare_exchange_weak(entry_tok, tail);
      if (!ret) {
        records_.emplace(2, entry, tkt, tail, entry_tok, false);
        continue;
      }

      // Update the tail
      entry = MARK_TICKET(tkt);
      records_.emplace(3, entry, tkt, tail, entry_tok, true);
      return qtok_t(entry_tok);
    } while (true);
  }

 public:
  /** Pop an element from the queue */
  qtok_t pop(T &tkt) {
    auto &queue = *queue_;
    T* data = (T*)queue.data();
    do {
      // Get the current head
      _qtok_t tail = tail_.load();
      if (tail == 0) {
        records_.emplace(4, -1, tkt, tail - 1, tail - 1, false);
        return qtok_t::GetNull();
      }
      _qtok_t entry_tok = tail - 1;

      // Verify head is marked
      auto &entry = queue[entry_tok];
      if (!IS_MARKED(entry)) {
        records_.emplace(5, entry, tkt, entry_tok, entry_tok, false);
        return qtok_t::GetNull();
      }

      // Claim the head
      bool ret = tail_.compare_exchange_weak(tail, entry_tok);
      if (!ret) {
        records_.emplace(6, entry, tkt, entry_tok, entry_tok, false);
        continue;
      }

      // Update the head
      tkt = UNMARK_TICKET(entry);
      entry = 0;
      records_.emplace(7, entry, tkt, entry_tok, entry_tok, true);
      return qtok_t(entry_tok);
    } while (true);
  }
};

}  // namespace hshm::ipc

#undef TYPED_HEADER
#undef TYPED_CLASS
#undef CLASS_NAME

#endif  // HERMES_SHM_INCLUDE_HERMES_SHM_DATA_STRUCTURES_IPC_TICKET_STACK_H_
