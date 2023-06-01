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

/**
 * Let's say the queue is massive.
 * 64 concurrent accesses are made and they all fit.
 * We place our elements into the queue
 *
 * 16 concurrent consumers come while this is happening
 *
 * */

#ifndef HERMES_SHM_INCLUDE_HERMES_SHM_DATA_STRUCTURES_IPC_mpmc_queue_templ_H_
#define HERMES_SHM_INCLUDE_HERMES_SHM_DATA_STRUCTURES_IPC_mpmc_queue_templ_H_

#include "hermes_shm/data_structures/ipc/internal/shm_internal.h"
#include "hermes_shm/thread/lock.h"
#include "vector.h"
#include "pair.h"
#include "_queue.h"

namespace hshm::ipc {

/** Forward declaration of mpmc_queue_templ */
template<typename T>
class mpmc_queue_templ;

/**
 * MACROS used to simplify the mpmc_queue_templ namespace
 * Used as inputs to the SHM_CONTAINER_TEMPLATE
 * */
#define CLASS_NAME mpmc_queue_templ
#define TYPED_CLASS mpmc_queue_templ<T>
#define TYPED_HEADER ShmHeader<mpmc_queue_templ<T>>

#define MPMC_SLOT_CLEAR 0
#define MPMC_SLOT_READY 1

/**
 * A queue optimized for multiple producers (emplace) with a single
 * consumer (pop).
 * */
template<typename T>
class mpmc_queue_templ : public ShmContainer {
 public:
  SHM_CONTAINER_TEMPLATE((CLASS_NAME), (TYPED_CLASS))
  ShmArchive<vector<pair<bitfield32_t, T>>> queue_;
  std::atomic<_qtok_t> tail_min_, tail_max_;
  std::atomic<_qtok_t> head_min_, head_max_;
  std::atomic<_qtok_t> conc_;
  RwLock lock_;

 public:
  /**====================================
   * Default Constructor
   * ===================================*/

  /** SHM constructor. Default. */
  explicit mpmc_queue_templ(Allocator *alloc,
                            size_t depth = 1024,
                            size_t max_conc = 64) {
    shm_init_container(alloc);
    HSHM_MAKE_AR(queue_, GetAllocator(), depth);
    SetNull();
  }

  /**====================================
   * Copy Constructors
   * ===================================*/

  /** SHM copy constructor */
  explicit mpmc_queue_templ(Allocator *alloc,
                            const mpmc_queue_templ &other) {
    shm_init_container(alloc);
    SetNull();
    shm_strong_copy_construct_and_op(other);
  }

  /** SHM copy assignment operator */
  mpmc_queue_templ& operator=(const mpmc_queue_templ &other) {
    if (this != &other) {
      shm_destroy();
      shm_strong_copy_construct_and_op(other);
    }
    return *this;
  }

  /** SHM copy constructor + operator main */
  void shm_strong_copy_construct_and_op(const mpmc_queue_templ &other) {
    head_min_ = other.head_min_.load();
    head_max_ = other.head_max_.load();
    tail_min_ = other.tail_.load();
    tail_max_ = other.tail_.load();
    (*queue_) = (*other.queue_);
  }

  /**====================================
   * Move Constructors
   * ===================================*/

  /** SHM move constructor. */
  mpmc_queue_templ(Allocator *alloc,
                   mpmc_queue_templ &&other) noexcept {
    shm_init_container(alloc);
    if (GetAllocator() == other.GetAllocator()) {
      head_min_ = other.head_min_.load();
      head_max_ = other.head_max_.load();
      tail_min_ = other.tail_.load();
      tail_max_ = other.tail_.load();
      (*queue_) = std::move(*other.queue_);
      other.SetNull();
    } else {
      shm_strong_copy_construct_and_op(other);
      other.shm_destroy();
    }
  }

  /** SHM move assignment operator. */
  mpmc_queue_templ& operator=(mpmc_queue_templ &&other) noexcept {
    if (this != &other) {
      shm_destroy();
      if (GetAllocator() == other.GetAllocator()) {
        head_min_ = other.head_min_.load();
        head_max_ = other.head_max_.load();
        tail_min_ = other.tail_.load();
        tail_max_ = other.tail_.load();
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
    conc_ = 0;
    head_min_ = 0;
    head_max_ = 0;
    tail_min_ = 0;
    tail_max_ = 0;
  }

  /**====================================
   * mpmc Queue Methods
   * ===================================*/

  /** Construct an element at \a pos position in the list */
  template<typename ...Args>
  qtok_t emplace(Args&&... args) {
    _qtok_t conc = conc_.fetch_add(1);
    _qtok_t tail_max;
    _qtok_t head_min;
    _qtok_t entry_tok;
    size_t size;

    // Verify the upper-bound entry exists
    do {
      tail_max = tail_max_.load();
      head_min = head_min_.load();
      entry_tok  = tail_max + conc;
      size = entry_tok - head_min + 1;
      if (size > (*queue_).size()) {
        UpdateHead(conc);
        UpdateTail(conc);
        conc_.fetch_sub(1);
        return qtok_t::GetNull();
      }
    } while (head_min_.load() != head_min && tail_max_.load() != tail_max);

    // Reserve a slot
    entry_tok = tail_max_.fetch_add(1);

    // Update the entry data in the queue
    uint32_t idx = entry_tok % (*queue_).size();
    auto iter = (*queue_).begin() + idx;
    hipc::pair<bitfield32_t, T> &entry = *iter;
    (*queue_).replace(iter,
                      hshm::PiecewiseConstruct(),
                      make_argpack(),
                      make_argpack(std::forward<Args>(args)...));
    entry.GetFirst().SetBits(MPMC_SLOT_READY);

    // Update the tail_min_ pointer to be the first entry that is CLEAR
    UpdateTail(conc);
    conc_.fetch_sub(1);
    return qtok_t(entry_tok);
  }

 public:
  /** Consumer pops the head object */
  qtok_t pop(T &val) {
    _qtok_t conc = conc_.fetch_add(1);
    _qtok_t head_max = head_max_.load();
    _qtok_t tail_min = tail_min_.load();
    _qtok_t entry_tok = head_max + conc;

    // Verify the entry exists
    if (entry_tok >= tail_min) {
      UpdateHead(conc);
      UpdateTail(conc);
      conc_.fetch_sub(1);
      return qtok_t::GetNull();
    }

    // Allocate the entry
    entry_tok = head_max_.fetch_add(1);

    // Get the value of the queue
    _qtok_t idx = entry_tok % (*queue_).size();
    hipc::pair<bitfield32_t, T> &entry = (*queue_)[idx];
    val = std::move(entry.GetSecond());
    entry.GetFirst().Clear();

    // Update the head_min_ ptr to be the first thing that is READY
    UpdateHead(conc);
    conc_.fetch_sub(1);
    return qtok_t(entry_tok);
  }

 private:
  HSHM_ALWAYS_INLINE void UpdateTail(_qtok_t &tail_conc) {
    if (tail_conc == 0) {
      _qtok_t tail_max = tail_max_.load();
      for (size_t tail_min = tail_min_.load();
           tail_min < tail_max; ++tail_min) {
        size_t idx = tail_min % (*queue_).size();
        auto &entry = (*queue_)[idx];
        if (!entry.GetFirst().Any(MPMC_SLOT_READY)) {
          break;
        }
        tail_min_.fetch_add(1);
      }
    }
  }

  HSHM_ALWAYS_INLINE void UpdateHead(_qtok_t &head_conc) {
    if (head_conc == 0) {
      _qtok_t head_max = head_max_.load();
      for (size_t head_min = head_min_.load();
           head_min < head_max; ++head_min) {
        size_t idx = head_min % (*queue_).size();
        auto &entry = (*queue_)[idx];
        if (entry.GetFirst().Any(MPMC_SLOT_READY)) {
          break;
        }
        head_min_.fetch_add(1);
      }
    }
  }
};

template<typename T>
using mpmc_queue = mpmc_queue_templ<T>;

}  // namespace hshm::ipc

#undef CLASS_NAME
#undef TYPED_CLASS
#undef TYPED_HEADER

#endif  // HERMES_SHM_INCLUDE_HERMES_SHM_DATA_STRUCTURES_IPC_mpmc_queue_templ_H_
