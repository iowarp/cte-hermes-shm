//
// Created by llogan on 28/10/24.
//

#ifndef HERMES_SHM_INCLUDE_HERMES_SHM_DATA_STRUCTURES_IPC_ring_queue_base_H_
#define HERMES_SHM_INCLUDE_HERMES_SHM_DATA_STRUCTURES_IPC_ring_queue_base_H_


#include "hermes_shm/data_structures/ipc/internal/shm_internal.h"
#include "hermes_shm/thread/lock.h"
#include "vector.h"
#include "pair.h"
#include "hermes_shm/types/qtok.h"

namespace hshm::ipc {

/** Forward declaration of ring_queue_base */
template<
    typename T,
    bool IsPushAtomic,
    bool IsPopAtomic,
    bool IsFixedSize,
    HSHM_CLASS_TEMPL_WITH_DEFAULTS>
class ring_queue_base;

/**
 * MACROS used to simplify the ring_queue_base namespace
 * Used as inputs to the HIPC_CONTAINER_TEMPLATE
 * */
#define CLASS_NAME ring_queue_base
#define TYPED_CLASS \
  ring_queue_base<T, IsPushAtomic, IsPopAtomic, IsFixedSize, HSHM_CLASS_TEMPL_ARGS>

#define PAIR_OR_POINTER

/**
 * A queue optimized for multiple producers (emplace) with a single
 * consumer (pop).
 * */
template<
    typename T,
    bool IsPushAtomic,
    bool IsPopAtomic,
    bool IsFixedSize,
    HSHM_CLASS_TEMPL>
class ring_queue_base : public ShmContainer {
 public:
 HIPC_CONTAINER_TEMPLATE((CLASS_NAME), (TYPED_CLASS))
  ShmArchive<vector<pair<bitfield32_t, T, HSHM_CLASS_TEMPL_ARGS>>> queue_;
  hipc::opt_atomic<qtok_id, IsPushAtomic> tail_;
  hipc::opt_atomic<qtok_id, IsPopAtomic> head_;
  bitfield32_t flags_;

 public:
  /**====================================
   * Default Constructor
   * ===================================*/

  /** Constructor. Default. */
  HSHM_CROSS_FUN
  explicit ring_queue_base(size_t depth = 1024) {
    shm_init(HERMES_MEMORY_MANAGER->GetDefaultAllocator(), depth);
  }

  /** SHM constructor. Default. */
  HSHM_CROSS_FUN
  explicit ring_queue_base(AllocT *alloc, size_t depth = 1024) {
    shm_init(alloc, depth);
  }

  /** SHM Constructor */
  HSHM_CROSS_FUN
  void shm_init(AllocT *alloc, size_t depth = 1024) {
    init_shm_container(alloc);
    HSHM_MAKE_AR(queue_, GetAllocator(), depth);
    flags_.Clear();
    SetNull();
  }

  /**====================================
   * Copy Constructors
   * ===================================*/

  /** SHM copy constructor */
  HSHM_CROSS_FUN
  explicit ring_queue_base(AllocT *alloc,
                      const ring_queue_base &other) {
    init_shm_container(alloc);
    SetNull();
    shm_strong_copy_op(other);
  }

  /** SHM copy assignment operator */
  HSHM_CROSS_FUN
  ring_queue_base& operator=(const ring_queue_base &other) {
    if (this != &other) {
      shm_destroy();
      shm_strong_copy_op(other);
    }
    return *this;
  }

  /** SHM copy constructor + operator main */
  HSHM_CROSS_FUN
  void shm_strong_copy_op(const ring_queue_base &other) {
    head_ = other.head_.load();
    tail_ = other.tail_.load();
    (*queue_) = (*other.queue_);
  }

  /**====================================
   * Move Constructors
   * ===================================*/

  /** Move constructor. */
  HSHM_CROSS_FUN
  ring_queue_base(ring_queue_base &&other) noexcept {
    shm_move_op<false>(other.GetAllocator(), other);
  }

  /** SHM move constructor. */
  HSHM_CROSS_FUN
  ring_queue_base(AllocT *alloc,
             ring_queue_base &&other) noexcept {
    shm_move_op<false>(alloc, other);
  }

  /** SHM move assignment operator. */
  HSHM_CROSS_FUN
  ring_queue_base& operator=(ring_queue_base &&other) noexcept {
    if (this != &other) {
      shm_move_op<true>(other.GetAllocator(), std::move(other));
    }
    return *this;
  }

  /** SHM move assignment operator. */
  template<bool IS_ASSIGN>
  HSHM_CROSS_FUN
  void shm_move_op(AllocT *alloc, ring_queue_base &&other) noexcept {
    if constexpr (IS_ASSIGN) {
      shm_destroy();
    } else {
      init_shm_container(alloc);
    }
    if (GetAllocator() == other.GetAllocator()) {
      head_ = other.head_.load();
      tail_ = other.tail_.load();
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
   * MPSC Queue Methods
   * ===================================*/

  /** Construct an element at \a pos position in the list */
  template<typename ...Args>
  HSHM_CROSS_FUN
  qtok_t emplace(Args&&... args) {
    // Allocate a slot in the queue
    // The slot is marked NULL, so pop won't do anything if context switch
    qtok_id head = head_.load();
    qtok_id tail = tail_.fetch_add(1);
    size_t size = tail - head + 1;
    vector<pair<bitfield32_t, T>> &queue = (*queue_);

    // Check if there's space in the queue.
    if constexpr (!IsFixedSize) {
      if (size > queue.size()) {
        while (true) {
          head = head_.load();
          size = tail - head + 1;
          if (size <= (*queue_).size()) {
            break;
          }
          HERMES_THREAD_MODEL->Yield();
        }
      }
    }

    // Emplace into queue at our slot
    uint32_t idx = tail % queue.size();
    auto iter = queue.begin() + idx;
    queue.replace(iter,
                  hshm::PiecewiseConstruct(),
                  make_argpack(),
                  make_argpack(std::forward<Args>(args)...));

    // Let pop know that the data is fully prepared
    pair<bitfield32_t, T> &entry = (*iter);
    entry.GetFirst().SetBits(1);
    return qtok_t(tail);
  }

 public:
  /** Consumer pops the head object */
  HSHM_CROSS_FUN
  qtok_t pop(T &val) {
    // Don't pop if there's no entries
    qtok_id head = head_.load();
    qtok_id tail = tail_.load();
    if (head >= tail) {
      return qtok_t::GetNull();
    }

    // Pop the element, but only if it's marked valid
    qtok_id idx = head % (*queue_).size();
    hipc::pair<bitfield32_t, T> &entry = (*queue_)[idx];
    if (entry.GetFirst().Any(1)) {
      val = std::move(entry.GetSecond());
      entry.GetFirst().Clear();
      head_.fetch_add(1);
      return qtok_t(head);
    } else {
      return qtok_t::GetNull();
    }
  }

  /** Consumer pops the head object */
  HSHM_CROSS_FUN
  qtok_t pop() {
    // Don't pop if there's no entries
    qtok_id head = head_.load();
    qtok_id tail = tail_.load();
    if (head >= tail) {
      return qtok_t::GetNull();
    }

    // Pop the element, but only if it's marked valid
    qtok_id idx = head % (*queue_).size();
    hipc::pair<bitfield32_t, T> &entry = (*queue_)[idx];
    if (entry.GetFirst().Any(1)) {
      entry.GetFirst().Clear();
      head_.fetch_add(1);
      return qtok_t(head);
    } else {
      return qtok_t::GetNull();
    }
  }

  /** Consumer peeks an object */
  HSHM_CROSS_FUN
  qtok_t peek(T *&val, int off = 0) {
    // Don't pop if there's no entries
    qtok_id head = head_.load() + off;
    qtok_id tail = tail_.load();
    if (head >= tail) {
      return qtok_t::GetNull();
    }

    // Pop the element, but only if it's marked valid
    qtok_id idx = (head) % (*queue_).size();
    hipc::pair<bitfield32_t, T> &entry = (*queue_)[idx];
    if (entry.GetFirst().Any(1)) {
      val = &entry.GetSecond();
      return qtok_t(head);
    } else {
      return qtok_t::GetNull();
    }
  }

  /** Consumer peeks an object */
  HSHM_CROSS_FUN
  qtok_t peek(pair<bitfield32_t, T> *&val, int off = 0) {
    // Don't pop if there's no entries
    qtok_id head = head_.load() + off;
    qtok_id tail = tail_.load();
    if (head >= tail) {
      return qtok_t::GetNull();
    }

    // Pop the element, but only if it's marked valid
    qtok_id idx = (head) % (*queue_).size();
    hipc::pair<bitfield32_t, T> &entry = (*queue_)[idx];
    if (entry.GetFirst().Any(1)) {
      val = &entry;
      return qtok_t(head);
    } else {
      return qtok_t::GetNull();
    }
  }

  /** Get size at this moment */
  HSHM_CROSS_FUN
  size_t GetSize() {
    size_t tail = tail_.load();
    size_t head = head_.load();
    if (tail < head) {
      return 0;
    }
    return tail - head;
  }
};

template<typename T, HSHM_CLASS_TEMPL_WITH_DEFAULTS>
using mpsc_queue = ring_queue_base<T, true, false, false, HSHM_CLASS_TEMPL_ARGS>;

template<typename T, HSHM_CLASS_TEMPL_WITH_DEFAULTS>
using fixed_spsc_queue = ring_queue_base<T, false, false, true, HSHM_CLASS_TEMPL_ARGS>;

template<typename T, HSHM_CLASS_TEMPL_WITH_DEFAULTS>
using fixed_mpmc_queue = ring_queue_base<T, true, true, true, HSHM_CLASS_TEMPL_ARGS>;

template<typename T, HSHM_CLASS_TEMPL_WITH_DEFAULTS>
using spsc_queue = ring_queue_base<T, false, false, false, HSHM_CLASS_TEMPL_ARGS>;

}  // namespace hshm::ipc

#undef CLASS_NAME
#undef TYPED_CLASS

#endif  // HERMES_SHM_INCLUDE_HERMES_SHM_DATA_STRUCTURES_IPC_ring_queue_base_H_
