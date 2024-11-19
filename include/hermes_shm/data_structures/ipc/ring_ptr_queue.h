//
// Created by llogan on 28/10/24.
//

#ifndef HERMES_SHM_INCLUDE_HERMES_SHM_DATA_STRUCTURES_IPC_ring_ptr_queue_base_H_
#define HERMES_SHM_INCLUDE_HERMES_SHM_DATA_STRUCTURES_IPC_ring_ptr_queue_base_H_

#include "hermes_shm/data_structures/ipc/internal/shm_internal.h"
#include "hermes_shm/thread/lock.h"
#include "vector.h"
#include "pair.h"
#include "hermes_shm/types/qtok.h"

namespace hshm::ipc {

/** Forward declaration of ring_ptr_queue_base */
template<
    typename T,
    bool IsPushAtomic,
    bool IsPopAtomic,
    bool IsFixedSize,
    HSHM_CLASS_TEMPL_WITH_DEFAULTS>
class ring_ptr_queue_base;

/**
 * MACROS used to simplify the ring_ptr_queue_base namespace
 * Used as inputs to the HIPC_CONTAINER_TEMPLATE
 * */
#define CLASS_NAME ring_ptr_queue_base
#define CLASS_NEW_ARGS \
  T, IsPushAtomic, IsPopAtomic, IsFixedSize

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
class ring_ptr_queue_base : public ShmContainer {
 public:
  HIPC_CONTAINER_TEMPLATE((CLASS_NAME), (CLASS_NEW_ARGS))

 public:
  /**====================================
   * Typedefs
   * ===================================*/
  typedef vector<T, HSHM_CLASS_TEMPL_ARGS> vector_t;

 public:
  delay_ar<vector_t> queue_;
  hipc::opt_atomic<qtok_id, IsPushAtomic> tail_;
  hipc::opt_atomic<qtok_id, IsPopAtomic> head_;
  bitfield32_t flags_;

 public:
  /**====================================
   * Default Constructor
   * ===================================*/

  /** Constructor. Default. */
  HSHM_CROSS_FUN
  ring_ptr_queue_base(size_t depth = 1024) {
    shm_init(HERMES_MEMORY_MANAGER->GetDefaultAllocator<AllocT>(), depth);
  }

  /** SHM constructor. Default. */
  HSHM_CROSS_FUN
  explicit ring_ptr_queue_base(const hipc::CtxAllocator<AllocT> &alloc,
                          size_t depth = 1024) {
    shm_init(alloc, depth);
  }

  HSHM_INLINE_CROSS_FUN
  void shm_init(const hipc::CtxAllocator<AllocT> &alloc,
                size_t depth = 1024) {
    init_shm_container(alloc);
    HSHM_MAKE_AR(queue_, GetCtxAllocator(), depth);
    flags_.Clear();
    SetNull();
  }

  /**====================================
   * Copy Constructors
   * ===================================*/

  /** SHM copy constructor */
  HSHM_CROSS_FUN
  explicit ring_ptr_queue_base(const ring_ptr_queue_base &other) {
    init_shm_container(other.GetCtxAllocator());
    SetNull();
    shm_strong_copy_op(other);
  }

  /** SHM copy constructor */
  HSHM_CROSS_FUN
  explicit ring_ptr_queue_base(const hipc::CtxAllocator<AllocT> &alloc,
                          const ring_ptr_queue_base &other) {
    init_shm_container(alloc);
    SetNull();
    shm_strong_copy_op(other);
  }

  /** SHM copy assignment operator */
  HSHM_CROSS_FUN
  ring_ptr_queue_base& operator=(const ring_ptr_queue_base &other) {
    if (this != &other) {
      shm_destroy();
      shm_strong_copy_op(other);
    }
    return *this;
  }

  /** SHM copy constructor + operator main */
  HSHM_CROSS_FUN
  void shm_strong_copy_op(const ring_ptr_queue_base &other) {
    head_ = other.head_.load();
    tail_ = other.tail_.load();
    (*queue_) = (*other.queue_);
  }

  /**====================================
   * Move Constructors
   * ===================================*/

  /** Move constructor. */
  HSHM_CROSS_FUN
  ring_ptr_queue_base(ring_ptr_queue_base &&other) noexcept {
    shm_move_op<false>(other.GetCtxAllocator(),
                       std::forward<ring_ptr_queue_base>(other));
  }

  /** SHM move constructor. */
  HSHM_CROSS_FUN
  ring_ptr_queue_base(const hipc::CtxAllocator<AllocT> &alloc,
                 ring_ptr_queue_base &&other) noexcept {
    shm_move_op<false>(alloc,
                       std::forward<ring_ptr_queue_base>(other));
  }

  /** SHM move assignment operator. */
  HSHM_CROSS_FUN
  ring_ptr_queue_base& operator=(ring_ptr_queue_base &&other) noexcept {
    if (this != &other) {
      shm_move_op<true>(other.GetCtxAllocator(),
                        std::forward<ring_ptr_queue_base>(other));
    }
    return *this;
  }

  /** Base shm move operator */
  template<bool IS_ASSIGN>
  HSHM_CROSS_FUN
  void shm_move_op(const hipc::CtxAllocator<AllocT> &alloc, ring_ptr_queue_base &&other) {
    if constexpr (IS_ASSIGN) {
      shm_destroy();
    } else {
      init_shm_container(alloc);
    }
    if (GetCtxAllocator() == other.GetCtxAllocator()) {
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

  /** Resize */
  HSHM_CROSS_FUN
  void resize(size_t new_depth) {
    queue_->resize(new_depth);
  }

  /** Resize (wrapper) */
  HSHM_INLINE_CROSS_FUN
  void Resize(size_t new_depth) {
    resize(new_depth);
  }

  /** Construct an element at \a pos position in the list */
  template<typename ...Args>
  HSHM_CROSS_FUN
  qtok_t emplace(const T &val) {
    // Allocate a slot in the queue
    // The slot is marked NULL, so pop won't do anything if context switch
    qtok_id head = head_.load();
    qtok_id tail = tail_.fetch_add(1);
    vector_t &queue = (*queue_);

    // Check if there's space in the queue.
    if constexpr (IsFixedSize) {
      size_t size = tail - head + 1;
      if (size > queue.size()) {
        while (true) {
          head = head_.load();
          size = tail - head + 1;
          if (size <= GetDepth()) {
            break;
          }
          HERMES_THREAD_MODEL->Yield();
        }
      }
    }

    // Emplace into queue at our slot
    uint32_t idx = tail % queue.size();
    if constexpr(std::is_arithmetic<T>::value) {
      queue[idx] = MARK_FIRST_BIT(T, val);
    } else if constexpr(IS_SHM_OFFSET_POINTER(T)) {
      queue[idx] = T(MARK_FIRST_BIT(size_t, val.off_.load()));
    } else if constexpr(IS_SHM_POINTER(T)) {
      queue[idx] = T(val.allocator_id_,
                     MARK_FIRST_BIT(size_t, val.off_.load()));
    }

    // Let pop know that the data is fully prepared
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
    T &entry = (*queue_)[idx];

    // Check if bit is marked
    bool is_marked;
    if constexpr(std::is_arithmetic<T>::value) {
      is_marked = IS_FIRST_BIT_MARKED(T, entry);
    } else {
      is_marked = IS_FIRST_BIT_MARKED(size_t, entry.off_.load());
    }

    // Complete dequeue if marked
    if (is_marked) {
      if constexpr(std::is_arithmetic<T>::value) {
        val = UNMARK_FIRST_BIT(T, entry);
        entry = 0;
      } else if constexpr(IS_SHM_OFFSET_POINTER(T)) {
        val = T(UNMARK_FIRST_BIT(size_t, entry.off_.load()));
        entry.off_ = 0;
      } else if constexpr(IS_SHM_POINTER(T)) {
        val = T(entry.allocator_id_,
                UNMARK_FIRST_BIT(size_t, entry.off_.load()));
        entry.off_ = 0;
      }
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
    T &entry = (*queue_)[idx];

    // Check if bit is marked
    bool is_marked;
    if constexpr(std::is_arithmetic<T>::value) {
      is_marked = IS_FIRST_BIT_MARKED(T, entry);
    } else {
      is_marked = IS_FIRST_BIT_MARKED(size_t, entry.off_.load());
    }

    if (is_marked) {
      head_.fetch_add(1);
      return qtok_t(head);
    } else {
      return qtok_t::GetNull();
    }
  }

  /** Get queue depth */
  HSHM_CROSS_FUN
  size_t GetDepth() {
    return queue_->size();
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

  /** Get size (wrapper) */
  HSHM_INLINE_CROSS_FUN
  size_t size() {
    return GetSize();
  }

  /** Get size (wrapper) */
  HSHM_INLINE_CROSS_FUN
  size_t Size() {
    return GetSize();
  }
};

template<typename T, HSHM_CLASS_TEMPL_WITH_DEFAULTS>
using mpsc_ptr_queue = ring_ptr_queue_base<T, true, false, false, HSHM_CLASS_TEMPL_ARGS>;

template<typename T, HSHM_CLASS_TEMPL_WITH_DEFAULTS>
using fixed_spsc_ptr_queue = ring_ptr_queue_base<T, false, false, true, HSHM_CLASS_TEMPL_ARGS>;

template<typename T, HSHM_CLASS_TEMPL_WITH_DEFAULTS>
using fixed_mpmc_ptr_queue = ring_ptr_queue_base<T, true, true, true, HSHM_CLASS_TEMPL_ARGS>;

template<typename T, HSHM_CLASS_TEMPL_WITH_DEFAULTS>
using spsc_ptr_queue = ring_ptr_queue_base<T, false, false, false, HSHM_CLASS_TEMPL_ARGS>;

}  // namespace hshm::ipc

#undef CLASS_NAME
#undef CLASS_NEW_ARGS

#endif  // HERMES_SHM_INCLUDE_HERMES_SHM_DATA_STRUCTURES_IPC_ring_ptr_queue_base_H_
