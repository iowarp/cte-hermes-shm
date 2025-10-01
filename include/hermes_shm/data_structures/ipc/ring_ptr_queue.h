//
// Created by llogan on 28/10/24.
//

#ifndef HSHM_SHM_INCLUDE_HSHM_SHM_DATA_STRUCTURES_IPC_ring_ptr_queue_base_H_
#define HSHM_SHM_INCLUDE_HSHM_SHM_DATA_STRUCTURES_IPC_ring_ptr_queue_base_H_

#include "hermes_shm/data_structures/internal/shm_internal.h"
#include "hermes_shm/thread/lock.h"
#include "hermes_shm/types/qtok.h"
#include "pair.h"
#include "ring_queue_flags.h"
#include "vector.h"

namespace hshm::ipc {

/** Forward declaration for EmptyHeader */
struct EmptyHeader;

/** Forward declaration of ring_ptr_queue_base */
template <typename T, typename HDR = EmptyHeader, RingQueueFlag RQ_FLAGS = RING_BUFFER_SPSC_FLAGS, HSHM_CLASS_TEMPL_WITH_DEFAULTS>
class ring_ptr_queue_base;

/**
 * MACROS used to simplify the ring_ptr_queue_base namespace
 * Used as inputs to the HIPC_CONTAINER_TEMPLATE
 * */
#define CLASS_NAME ring_ptr_queue_base
#define CLASS_NEW_ARGS T, HDR, RQ_FLAGS

/**
 * A queue optimized for multiple producers (emplace) with a single
 * consumer (pop).
 * @param T The type of the data to store in the queue
 * @param HDR Optional header type to store additional data (default: EmptyHeader)
 * @param RQ_FLAGS Configuration flags
 * */
template <typename T, typename HDR, RingQueueFlag RQ_FLAGS, HSHM_CLASS_TEMPL>
class ring_ptr_queue_base : public ShmContainer {
 public:
  HIPC_CONTAINER_TEMPLATE((CLASS_NAME), (CLASS_NEW_ARGS))
  RING_QUEUE_DEFS

 public:
  /**====================================
   * Typedefs
   * ===================================*/
  typedef vector<T, HSHM_CLASS_TEMPL_ARGS> vector_t;

 public:
  delay_ar<vector_t> queue_;
  hipc::opt_atomic<qtok_id, IsPushAtomic> tail_;
  hipc::opt_atomic<qtok_id, IsPopAtomic> head_;
  ibitfield flags_;
  HDR header_;

 public:
  /**====================================
   * Default Constructor
   * ===================================*/

  /** Constructor. Default. */
  HSHM_CROSS_FUN
  ring_ptr_queue_base(size_t depth = 1024) {
    shm_init(HSHM_MEMORY_MANAGER->GetDefaultAllocator<AllocT>(), depth);
  }

  /** SHM constructor. Default. */
  HSHM_CROSS_FUN
  explicit ring_ptr_queue_base(const hipc::CtxAllocator<AllocT> &alloc,
                               size_t depth = 1024) {
    shm_init(alloc, depth);
  }

  HSHM_INLINE_CROSS_FUN
  void shm_init(const hipc::CtxAllocator<AllocT> &alloc, size_t depth = 1024) {
    init_shm_container(alloc);
    queue_.shm_init(GetCtxAllocator(), depth);
    flags_.Clear();
    if constexpr (!std::is_same_v<HDR, EmptyHeader>) {
      header_ = HDR{};
    }
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
  ring_ptr_queue_base &operator=(const ring_ptr_queue_base &other) {
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
    if constexpr (!std::is_same_v<HDR, EmptyHeader>) {
      header_ = other.header_;
    }
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
    shm_move_op<false>(alloc, std::forward<ring_ptr_queue_base>(other));
  }

  /** SHM move assignment operator. */
  HSHM_CROSS_FUN
  ring_ptr_queue_base &operator=(ring_ptr_queue_base &&other) noexcept {
    if (this != &other) {
      shm_move_op<true>(other.GetCtxAllocator(),
                        std::forward<ring_ptr_queue_base>(other));
    }
    return *this;
  }

  /** Base shm move operator */
  template <bool IS_ASSIGN>
  HSHM_CROSS_FUN void shm_move_op(const hipc::CtxAllocator<AllocT> &alloc,
                                  ring_ptr_queue_base &&other) {
    if constexpr (IS_ASSIGN) {
      shm_destroy();
    } else {
      init_shm_container(alloc);
    }
    if (GetAllocator() == other.GetAllocator()) {
      head_ = other.head_.load();
      tail_ = other.tail_.load();
      (*queue_) = std::move(*other.queue_);
      if constexpr (!std::is_same_v<HDR, EmptyHeader>) {
        header_ = std::move(other.header_);
      }
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
  void shm_destroy_main() { (*queue_).shm_destroy(); }

  /** Check if the list is empty */
  HSHM_CROSS_FUN
  bool IsNull() const { return (*queue_).IsNull(); }

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
    ring_ptr_queue_base new_queue(GetCtxAllocator(), new_depth);
    T val;
    while (!pop(val).IsNull()) {
      new_queue.push(val);
    }
    (*this) = std::move(new_queue);
  }

  /** Resize (wrapper) */
  HSHM_INLINE_CROSS_FUN
  void Resize(size_t new_depth) { resize(new_depth); }

  /** Construct an element at \a pos position in the list */
  template <typename... Args>
  HSHM_CROSS_FUN qtok_t emplace(const T &val) {
    // Allocate a slot in the queue
    // The slot is marked NULL, so pop won't do anything if context switch
    qtok_id head = head_.load();
    qtok_id tail = tail_.fetch_add(1);
    vector_t &queue = (*queue_);

    // Check if there's space in the queue.
    if constexpr (WaitForSpace) {
      size_t size = tail - head + 1;
      if (size > queue.size()) {
        while (true) {
          head = head_.load();
          size = tail - head + 1;
          if (size <= GetDepth()) {
            break;
          }
          HSHM_THREAD_MODEL->Yield();
        }
      }
    } else if constexpr (ErrorOnNoSpace) {
      size_t size = tail - head + 1;
      if (size > queue.size()) {
        tail_.fetch_sub(1);
        return qtok_t::GetNull();
      }
    } else if constexpr (DynamicSize) {
      size_t size = tail - head + 1;
      if (size > queue.size()) {
        resize(queue.size() * 2);
      }
    }

    // Emplace into queue at our slot
    size_t depth = queue.size();
    uint32_t idx = tail % depth;
    Mark(val, queue[idx]);

    // Let pop know that the data is fully prepared
    return qtok_t(tail);
  }

  /** Push an elemnt in the list (wrapper) */
  template <typename... Args>
  HSHM_INLINE_CROSS_FUN qtok_t push(Args &&...args) {
    return emplace(std::forward<Args>(args)...);
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
    T &entry = (*queue_)[(size_t)idx];

    // Check if bit is marked
    bool is_marked = IsMarked(entry);

    // Complete dequeue if marked
    if (is_marked) {
      Unmark(val, entry);
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
    bool is_marked = IsMarked(entry);
    if (is_marked) {
      head_.fetch_add(1);
      return qtok_t(head);
    } else {
      return qtok_t::GetNull();
    }
  }

  /** Consumer pops the tail object */
  HSHM_CROSS_FUN
  qtok_t pop_back(T &val) {
    // Don't pop if there's no entries
    qtok_id head = head_.load();
    qtok_id tail = tail_.load();
    if (head >= tail) {
      return qtok_t::GetNull();
    }
    tail -= 1;

    // Pop the element at tail
    qtok_id idx = tail % (*queue_).size();
    T &entry = (*queue_)[idx];

    // Check if bit is marked
    bool is_marked = IsMarked(entry);

    // Complete dequeue if marked
    if (is_marked) {
      Unmark(val, entry);
      tail_.fetch_sub(1);
      return qtok_t(tail);
    } else {
      return qtok_t::GetNull();
    }
  }

  /** Mark an entry */
  HSHM_INLINE_CROSS_FUN
  void Mark(const T &val, T &entry) {
    if constexpr (std::is_arithmetic_v<T>) {
      entry = MARK_FIRST_BIT(T, val);
    } else if constexpr (std::is_pointer_v<T>) {
      entry = (T)MARK_FIRST_BIT(size_t, (size_t)val);
    } else if constexpr (IS_SHM_POINTER(T)) {
      entry = val.Mark();
    } else {
      STATIC_ASSERT(false, "Unsupported type", T);
    }
  }

  /** Check if a pointer is marked */
  HSHM_INLINE_CROSS_FUN
  bool IsMarked(T &entry) {
    if constexpr (std::is_arithmetic_v<T>) {
      return IS_FIRST_BIT_MARKED(T, entry);
    } else if constexpr (std::is_pointer_v<T>) {
      return IS_FIRST_BIT_MARKED(size_t, (size_t)entry);
    } else if constexpr (IS_SHM_POINTER(T)) {
      return entry.IsMarked();
    } else {
      STATIC_ASSERT(false, "Unsupported type", T);
    }
  }

  /** Unmark pointer */
  HSHM_INLINE_CROSS_FUN
  void Unmark(T &val, T &entry) {
    if constexpr (std::is_arithmetic<T>::value) {
      val = UNMARK_FIRST_BIT(T, entry);
      entry = 0;
    } else if constexpr (std::is_pointer_v<T>) {
      val = (T)UNMARK_FIRST_BIT(size_t, (size_t)entry);
      entry = 0;
    } else if constexpr (IS_SHM_POINTER(T)) {
      val = entry.Unmark();
      entry.SetZero();
    } else {
      STATIC_ASSERT(false, "Unsupported type", T);
    }
  }

  /** Get queue depth */
  HSHM_INLINE_CROSS_FUN
  size_t GetDepth() { return queue_->size(); }

  /** Get size at this moment */
  HSHM_INLINE_CROSS_FUN
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
  size_t size() { return GetSize(); }

  /** Get size (wrapper) */
  HSHM_INLINE_CROSS_FUN
  size_t Size() { return GetSize(); }

  /** Get header reference */
  HSHM_INLINE_CROSS_FUN
  HDR& GetHeader() { return header_; }

  /** Get const header reference */
  HSHM_INLINE_CROSS_FUN
  const HDR& GetHeader() const { return header_; }
};

template <typename T, typename HDR = EmptyHeader, HSHM_CLASS_TEMPL_WITH_DEFAULTS>
using mpsc_ptr_queue =
    ring_ptr_queue_base<T, HDR, RING_BUFFER_MPSC_FLAGS, HSHM_CLASS_TEMPL_ARGS>;

template <typename T, typename HDR = EmptyHeader, HSHM_CLASS_TEMPL_WITH_DEFAULTS>
using spsc_ptr_queue =
    ring_ptr_queue_base<T, HDR, RING_BUFFER_SPSC_FLAGS, HSHM_CLASS_TEMPL_ARGS>;

template <typename T, typename HDR = EmptyHeader, HSHM_CLASS_TEMPL_WITH_DEFAULTS>
using fixed_spsc_ptr_queue =
    ring_ptr_queue_base<T, HDR, RING_BUFFER_FIXED_SPSC_FLAGS, HSHM_CLASS_TEMPL_ARGS>;

template <typename T, typename HDR = EmptyHeader, HSHM_CLASS_TEMPL_WITH_IPC_DEFAULTS>
using fixed_mpsc_ptr_queue =
    ring_ptr_queue_base<T, HDR, RING_BUFFER_FIXED_MPMC_FLAGS, HSHM_CLASS_TEMPL_ARGS>;

template <typename T, typename HDR = EmptyHeader, HSHM_CLASS_TEMPL_WITH_IPC_DEFAULTS>
using circular_spsc_ptr_queue =
    ring_ptr_queue_base<T, HDR, RING_BUFFER_CIRCULAR_SPSC_FLAGS,
                        HSHM_CLASS_TEMPL_ARGS>;

template <typename T, typename HDR = EmptyHeader, HSHM_CLASS_TEMPL_WITH_IPC_DEFAULTS>
using circular_mpsc_ptr_queue =
    ring_ptr_queue_base<T, HDR, RING_BUFFER_CIRCULAR_MPMC_FLAGS,
                        HSHM_CLASS_TEMPL_ARGS>;

template <typename T, typename HDR = EmptyHeader, HSHM_CLASS_TEMPL_WITH_IPC_DEFAULTS>
using ext_ptr_ring_buffer =
    ring_ptr_queue_base<T, HDR, RING_BUFFER_EXTENSIBLE_FLAGS, HSHM_CLASS_TEMPL_ARGS>;

}  // namespace hshm::ipc

namespace hshm {

template <typename T, typename HDR = hipc::EmptyHeader, HSHM_CLASS_TEMPL_WITH_PRIV_DEFAULTS>
using mpsc_ptr_queue =
    hipc::ring_ptr_queue_base<T, HDR, RING_BUFFER_MPSC_FLAGS, HSHM_CLASS_TEMPL_ARGS>;

template <typename T, typename HDR = hipc::EmptyHeader, HSHM_CLASS_TEMPL_WITH_PRIV_DEFAULTS>
using spsc_ptr_queue =
    hipc::ring_ptr_queue_base<T, HDR, RING_BUFFER_SPSC_FLAGS, HSHM_CLASS_TEMPL_ARGS>;

template <typename T, typename HDR = hipc::EmptyHeader, HSHM_CLASS_TEMPL_WITH_PRIV_DEFAULTS>
using fixed_spsc_ptr_queue =
    hipc::ring_ptr_queue_base<T, HDR, RING_BUFFER_FIXED_SPSC_FLAGS,
                              HSHM_CLASS_TEMPL_ARGS>;

template <typename T, typename HDR = hipc::EmptyHeader, HSHM_CLASS_TEMPL_WITH_PRIV_DEFAULTS>
using fixed_mpsc_ptr_queue =
    hipc::ring_ptr_queue_base<T, HDR, RING_BUFFER_FIXED_MPMC_FLAGS,
                              HSHM_CLASS_TEMPL_ARGS>;

template <typename T, typename HDR = hipc::EmptyHeader, HSHM_CLASS_TEMPL_WITH_PRIV_DEFAULTS>
using circular_spsc_ptr_queue =
    hipc::ring_ptr_queue_base<T, HDR, RING_BUFFER_CIRCULAR_SPSC_FLAGS,
                              HSHM_CLASS_TEMPL_ARGS>;

template <typename T, typename HDR = hipc::EmptyHeader, HSHM_CLASS_TEMPL_WITH_PRIV_DEFAULTS>
using circular_mpsc_ptr_queue =
    hipc::ring_ptr_queue_base<T, HDR, RING_BUFFER_CIRCULAR_MPMC_FLAGS,
                              HSHM_CLASS_TEMPL_ARGS>;

template <typename T, typename HDR = hipc::EmptyHeader, HSHM_CLASS_TEMPL_WITH_PRIV_DEFAULTS>
using ext_ptr_ring_buffer =
    hipc::ring_ptr_queue_base<T, HDR, RING_BUFFER_EXTENSIBLE_FLAGS,
                              HSHM_CLASS_TEMPL_ARGS>;

}  // namespace hshm

#undef CLASS_NAME
#undef CLASS_NEW_ARGS

#endif  // HSHM_SHM_INCLUDE_HSHM_SHM_DATA_STRUCTURES_IPC_ring_ptr_queue_base_H_
