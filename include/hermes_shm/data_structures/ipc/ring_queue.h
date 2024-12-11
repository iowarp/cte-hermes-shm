//
// Created by llogan on 28/10/24.
//

#ifndef HERMES_SHM_INCLUDE_HERMES_SHM_DATA_STRUCTURES_IPC_ring_queue_base_H_
#define HERMES_SHM_INCLUDE_HERMES_SHM_DATA_STRUCTURES_IPC_ring_queue_base_H_


#include "hermes_shm/data_structures/internal/shm_internal.h"
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
#define CLASS_NEW_ARGS \
  T, IsPushAtomic, IsPopAtomic, IsFixedSize

/**
 * A queue optimized for multiple producers (emplace) with a single
 * consumer (pop).
 * @param T The type of the data to store in the queue
 * @param IsPushAtomic If true, the push operation is atomic (thread-safe)
 * @param IsPopAtomic If true, the pop operation is atomic (thread-safe)
 * @param IsFixedSize If true, the queue is guaranteed to push/pop a fixed number of requests.
 * */
template<
    typename T,
    bool IsPushAtomic,
    bool IsPopAtomic,
    bool IsFixedSize,
    HSHM_CLASS_TEMPL>
class ring_queue_base : public ShmContainer {
 public:
  HIPC_CONTAINER_TEMPLATE((CLASS_NAME), (CLASS_NEW_ARGS))

 public:
  /**====================================
   * Typedefs
   * ===================================*/
  typedef pair<bitfield32_t, T, HSHM_CLASS_TEMPL_ARGS> pair_t;
  typedef vector<pair_t, HSHM_CLASS_TEMPL_ARGS> vector_t;

 public:
  /**====================================
   * Variables
   * ===================================*/
  delay_ar<vector_t> queue_;
  hipc::opt_atomic<qtok_id, IsPushAtomic> tail_;
  hipc::opt_atomic<qtok_id, IsPopAtomic> head_;
  bitfield32_t flags_;

 public:
  /**====================================
   * Default Constructor
   * ===================================*/

  /** Constructor. Default. */
  template<typename ...Args>
  HSHM_CROSS_FUN
  explicit ring_queue_base(size_t depth = 1024, Args &&...args) {
    shm_init(HERMES_MEMORY_MANAGER->GetDefaultAllocator<AllocT>(), depth,
             std::forward<Args>(args)...);
  }

  /** SHM constructor. Default. */
  template<typename ...Args>
  HSHM_CROSS_FUN
  explicit ring_queue_base(const hipc::CtxAllocator<AllocT> &alloc,
                           size_t depth = 1024,
                           Args &&...args) {
    shm_init(alloc, depth, std::forward<Args>(args)...);
  }

  /** SHM Constructor */
  template<typename ...Args>
  HSHM_CROSS_FUN
  void shm_init(const hipc::CtxAllocator<AllocT> &alloc,
                size_t depth = 1024,
                Args &&...args) {
    init_shm_container(alloc);
    HSHM_MAKE_AR(queue_, GetCtxAllocator(), depth,
                 std::forward<Args>(args)...);
    flags_.Clear();
    SetNull();
  }

  /**====================================
   * Copy Constructors
   * ===================================*/

  /** SHM copy constructor */
  HSHM_CROSS_FUN
  explicit ring_queue_base(
      const hipc::CtxAllocator<AllocT> &alloc,
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
    shm_move_op<false>(other.GetCtxAllocator(),
                       std::forward<ring_queue_base>(other));
  }

  /** SHM move constructor. */
  HSHM_CROSS_FUN
  ring_queue_base(const hipc::CtxAllocator<AllocT> &alloc,
             ring_queue_base &&other) noexcept {
    shm_move_op<false>(alloc, std::forward<ring_queue_base>(other));
  }

  /** SHM move assignment operator. */
  HSHM_CROSS_FUN
  ring_queue_base& operator=(ring_queue_base &&other) noexcept {
    if (this != &other) {
      shm_move_op<true>(other.GetCtxAllocator(),
                        std::forward<ring_queue_base>(other));
    }
    return *this;
  }

  /** SHM move assignment operator. */
  template<bool IS_ASSIGN>
  HSHM_CROSS_FUN
  void shm_move_op(const hipc::CtxAllocator<AllocT> &alloc, ring_queue_base &&other) noexcept {
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

  /** Resize */
  HSHM_CROSS_FUN
  void resize(size_t new_depth) {
    queue_->resize(new_depth);
  }

  /** Resize (wrapper) */
  HSHM_INLINE_CROSS
  void Resize(size_t new_depth) {
    resize(new_depth);
  }

  /** Construct an element at \a pos position in the list */
  template<typename ...Args>
  HSHM_CROSS_FUN
  qtok_t emplace(Args&&... args) {
    // Allocate a slot in the queue
    // The slot is marked NULL, so pop won't do anything if context switch
    qtok_id head = head_.load();
    qtok_id tail;
    vector_t &queue = (*queue_);

    // Get the tail
    if constexpr (IsPushAtomic) {
      tail = tail_.fetch_add(1);
    } else {
      tail = tail_.load();
    }

    // Check if there's space in the queue.
    if constexpr (!IsFixedSize) {
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
    } else if constexpr (!IsPushAtomic) {
      size_t size = tail - head + 1;
      if (size > queue.size()) {
        return qtok_t::GetNull();
      }
    }

    // Increment tail
    if constexpr (!IsPushAtomic) {
      tail_.fetch_add(1);
    }

    // Emplace into queue at our slot
    uint32_t idx = tail % queue.size();
    auto iter = queue.begin() + idx;
    queue.replace(iter,
                  hshm::PiecewiseConstruct(),
                  make_argpack(),
                  make_argpack(std::forward<Args>(args)...));

    // Let pop know that the data is fully prepared
    pair_t &entry = (*iter);
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
    pair_t &entry = (*queue_)[idx];
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
    pair_t &entry = (*queue_)[idx];
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
    pair_t &entry = (*queue_)[idx];
    if (entry.GetFirst().Any(1)) {
      val = &entry.GetSecond();
      return qtok_t(head);
    } else {
      return qtok_t::GetNull();
    }
  }

  /** Consumer peeks an object */
  HSHM_CROSS_FUN
  qtok_t peek(pair_t *&val, int off = 0) {
    // Don't pop if there's no entries
    qtok_id head = head_.load() + off;
    qtok_id tail = tail_.load();
    if (head >= tail) {
      return qtok_t::GetNull();
    }

    // Pop the element, but only if it's marked valid
    qtok_id idx = (head) % (*queue_).size();
    pair_t &entry = (*queue_)[idx];
    if (entry.GetFirst().Any(1)) {
      val = &entry;
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
  HSHM_INLINE_CROSS
  size_t size() {
    return GetSize();
  }

  /** Get size (wrapper) */
  HSHM_INLINE_CROSS
  size_t Size() {
    return GetSize();
  }
};

// bool IsPushAtomic,
// bool IsPopAtomic,
// bool IsFixedSize,
#define RING_BUFFER_FLAGS bool IsPushAtomic, bool IsPopAtomic, bool IsFixedSize
#define RING_BUFFER_FLAGS_ARGS IsPushAtomic, IsPopAtomic, IsFixedSize
#define RING_BUFFER_MPSC_FLAGS true, false, false
#define RING_BUFFER_SPSC_FLAGS false, false, false
#define RING_BUFFER_FIXED_SPSC_FLAGS false, false, true
#define RING_BUFFER_FIXED_MPMC_FLAGS true, true, true

template<typename T, HSHM_CLASS_TEMPL_WITH_DEFAULTS>
using mpsc_queue = ring_queue_base<T, RING_BUFFER_MPSC_FLAGS, HSHM_CLASS_TEMPL_ARGS>;

template<typename T, HSHM_CLASS_TEMPL_WITH_DEFAULTS>
using spsc_queue = ring_queue_base<T, RING_BUFFER_SPSC_FLAGS, HSHM_CLASS_TEMPL_ARGS>;

template<typename T, HSHM_CLASS_TEMPL_WITH_DEFAULTS>
using fixed_spsc_queue = ring_queue_base<T, RING_BUFFER_FIXED_SPSC_FLAGS, HSHM_CLASS_TEMPL_ARGS>;

template<typename T, HSHM_CLASS_TEMPL_WITH_DEFAULTS>
using fixed_mpmc_queue = ring_queue_base<T, RING_BUFFER_FIXED_MPMC_FLAGS, HSHM_CLASS_TEMPL_ARGS>;

template<typename T, HSHM_CLASS_TEMPL_WITH_PRIV_DEFAULTS>
using fixed_mpsc_queue = fixed_mpmc_queue<T, HSHM_CLASS_TEMPL_ARGS>;

}  // namespace hshm::ipc

namespace hshm {

template<typename T, HSHM_CLASS_TEMPL_WITH_PRIV_DEFAULTS>
using mpsc_queue = hipc::ring_queue_base<T, RING_BUFFER_MPSC_FLAGS, HSHM_CLASS_TEMPL_ARGS>;

template<typename T, HSHM_CLASS_TEMPL_WITH_PRIV_DEFAULTS>
using spsc_queue = hipc::ring_queue_base<T, RING_BUFFER_SPSC_FLAGS, HSHM_CLASS_TEMPL_ARGS>;

template<typename T, HSHM_CLASS_TEMPL_WITH_PRIV_DEFAULTS>
using fixed_spsc_queue = hipc::ring_queue_base<T, RING_BUFFER_FIXED_SPSC_FLAGS, HSHM_CLASS_TEMPL_ARGS>;

template<typename T, HSHM_CLASS_TEMPL_WITH_PRIV_DEFAULTS>
using fixed_mpmc_queue = hipc::ring_queue_base<T, RING_BUFFER_FIXED_MPMC_FLAGS, HSHM_CLASS_TEMPL_ARGS>;

template<typename T, HSHM_CLASS_TEMPL_WITH_PRIV_DEFAULTS>
using fixed_mpsc_queue = fixed_mpmc_queue<T, HSHM_CLASS_TEMPL_ARGS>;

}  // namespace hshm

#undef CLASS_NAME
#undef CLASS_NEW_ARGS

#endif  // HERMES_SHM_INCLUDE_HERMES_SHM_DATA_STRUCTURES_IPC_ring_queue_base_H_
