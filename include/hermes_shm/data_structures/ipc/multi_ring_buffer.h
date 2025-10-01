#ifndef HERMES_SHM_DATA_STRUCTURES_MULTI_RING_BUFFER_H_
#define HERMES_SHM_DATA_STRUCTURES_MULTI_RING_BUFFER_H_

#include <atomic>

#include "hermes_shm/types/qtok.h"
#include "ring_queue.h"
#include "vector.h"

namespace hshm::ipc {

/**
 * Multi-lane concurrent ring buffer with priority levels
 * Structure: [lane][priority][queue]
 *
 * @tparam T The type of data stored in the queues
 * @tparam HDR Optional header type to store additional data (default: EmptyHeader)
 * @tparam RQ_FLAGS Ring queue configuration flags
 * @tparam HSHM_CLASS_TEMPL Template parameters for SHM containers
 */
template <typename T, typename HDR = EmptyHeader, RingQueueFlag RQ_FLAGS = RING_BUFFER_MPSC_FLAGS, HSHM_CLASS_TEMPL_WITH_DEFAULTS>
class multi_ring_buffer : public ShmContainer {
 public:
  HIPC_CONTAINER_TEMPLATE((multi_ring_buffer), (T, HDR, RQ_FLAGS))

 public:
  /**====================================
   * Typedefs
   * ===================================*/
  typedef ring_queue_base<T, HDR, RQ_FLAGS, HSHM_CLASS_TEMPL_ARGS> queue_t;
  typedef vector<queue_t, HSHM_CLASS_TEMPL_ARGS> queue_vector_t;

 private:
  /**====================================
   * Variables
   * ===================================*/
  delay_ar<queue_vector_t> queues_;
  std::atomic<size_t> round_robin_counter_;
  size_t num_lanes_;
  size_t num_priorities_;
  ibitfield flags_;
  HDR header_;

  /**====================================
   * Helper Methods
   * ===================================*/

  /** Calculate the index for a given lane and priority */
  HSHM_INLINE_CROSS_FUN
  size_t GetQueueIndex(size_t lane_id, size_t priority) const {
    return lane_id * num_priorities_ + priority;
  }

 public:
  /**====================================
   * Constructors
   * ===================================*/

  /** Default constructor */
  template <typename... Args>
  HSHM_CROSS_FUN explicit multi_ring_buffer(size_t num_lanes,
                                            size_t num_priorities,
                                            size_t queue_depth = 1024,
                                            Args &&...args) {
    shm_init(HSHM_MEMORY_MANAGER->GetDefaultAllocator<AllocT>(), num_lanes,
             num_priorities, queue_depth, std::forward<Args>(args)...);
  }

  /** SHM constructor */
  template <typename... Args>
  HSHM_CROSS_FUN explicit multi_ring_buffer(
      const hipc::CtxAllocator<AllocT> &alloc, size_t num_lanes,
      size_t num_priorities, size_t queue_depth = 1024, Args &&...args) {
    shm_init(alloc, num_lanes, num_priorities, queue_depth,
             std::forward<Args>(args)...);
  }

  /** SHM initializer */
  template <typename... Args>
  HSHM_CROSS_FUN void shm_init(const hipc::CtxAllocator<AllocT> &alloc,
                               size_t num_lanes, size_t num_priorities,
                               size_t queue_depth = 1024, Args &&...args) {
    init_shm_container(alloc);

    // Store runtime dimensions
    num_lanes_ = num_lanes;
    num_priorities_ = num_priorities;

    // Initialize the single queue vector with size = lanes * priorities
    const size_t total_queues = num_lanes_ * num_priorities_;
    queues_.shm_init(GetCtxAllocator(), total_queues);

    // Initialize each queue
    for (size_t i = 0; i < total_queues; ++i) {
      (*queues_)[i].shm_init(GetCtxAllocator(), queue_depth,
                             std::forward<Args>(args)...);
    }

    round_robin_counter_.store(0);
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
  explicit multi_ring_buffer(const hipc::CtxAllocator<AllocT> &alloc,
                             const multi_ring_buffer &other) {
    init_shm_container(alloc);
    SetNull();
    shm_strong_copy_op(other);
  }

  /** SHM copy assignment operator */
  HSHM_CROSS_FUN
  multi_ring_buffer &operator=(const multi_ring_buffer &other) {
    if (this != &other) {
      shm_destroy();
      shm_strong_copy_op(other);
    }
    return *this;
  }

  /** SHM copy constructor + operator main */
  HSHM_CROSS_FUN
  void shm_strong_copy_op(const multi_ring_buffer &other) {
    round_robin_counter_.store(other.round_robin_counter_.load());
    num_lanes_ = other.num_lanes_;
    num_priorities_ = other.num_priorities_;
    (*queues_) = (*other.queues_);
    if constexpr (!std::is_same_v<HDR, EmptyHeader>) {
      header_ = other.header_;
    }
  }

  /**====================================
   * Move Constructors
   * ===================================*/

  /** Move constructor */
  HSHM_CROSS_FUN
  multi_ring_buffer(multi_ring_buffer &&other) noexcept {
    shm_move_op<false>(other.GetCtxAllocator(),
                       std::forward<multi_ring_buffer>(other));
  }

  /** SHM move constructor */
  HSHM_CROSS_FUN
  multi_ring_buffer(const hipc::CtxAllocator<AllocT> &alloc,
                    multi_ring_buffer &&other) noexcept {
    shm_move_op<false>(alloc, std::forward<multi_ring_buffer>(other));
  }

  /** SHM move assignment operator */
  HSHM_CROSS_FUN
  multi_ring_buffer &operator=(multi_ring_buffer &&other) noexcept {
    if (this != &other) {
      shm_move_op<true>(other.GetCtxAllocator(),
                        std::forward<multi_ring_buffer>(other));
    }
    return *this;
  }

  /** SHM move assignment operator implementation */
  template <bool IS_ASSIGN>
  HSHM_CROSS_FUN void shm_move_op(const hipc::CtxAllocator<AllocT> &alloc,
                                  multi_ring_buffer &&other) noexcept {
    if constexpr (!IS_ASSIGN) {
      init_shm_container(alloc);
    }
    if (GetAllocator() == other.GetAllocator()) {
      round_robin_counter_.store(other.round_robin_counter_.load());
      num_lanes_ = other.num_lanes_;
      num_priorities_ = other.num_priorities_;
      (*queues_) = std::move(*other.queues_);
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

  /** SHM destructor */
  HSHM_CROSS_FUN
  void shm_destroy_main() { (*queues_).shm_destroy(); }

  /** Check if the buffer is empty */
  HSHM_CROSS_FUN
  bool IsNull() const { return (*queues_).IsNull(); }

  /** Sets this buffer as empty */
  HSHM_CROSS_FUN
  void SetNull() { round_robin_counter_.store(0); }

  /**====================================
   * Multi-Ring Buffer Methods
   * ===================================*/


  /**
   * Get direct access to a specific lane (priority-specific queue)
   */
  HSHM_CROSS_FUN
  queue_t &GetLane(size_t lane_id, size_t priority) {
    size_t queue_idx = GetQueueIndex(lane_id, priority);
    return (*queues_)[queue_idx];
  }

  /**
   * Get direct access to a specific lane (const version)
   */
  HSHM_CROSS_FUN
  const queue_t &GetLane(size_t lane_id, size_t priority) const {
    size_t queue_idx = GetQueueIndex(lane_id, priority);
    return (*queues_)[queue_idx];
  }

  /**
   * Get the number of lanes
   */
  HSHM_CROSS_FUN
  size_t GetNumLanes() const { return num_lanes_; }

  /**
   * Get the number of priorities
   */
  HSHM_CROSS_FUN
  size_t GetNumPriorities() const { return num_priorities_; }

  /** Get header reference */
  HSHM_INLINE_CROSS_FUN
  HDR& GetHeader() { return header_; }

  /** Get const header reference */
  HSHM_INLINE_CROSS_FUN
  const HDR& GetHeader() const { return header_; }
};

// Multi-lane MPSC queue
template <typename T, typename HDR = EmptyHeader, HSHM_CLASS_TEMPL_WITH_IPC_DEFAULTS>
using multi_mpsc_queue =
    multi_ring_buffer<T, HDR, RING_BUFFER_MPSC_FLAGS, HSHM_CLASS_TEMPL_ARGS>;

// Multi-lane SPSC queue
template <typename T, typename HDR = EmptyHeader, HSHM_CLASS_TEMPL_WITH_IPC_DEFAULTS>
using multi_spsc_queue =
    multi_ring_buffer<T, HDR, RING_BUFFER_SPSC_FLAGS, HSHM_CLASS_TEMPL_ARGS>;

// Multi-lane fixed MPSC queue
template <typename T, typename HDR = EmptyHeader, HSHM_CLASS_TEMPL_WITH_IPC_DEFAULTS>
using multi_fixed_mpsc_queue =
    multi_ring_buffer<T, HDR, RING_BUFFER_FIXED_MPMC_FLAGS,
                      HSHM_CLASS_TEMPL_ARGS>;

// Multi-lane circular MPSC queue
template <typename T, typename HDR = EmptyHeader, HSHM_CLASS_TEMPL_WITH_IPC_DEFAULTS>
using multi_circular_mpsc_queue =
    multi_ring_buffer<T, HDR, RING_BUFFER_CIRCULAR_MPMC_FLAGS,
                      HSHM_CLASS_TEMPL_ARGS>;

}  // namespace hshm::ipc

namespace hshm {

/**
 * Convenient typedefs for common multi-ring buffer configurations
 */

// Multi-lane MPSC queue
template <typename T, typename HDR = hipc::EmptyHeader, HSHM_CLASS_TEMPL_WITH_PRIV_DEFAULTS>
using multi_mpsc_queue =
    hipc::multi_ring_buffer<T, HDR, RING_BUFFER_MPSC_FLAGS, HSHM_CLASS_TEMPL_ARGS>;

// Multi-lane SPSC queue
template <typename T, typename HDR = hipc::EmptyHeader, HSHM_CLASS_TEMPL_WITH_PRIV_DEFAULTS>
using multi_spsc_queue =
    hipc::multi_ring_buffer<T, HDR, RING_BUFFER_SPSC_FLAGS, HSHM_CLASS_TEMPL_ARGS>;

// Multi-lane fixed MPSC queue
template <typename T, typename HDR = hipc::EmptyHeader, HSHM_CLASS_TEMPL_WITH_PRIV_DEFAULTS>
using multi_fixed_mpsc_queue =
    hipc::multi_ring_buffer<T, HDR, RING_BUFFER_FIXED_MPMC_FLAGS,
                            HSHM_CLASS_TEMPL_ARGS>;

// Multi-lane circular MPSC queue
template <typename T, typename HDR = hipc::EmptyHeader, HSHM_CLASS_TEMPL_WITH_PRIV_DEFAULTS>
using multi_circular_mpsc_queue =
    hipc::multi_ring_buffer<T, HDR, RING_BUFFER_CIRCULAR_MPMC_FLAGS,
                            HSHM_CLASS_TEMPL_ARGS>;

}  // namespace hshm

#endif  // HERMES_SHM_DATA_STRUCTURES_MULTI_RING_BUFFER_H_