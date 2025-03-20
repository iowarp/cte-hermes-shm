//
// Created by llogan on 11/29/24.
//

#ifndef HSHM_SHM_INCLUDE_HSHM_SHM_DATA_STRUCTURES_IPC_DYNAMIC_QUEUE_H_
#define HSHM_SHM_INCLUDE_HSHM_SHM_DATA_STRUCTURES_IPC_DYNAMIC_QUEUE_H_

#include <utility>

#include "hermes_shm/data_structures/internal/shm_internal.h"
#include "hermes_shm/data_structures/ipc/functional.h"
#include "hermes_shm/data_structures/serialization/serialize_common.h"
#include "list.h"
#include "ring_queue.h"

namespace hshm::ipc {

/**
 * MACROS used to simplify the ring_queue_base namespace
 * Used as inputs to the HIPC_CONTAINER_TEMPLATE
 * */
#define CLASS_NAME dynamic_queue
#define CLASS_NEW_ARGS T

/**
 * An mpsc queue that changes in size dynamically
 */
template <typename T, HSHM_CLASS_TEMPL_WITH_DEFAULTS>
class dynamic_queue : public ShmContainer {
 public:
  HIPC_CONTAINER_TEMPLATE((CLASS_NAME), (CLASS_NEW_ARGS))
  typedef hipc::list<T, HSHM_CLASS_TEMPL_ARGS> Vector;

 public:
  hshm::Mutex lock_;
  Vector splits_;
  hipc::atomic<hshm::size_t> head_, tail_;
  size_t block_size_;

  /**====================================
   * Default Constructor
   * ===================================*/
 public:
  dynamic_queue(size_t block_size = 64) : splits_() {
    shm_init(HSHM_MEMORY_MANAGER->GetDefaultAllocator<AllocT>(), block_size);
  }

  dynamic_queue(const hipc::CtxAllocator<AllocT> &alloc, size_t block_size = 64)
      : splits_(alloc), block_size_(block_size) {
    shm_init(alloc, block_size);
  }

  void shm_init(const hipc::CtxAllocator<AllocT> &alloc,
                size_t block_size = 64) {
    init_shm_container(alloc);
    block_size_ = block_size;
    head_ = 0;
    tail_ = 0;
  }

  /**====================================
   * Copy Constructors
   * ===================================*/

  /** SHM copy constructor */
  HSHM_CROSS_FUN
  explicit dynamic_queue(const hipc::CtxAllocator<AllocT> &alloc,
                         const dynamic_queue &other) {
    init_shm_container(alloc);
    SetNull();
    shm_strong_copy_op(other);
  }

  /** SHM copy assignment operator */
  HSHM_CROSS_FUN
  dynamic_queue &operator=(const dynamic_queue &other) {
    if (this != &other) {
      shm_destroy();
      shm_strong_copy_op(other);
    }
    return *this;
  }

  /** SHM copy constructor + operator main */
  HSHM_CROSS_FUN
  void shm_strong_copy_op(const dynamic_queue &other) {
    splits_ = other.splits_;
    head_ = other.head_.load();
    tail_ = other.tail_.load();
    block_size_ = other.block_size_;
  }

  /**====================================
   * Move Constructors
   * ===================================*/

  /** Move constructor. */
  HSHM_CROSS_FUN
  dynamic_queue(dynamic_queue &&other) noexcept {
    shm_move_op<false>(other.GetCtxAllocator(),
                       std::forward<dynamic_queue>(other));
  }

  /** SHM move constructor. */
  HSHM_CROSS_FUN
  dynamic_queue(const hipc::CtxAllocator<AllocT> &alloc,
                dynamic_queue &&other) noexcept {
    shm_move_op<false>(alloc, std::forward<dynamic_queue>(other));
  }

  /** SHM move assignment operator. */
  HSHM_CROSS_FUN
  dynamic_queue &operator=(dynamic_queue &&other) noexcept {
    if (this != &other) {
      shm_move_op<true>(other.GetCtxAllocator(),
                        std::forward<dynamic_queue>(other));
    }
    return *this;
  }

  /** SHM move assignment operator. */
  template <bool IS_ASSIGN>
  HSHM_CROSS_FUN void shm_move_op(const hipc::CtxAllocator<AllocT> &alloc,
                                  dynamic_queue &&other) noexcept {
    if constexpr (!IS_ASSIGN) {
      init_shm_container(alloc);
    }
    if (GetAllocator() == other.GetAllocator()) {
      splits_ = std::move(other.splits_);
      head_ = other.head_.load();
      tail_ = other.tail_.load();
      block_size_ = other.block_size_;
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
  void shm_destroy_main() {}

  /** Check if the list is empty */
  HSHM_CROSS_FUN
  bool IsNull() const { return false; }

  /** Sets this list as empty */
  HSHM_CROSS_FUN
  void SetNull() {}

  /**====================================
   * MPSC Queue Methods
   * ===================================*/

  /** Construct an element at \a pos position in the list */
  template <typename... Args>
  HSHM_CROSS_FUN qtok_t emplace(Args &&...args) {
    ScopedMutex lock(lock_, 0);
    qtok_id tail = tail_.fetch_add(1);
    splits_.emplace_back(std::forward<Args>(args)...);
    return qtok_t(tail);
  }

  /** Push an elemnt in the list (wrapper) */
  template <typename... Args>
  HSHM_INLINE_CROSS_FUN qtok_t push(Args &&...args) {
    return emplace(std::forward<Args>(args)...);
  }

  /** Consumer pops the head object */
  HSHM_CROSS_FUN
  qtok_t pop(T &val) {
    ScopedMutex lock(lock_, 0);
    qtok_t ret = qtok_t::GetNull();
    if (splits_.size() > 0) {
      ret = qtok_t(head_.fetch_add(1));
      val = splits_.front();
      splits_.erase(splits_.begin());
    }
    return ret;
  }

  /** Consumer pops the head object */
  HSHM_CROSS_FUN
  qtok_t pop() {
    ScopedMutex lock(lock_, 0);
    qtok_t ret = qtok_t::GetNull();
    if (splits_.size() > 0) {
      ret = qtok_t(head_.fetch_add(1));
      splits_.erase(splits_.begin());
    }
    return ret;
  }

  /** Get queue depth */
  HSHM_CROSS_FUN
  size_t GetDepth() { return splits_.size(); }

  /** Get size at this moment */
  HSHM_CROSS_FUN
  size_t GetSize() {
    hshm::size_t tail = tail_.load();
    hshm::size_t head = head_.load();
    if (tail < head) {
      return 0;
    }
    return (size_t)(tail - head);
  }

  /** Get size (wrapper) */
  HSHM_INLINE_CROSS_FUN
  size_t size() { return GetSize(); }

  /** Get size (wrapper) */
  HSHM_INLINE_CROSS_FUN
  size_t Size() { return GetSize(); }
};

}  // namespace hshm::ipc

namespace hshm {

template <typename T, HSHM_CLASS_TEMPL_WITH_PRIV_DEFAULTS>
using dynamic_queue = ipc::dynamic_queue<T, HSHM_CLASS_TEMPL_ARGS>;

}  // namespace hshm

#undef CLASS_NAME
#undef CLASS_NEW_ARGS

#endif  // HSHM_SHM_INCLUDE_HSHM_SHM_DATA_STRUCTURES_IPC_DYNAMIC_QUEUE_H_
