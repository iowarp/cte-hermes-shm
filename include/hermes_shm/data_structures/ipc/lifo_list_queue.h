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

#ifndef HSHM_DATA_STRUCTURES_THREAD_UNSAFE_lifo_list_queue_H
#define HSHM_DATA_STRUCTURES_THREAD_UNSAFE_lifo_list_queue_H

#include "hermes_shm/data_structures/internal/shm_internal.h"
#include "hermes_shm/types/qtok.h"

namespace hshm::ipc {

/** forward pointer for lifo_list_queue */
template <typename T, HSHM_CLASS_TEMPL_WITH_DEFAULTS>
class lifo_list_queue;

/** represents an object within a lifo_list_queue */
struct list_queue_entry {
  OffsetPointer next_shm_;
};

/** represents an object within a lifo_list_queue */
struct atomic_list_queue_entry {
  AtomicOffsetPointer next_shm_;
};

/**
 * The lifo_list_queue iterator
 * */
template <typename T, HSHM_CLASS_TEMPL>
struct lifo_list_queue_iterator_templ {
 public:
  /**< A shm reference to the containing lifo_list_queue object. */
  lifo_list_queue<T, HSHM_CLASS_TEMPL_ARGS> *lifo_list_queue_;
  /**< A pointer to the entry in shared memory */
  list_queue_entry *entry_;
  /**< A pointer to the entry prior to this one */
  list_queue_entry *prior_entry_;

  /** Default constructor */
  HSHM_CROSS_FUN
  lifo_list_queue_iterator_templ() = default;

  /** Construct begin iterator  */
  HSHM_CROSS_FUN
  explicit lifo_list_queue_iterator_templ(
      lifo_list_queue<T, HSHM_CLASS_TEMPL_ARGS> &lifo_list_queue,
      list_queue_entry *entry)
      : lifo_list_queue_(&lifo_list_queue),
        entry_(entry),
        prior_entry_(nullptr) {}

  /** Copy constructor */
  HSHM_CROSS_FUN
  lifo_list_queue_iterator_templ(const lifo_list_queue_iterator_templ &other)
      : lifo_list_queue_(other.lifo_list_queue_) {
    lifo_list_queue_ = other.lifo_list_queue_;
    entry_ = other.entry_;
    prior_entry_ = other.prior_entry_;
  }

  /** Assign this iterator from another iterator */
  HSHM_CROSS_FUN
  lifo_list_queue_iterator_templ &operator=(
      const lifo_list_queue_iterator_templ &other) {
    if (this != &other) {
      lifo_list_queue_ = other.lifo_list_queue_;
      entry_ = other.entry_;
      prior_entry_ = other.prior_entry_;
    }
    return *this;
  }

  /** Get the object the iterator points to */
  HSHM_CROSS_FUN
  T *operator*() { return reinterpret_cast<T *>(entry_); }

  /** Get the object the iterator points to */
  HSHM_CROSS_FUN
  T *operator*() const { return reinterpret_cast<T *>(entry_); }

  /** Get the next iterator (in place) */
  HSHM_CROSS_FUN
  lifo_list_queue_iterator_templ &operator++() {
    if (is_end()) {
      return *this;
    }
    prior_entry_ = entry_;
    entry_ =
        lifo_list_queue_->GetAllocator()->template Convert<list_queue_entry>(
            entry_->next_shm_);
    return *this;
  }

  /** Return the next iterator */
  HSHM_CROSS_FUN
  lifo_list_queue_iterator_templ operator++(int) const {
    lifo_list_queue_iterator_templ next_iter(*this);
    ++next_iter;
    return next_iter;
  }

  /** Return the iterator at count after this one */
  HSHM_CROSS_FUN
  lifo_list_queue_iterator_templ operator+(size_t count) const {
    lifo_list_queue_iterator_templ pos(*this);
    for (size_t i = 0; i < count; ++i) {
      ++pos;
    }
    return pos;
  }

  /** Get the iterator at count after this one (in-place) */
  HSHM_CROSS_FUN
  void operator+=(size_t count) {
    lifo_list_queue_iterator_templ pos = (*this) + count;
    entry_ = pos.entry_;
    prior_entry_ = pos.prior_entry_;
  }

  /** Determine if two iterators are equal */
  HSHM_CROSS_FUN
  friend bool operator==(const lifo_list_queue_iterator_templ &a,
                         const lifo_list_queue_iterator_templ &b) {
    return (a.is_end() && b.is_end()) || (a.entry_ == b.entry_);
  }

  /** Determine if two iterators are inequal */
  HSHM_CROSS_FUN
  friend bool operator!=(const lifo_list_queue_iterator_templ &a,
                         const lifo_list_queue_iterator_templ &b) {
    return !(a.is_end() && b.is_end()) && (a.entry_ != b.entry_);
  }

  /** Determine whether this iterator is the end iterator */
  HSHM_CROSS_FUN
  bool is_end() const { return entry_ == nullptr; }

  /** Determine whether this iterator is the begin iterator */
  HSHM_CROSS_FUN
  bool is_begin() const {
    if (entry_) {
      return prior_entry_ == nullptr;
    } else {
      return false;
    }
  }
};

/**
 * MACROS used to simplify the lifo_list_queue namespace
 * Used as inputs to the HIPC_CONTAINER_TEMPLATE
 * */
#define CLASS_NAME lifo_list_queue
#define CLASS_NEW_ARGS T

/**
 * Doubly linked lifo_list_queue implementation
 * */
template <typename T, HSHM_CLASS_TEMPL>
class lifo_list_queue : public ShmContainer {
 public:
  HIPC_CONTAINER_TEMPLATE((CLASS_NAME), (CLASS_NEW_ARGS))
  OffsetPointer tail_shm_;
  size_t length_;

  /**====================================
   * Typedefs
   * ===================================*/

  /** forward iterator typedef */
  typedef lifo_list_queue_iterator_templ<T, HSHM_CLASS_TEMPL_ARGS> iterator_t;
  /** const forward iterator typedef */
  typedef lifo_list_queue_iterator_templ<T, HSHM_CLASS_TEMPL_ARGS> citerator_t;

 public:
  /**====================================
   * Default Constructor
   * ===================================*/

  /** Constructor. Default. */
  HSHM_CROSS_FUN
  lifo_list_queue() {
    shm_init(HSHM_MEMORY_MANAGER->GetDefaultAllocator<AllocT>());
  }

  /** SHM constructor. Default. */
  HSHM_CROSS_FUN
  explicit lifo_list_queue(const hipc::CtxAllocator<AllocT> &alloc) {
    shm_init(alloc);
  }

  /** SHM constructor. Default. */
  HSHM_CROSS_FUN
  void shm_init(const hipc::CtxAllocator<AllocT> &alloc) {
    init_shm_container(alloc);
    length_ = 0;
    tail_shm_.SetNull();
  }

  /**====================================
   * Copy Constructors
   * ===================================*/

  /** Copy constructor */
  HSHM_CROSS_FUN
  explicit lifo_list_queue(const lifo_list_queue &other) {
    init_shm_container(HSHM_MEMORY_MANAGER->GetDefaultAllocator<AllocT>());
    shm_strong_copy_op(other);
  }

  /** SHM copy constructor */
  HSHM_CROSS_FUN
  explicit lifo_list_queue(const hipc::CtxAllocator<AllocT> &alloc,
                           const lifo_list_queue &other) {
    init_shm_container(alloc);
    shm_strong_copy_op(other);
  }

  /** SHM copy assignment operator */
  HSHM_CROSS_FUN
  lifo_list_queue &operator=(const lifo_list_queue &other) {
    if (this != &other) {
      shm_destroy();
      shm_strong_copy_op(other);
    }
    return *this;
  }

  /** SHM copy constructor + operator */
  HSHM_CROSS_FUN
  void shm_strong_copy_op(const lifo_list_queue &other) {
    memcpy((void *)this, (void *)&other, sizeof(*this));
  }

  /**====================================
   * Move Constructors
   * ===================================*/

  /** Move constructor. */
  HSHM_CROSS_FUN
  lifo_list_queue(lifo_list_queue &&other) noexcept {
    init_shm_container(other.GetAllocator());
    memcpy((void *)this, (void *)&other, sizeof(*this));
    other.SetNull();
  }

  /** SHM move constructor. */
  HSHM_CROSS_FUN
  lifo_list_queue(const hipc::CtxAllocator<AllocT> &alloc,
                  lifo_list_queue &&other) noexcept {
    init_shm_container(alloc);
    if (GetAllocator() == other.GetAllocator()) {
      memcpy((void *)this, (void *)&other, sizeof(*this));
      other.SetNull();
    } else {
      shm_strong_copy_op(other);
      other.shm_destroy();
    }
  }

  /** SHM move assignment operator. */
  HSHM_CROSS_FUN
  lifo_list_queue &operator=(lifo_list_queue &&other) noexcept {
    if (this != &other) {
      if (this != &other) {
        memcpy((void *)this, (void *)&other, sizeof(*this));
        other.SetNull();
      } else {
        shm_strong_copy_op(other);
        other.shm_destroy();
      }
    }
    return *this;
  }

  /**====================================
   * Destructor
   * ===================================*/

  /** Check if the lifo_list_queue is null */
  HSHM_CROSS_FUN
  bool IsNull() { return length_ == 0; }

  /** Set the lifo_list_queue to null */
  HSHM_CROSS_FUN
  void SetNull() { length_ = 0; }

  /** SHM destructor. */
  HSHM_CROSS_FUN
  void shm_destroy_main() { clear(); }

  /**====================================
   * lifo_list_queue Methods
   * ===================================*/

  /** Construct an element at tail. FullPtr */
  HSHM_CROSS_FUN
  void enqueue(const FullPtr<T> &entry) {
    entry.ptr_->next_shm_ = tail_shm_;
    tail_shm_ = entry.shm_.off_;
    ++length_;
  }

  /** Emplace. wrapper for enqueue. FullPtr */
  HSHM_CROSS_FUN
  void emplace(const FullPtr<T> &entry) { enqueue(entry); }

  /** Push. wrapper for enqueue. FullPtr */
  HSHM_INLINE_CROSS_FUN
  void push(const FullPtr<T> &entry) { enqueue(entry); }

  /** Construct an element at tail. Raw Ptr */
  HSHM_CROSS_FUN
  void enqueue(T *entry) {
    FullPtr<T> entry_ptr(GetAllocator(), entry);
    enqueue(entry_ptr);
  }

  /** Emplace. wrapper for enqueue. Raw Ptr */
  HSHM_CROSS_FUN
  void emplace(T *entry) { enqueue(entry); }

  /** Push. wrapper for enqueue. Raw Ptr */
  HSHM_INLINE_CROSS_FUN
  void push(T *entry) { enqueue(entry); }

  /** Dequeue the first element */
  HSHM_INLINE_CROSS_FUN
  T *dequeue() {
    T *val;
    if (dequeue(val).IsNull()) {
      return nullptr;
    }
    return val;
  }

  /** Wrapper for dequeue */
  HSHM_INLINE_CROSS_FUN
  T *pop() { return dequeue(); }

  /** Dequeue the element (qtok_t, FullPtr) */
  HSHM_INLINE_CROSS_FUN
  qtok_t dequeue(FullPtr<T> &val) {
    if (size() == 0) {
      return qtok_t::GetNull();
    }
    val = FullPtr<T>(GetAllocator(), tail_shm_);
    tail_shm_ = val.ptr_->next_shm_;
    --length_;
    return qtok_t(1);
  }

  /** Pop the element (qtok_t, FullPtr) */
  HSHM_INLINE_CROSS_FUN
  qtok_t pop(FullPtr<T> &val) { return dequeue(val); }

  /** Dequeue the element (qtok_t) */
  HSHM_INLINE_CROSS_FUN
  qtok_t dequeue(T *&val) {
    FullPtr<T> entry;
    qtok_t ret = dequeue(entry);
    val = entry.ptr_;
    return ret;
  }

  /** Pop the element (qtok_t) */
  HSHM_INLINE_CROSS_FUN
  qtok_t pop(T *&val) { return dequeue(val); }

  /** Dequeue the element at the iterator position */
  HSHM_CROSS_FUN
  T *dequeue(iterator_t pos) {
    if (pos.prior_entry_ == nullptr) {
      return dequeue();
    }
    auto entry = *pos;
    auto prior_cast = reinterpret_cast<T *>(pos.prior_entry_);
    auto pos_cast = reinterpret_cast<T *>(pos.entry_);
    prior_cast->next_shm_ = pos_cast->next_shm_;
    --length_;
    return entry;
  }

  /** Dequeue at position */
  HSHM_INLINE_CROSS_FUN
  T *pop(iterator_t pos) { return dequeue(pos); }

  /** Peek the first element of the queue */
  HSHM_CROSS_FUN
  T *peek() {
    if (size() == 0) {
      return nullptr;
    }
    auto entry = GetAllocator()->template Convert<list_queue_entry>(tail_shm_);
    return reinterpret_cast<T *>(entry);
  }

  /** Destroy all elements in the lifo_list_queue */
  HSHM_CROSS_FUN
  void clear() {
    while (size()) {
      dequeue();
    }
  }

  /** Get the number of elements in the lifo_list_queue */
  HSHM_CROSS_FUN
  size_t size() const { return length_; }

  /**====================================
   * Iterators
   * ===================================*/

  /** Forward iterator begin */
  HSHM_CROSS_FUN
  iterator_t begin() {
    if (size() == 0) {
      return end();
    }
    auto head = GetAllocator()->template Convert<list_queue_entry>(tail_shm_);
    return iterator_t(*this, head);
  }

  /** Forward iterator end */
  HSHM_CROSS_FUN
  iterator_t const end() { return iterator_t(*this, nullptr); }

  /** Constant forward iterator begin */
  HSHM_CROSS_FUN
  citerator_t cbegin() const {
    if (size() == 0) {
      return cend();
    }
    auto head = GetAllocator()->template Convert<list_queue_entry>(tail_shm_);
    return citerator_t(const_cast<lifo_list_queue &>(*this), head);
  }

  /** Constant forward iterator end */
  HSHM_CROSS_FUN
  citerator_t const cend() const {
    return citerator_t(const_cast<lifo_list_queue &>(*this), nullptr);
  }
};

}  // namespace hshm::ipc

namespace hshm {

template <typename T, HSHM_CLASS_TEMPL_WITH_PRIV_DEFAULTS>
using lifo_list_queue = hshm::ipc::lifo_list_queue<T, HSHM_CLASS_TEMPL_ARGS>;

}  // namespace hshm

#undef CLASS_NAME
#undef CLASS_NEW_ARGS

#endif  // HSHM_DATA_STRUCTURES_THREAD_UNSAFE_lifo_list_queue_H
