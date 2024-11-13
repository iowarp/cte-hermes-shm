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


#ifndef HERMES_DATA_STRUCTURES_THREAD_UNSAFE_IQUEUE_H
#define HERMES_DATA_STRUCTURES_THREAD_UNSAFE_IQUEUE_H

#include "hermes_shm/data_structures/ipc/internal/shm_internal.h"

namespace hshm::ipc {

/** forward pointer for iqueue */
template<typename T, HSHM_CLASS_TEMPL_WITH_DEFAULTS>
class iqueue;

/** represents an object within a iqueue */
struct iqueue_entry {
  OffsetPointer next_shm_;
};

/**
 * The iqueue iterator
 * */
template<typename T, HSHM_CLASS_TEMPL>
struct iqueue_iterator_templ {
 public:
  /**< A shm reference to the containing iqueue object. */
  iqueue<T, HSHM_CLASS_TEMPL_ARGS> *iqueue_;
  /**< A pointer to the entry in shared memory */
  iqueue_entry *entry_;
  /**< A pointer to the entry prior to this one */
  iqueue_entry *prior_entry_;

  /** Default constructor */
  HSHM_CROSS_FUN
  iqueue_iterator_templ() = default;

  /** Construct begin iterator  */
  HSHM_CROSS_FUN
  explicit iqueue_iterator_templ(iqueue<T, HSHM_CLASS_TEMPL_ARGS> &iqueue,
                                 iqueue_entry *entry)
    : iqueue_(&iqueue), entry_(entry), prior_entry_(nullptr) {}

  /** Copy constructor */
  HSHM_CROSS_FUN
  iqueue_iterator_templ(const iqueue_iterator_templ &other)
  : iqueue_(other.iqueue_) {
    iqueue_ = other.iqueue_;
    entry_ = other.entry_;
    prior_entry_ = other.prior_entry_;
  }

  /** Assign this iterator from another iterator */
  HSHM_CROSS_FUN
  iqueue_iterator_templ& operator=(const iqueue_iterator_templ &other) {
    if (this != &other) {
      iqueue_ = other.iqueue_;
      entry_ = other.entry_;
      prior_entry_ = other.prior_entry_;
    }
    return *this;
  }

  /** Get the object the iterator points to */
  HSHM_CROSS_FUN
  T* operator*() {
    return reinterpret_cast<T*>(entry_);
  }

  /** Get the object the iterator points to */
  HSHM_CROSS_FUN
  T* operator*() const {
    return reinterpret_cast<T*>(entry_);
  }

  /** Get the next iterator (in place) */
  HSHM_CROSS_FUN
  iqueue_iterator_templ& operator++() {
    if (is_end()) { return *this; }
    prior_entry_ = entry_;
    entry_ = iqueue_->GetCtxAllocator()->template
      Convert<iqueue_entry>(entry_->next_shm_);
    return *this;
  }

  /** Return the next iterator */
  HSHM_CROSS_FUN
  iqueue_iterator_templ operator++(int) const {
    iqueue_iterator_templ next_iter(*this);
    ++next_iter;
    return next_iter;
  }

  /** Return the iterator at count after this one */
  HSHM_CROSS_FUN
  iqueue_iterator_templ operator+(size_t count) const {
    iqueue_iterator_templ pos(*this);
    for (size_t i = 0; i < count; ++i) {
      ++pos;
    }
    return pos;
  }

  /** Get the iterator at count after this one (in-place) */
  HSHM_CROSS_FUN
  void operator+=(size_t count) {
    iqueue_iterator_templ pos = (*this) + count;
    entry_ = pos.entry_;
    prior_entry_ = pos.prior_entry_;
  }

  /** Determine if two iterators are equal */
  HSHM_CROSS_FUN
  friend bool operator==(const iqueue_iterator_templ &a,
                         const iqueue_iterator_templ &b) {
    return (a.is_end() && b.is_end()) || (a.entry_ == b.entry_);
  }

  /** Determine if two iterators are inequal */
  HSHM_CROSS_FUN
  friend bool operator!=(const iqueue_iterator_templ &a,
                         const iqueue_iterator_templ &b) {
    return !(a.is_end() && b.is_end()) && (a.entry_ != b.entry_);
  }

  /** Determine whether this iterator is the end iterator */
  HSHM_CROSS_FUN
  bool is_end() const {
    return entry_ == nullptr;
  }

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
 * MACROS used to simplify the iqueue namespace
 * Used as inputs to the HIPC_CONTAINER_TEMPLATE
 * */
#define CLASS_NAME iqueue
#define CLASS_NEW_ARGS T

/**
 * Doubly linked iqueue implementation
 * */
template<typename T, HSHM_CLASS_TEMPL>
class iqueue : public ShmContainer {
 public:
  HIPC_CONTAINER_TEMPLATE((CLASS_NAME), (CLASS_NEW_ARGS))
  OffsetPointer head_shm_, tail_shm_;
  size_t length_;

  /**====================================
   * Typedefs
   * ===================================*/

  /** forward iterator typedef */
  typedef iqueue_iterator_templ<T, HSHM_CLASS_TEMPL_ARGS> iterator_t;
  /** const forward iterator typedef */
  typedef iqueue_iterator_templ<T, HSHM_CLASS_TEMPL_ARGS> citerator_t;

 public:
  /**====================================
   * Default Constructor
   * ===================================*/

  /** Constructor. Default. */
  HSHM_CROSS_FUN
  iqueue() {
    shm_init(HERMES_MEMORY_MANAGER->GetDefaultAllocator());
  }

  /** SHM constructor. Default. */
  HSHM_CROSS_FUN
  explicit iqueue(const hipc::CtxAllocator<AllocT> &alloc) {
    shm_init(alloc);
  }

  /** SHM constructor. Default. */
  HSHM_CROSS_FUN
  void shm_init(const hipc::CtxAllocator<AllocT> &alloc) {
    init_shm_container(alloc);
    length_ = 0;
    head_shm_.SetNull();
    tail_shm_.SetNull();
  }

  /**====================================
   * Copy Constructors
   * ===================================*/

  /** Copy constructor */
  HSHM_CROSS_FUN
  explicit iqueue(const iqueue &other) {
    init_shm_container(HERMES_MEMORY_MANAGER->GetDefaultAllocator());
    shm_strong_copy_op(other);
  }

  /** SHM copy constructor */
  HSHM_CROSS_FUN
  explicit iqueue(const hipc::CtxAllocator<AllocT> &alloc,
                  const iqueue &other) {
    init_shm_container(alloc);
    shm_strong_copy_op(other);
  }

  /** SHM copy assignment operator */
  HSHM_CROSS_FUN
  iqueue& operator=(const iqueue &other) {
    if (this != &other) {
      shm_destroy();
      shm_strong_copy_op(other);
    }
    return *this;
  }

  /** SHM copy constructor + operator */
  HSHM_CROSS_FUN
  void shm_strong_copy_op(const iqueue &other) {
    memcpy((void*)this, (void*)&other, sizeof(*this));
  }

  /**====================================
   * Move Constructors
   * ===================================*/

  /** Move constructor. */
  HSHM_CROSS_FUN
  iqueue(iqueue &&other) noexcept {
    init_shm_container(other.GetCtxAllocator());
    memcpy((void*)this, (void*)&other, sizeof(*this));
    other.SetNull();
  }

  /** SHM move constructor. */
  HSHM_CROSS_FUN
  iqueue(const hipc::CtxAllocator<AllocT> &alloc, iqueue &&other) noexcept {
    init_shm_container(alloc);
    if (GetCtxAllocator() == other.GetCtxAllocator()) {
      memcpy((void*)this, (void*)&other, sizeof(*this));
      other.SetNull();
    } else {
      shm_strong_copy_op(other);
      other.shm_destroy();
    }
  }

  /** SHM move assignment operator. */
  HSHM_CROSS_FUN
  iqueue& operator=(iqueue &&other) noexcept {
    if (this != &other) {
      shm_destroy();
      if (this != &other) {
        memcpy((void *) this, (void *) &other, sizeof(*this));
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

  /** Check if the iqueue is null */
  HSHM_CROSS_FUN
  bool IsNull() {
    return length_ == 0;
  }

  /** Set the iqueue to null */
  HSHM_CROSS_FUN
  void SetNull() {
    length_ = 0;
  }

  /** SHM destructor. */
  HSHM_CROSS_FUN
  void shm_destroy_main() {
    clear();
  }

  /**====================================
   * iqueue Methods
   * ===================================*/

  /** Construct an element at \a pos position in the iqueue */
  HSHM_CROSS_FUN
  void enqueue(T *entry) {
    CtxAllocator alloc = GetCtxAllocator();
    OffsetPointer entry_shm = alloc->
        template Convert<T, OffsetPointer>(entry);
    if (tail_shm_.IsNull()) {
      head_shm_ = entry_shm;
    } else {
      auto tail = alloc->
          template Convert<iqueue_entry>(tail_shm_);
      tail->next_shm_ = entry_shm;
    }
    entry->next_shm_ = OffsetPointer::GetNull();
    tail_shm_ = entry_shm;
    ++length_;
  }

  /** Wrapper for enqueue */
  HSHM_INLINE_CROSS_FUN
  void push(T *entry) {
    enqueue(entry);
  }

  /** Dequeue the first element */
  HSHM_CROSS_FUN
  T* dequeue() {
    if (size() == 0) { return nullptr; }
    auto entry = GetCtxAllocator()->
      template Convert<iqueue_entry>(head_shm_);
    head_shm_ = entry->next_shm_;
    --length_;
    if (size() == 0) {
      tail_shm_.SetNull();
    }
    return reinterpret_cast<T*>(entry);
  }

  /** Wrapper for dequeue */
  HSHM_INLINE_CROSS_FUN
  T* pop() {
    return dequeue();
  }

  /** Dequeue the element at the iterator position */
  HSHM_CROSS_FUN
  T* dequeue(iterator_t pos) {
    if (pos.prior_entry_ == nullptr) {
      return dequeue();
    }
    auto entry = *pos;
    auto prior_cast = reinterpret_cast<iqueue_entry*>(pos.prior_entry_);
    auto pos_cast = reinterpret_cast<iqueue_entry*>(pos.entry_);
    prior_cast->next_shm_ = pos_cast->next_shm_;
    --length_;
    return reinterpret_cast<T*>(entry);
  }

  /** Wrapper for dequeue */
  HSHM_INLINE_CROSS_FUN
  T* pop(iterator_t pos) {
    return dequeue(pos);
  }

  /** Peek the first element of the queue */
  HSHM_CROSS_FUN
  T* peek() {
    if (size() == 0) { return nullptr; }
    auto entry = GetCtxAllocator()->
      template Convert<iqueue_entry>(head_shm_);
    return reinterpret_cast<T*>(entry);
  }

  /** Destroy all elements in the iqueue */
  HSHM_CROSS_FUN
  void clear() {
    while (size()) {
      dequeue();
    }
  }

  /** Get the number of elements in the iqueue */
  HSHM_CROSS_FUN
  size_t size() const {
    return length_;
  }

  /**====================================
  * Iterators
  * ===================================*/

  /** Forward iterator begin */
  HSHM_CROSS_FUN
  iterator_t begin() {
    if (size() == 0) { return end(); }
    auto head = GetCtxAllocator()->template
      Convert<iqueue_entry>(head_shm_);
    return iterator_t(*this, head);
  }

  /** Forward iterator end */
  HSHM_CROSS_FUN
  iterator_t const end() {
    return iterator_t(*this, nullptr);
  }

  /** Constant forward iterator begin */
  HSHM_CROSS_FUN
  citerator_t cbegin() const {
    if (size() == 0) { return cend(); }
    auto head = GetCtxAllocator()->template
      Convert<iqueue_entry>(head_shm_);
    return citerator_t(const_cast<iqueue&>(*this), head);
  }

  /** Constant forward iterator end */
  HSHM_CROSS_FUN
  citerator_t const cend() const {
    return citerator_t(const_cast<iqueue&>(*this), nullptr);
  }
};

}  // namespace hshm::ipc

#undef CLASS_NAME
#undef CLASS_NEW_ARGS

#endif  // HERMES_DATA_STRUCTURES_THREAD_UNSAFE_IQUEUE_H
