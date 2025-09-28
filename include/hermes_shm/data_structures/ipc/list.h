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

#ifndef HSHM_DATA_STRUCTURES__LIST_H_
#define HSHM_DATA_STRUCTURES__LIST_H_

#include <list>

#include "hermes_shm/data_structures/internal/shm_internal.h"
#include "hermes_shm/data_structures/ipc/functional.h"
#include "hermes_shm/data_structures/serialization/serialize_common.h"

namespace hshm::ipc {

/** forward pointer for list */
template <typename T, HSHM_CLASS_TEMPL_WITH_DEFAULTS>
class list;

/** represents an object within a list */
template <typename T>
struct list_entry {
 public:
  OffsetPointer next_ptr_, prior_ptr_;
  delay_ar<T> data_;
};

/**
 * The list iterator
 * */
template <typename T, HSHM_CLASS_TEMPL>
struct list_iterator_templ {
 public:
  /**< A shm reference to the containing list object. */
  list<T, HSHM_CLASS_TEMPL_ARGS> *list_;
  /**< A pointer to the entry in shared memory */
  list_entry<T> *entry_;
  /**< The offset of the entry in the shared-memory allocator */
  OffsetPointer entry_ptr_;

  /** Default constructor */
  HSHM_CROSS_FUN
  list_iterator_templ() = default;

  /** Construct an iterator  */
  HSHM_CROSS_FUN
  explicit list_iterator_templ(list<T, HSHM_CLASS_TEMPL_ARGS> &list,
                               list_entry<T> *entry, OffsetPointer entry_ptr)
      : list_(&list), entry_(entry), entry_ptr_(entry_ptr) {}

  /** Copy constructor */
  HSHM_CROSS_FUN
  list_iterator_templ(const list_iterator_templ &other) {
    list_ = other.list_;
    entry_ = other.entry_;
    entry_ptr_ = other.entry_ptr_;
  }

  /** Assign this iterator from another iterator */
  HSHM_CROSS_FUN
  list_iterator_templ &operator=(const list_iterator_templ &other) {
    if (this != &other) {
      list_ = other.list_;
      entry_ = other.entry_;
      entry_ptr_ = other.entry_ptr_;
    }
    return *this;
  }

  /** Get the object the iterator points to */
  HSHM_CROSS_FUN
  T &operator*() { return entry_->data_.get_ref(); }

  /** Get the object the iterator points to */
  HSHM_CROSS_FUN
  const T &operator*() const { return entry_->data_.get_ref(); }

  /** Get the next iterator (in place) */
  HSHM_CROSS_FUN
  list_iterator_templ &operator++() {
    if (is_end()) {
      return *this;
    }
    entry_ptr_ = entry_->next_ptr_;
    entry_ = list_->GetAllocator()->template Convert<list_entry<T>>(
        entry_->next_ptr_);
    return *this;
  }

  /** Get the prior iterator (in place) */
  HSHM_CROSS_FUN
  list_iterator_templ &operator--() {
    if (is_end() || is_begin()) {
      return *this;
    }
    entry_ptr_ = entry_->prior_ptr_;
    entry_ = list_->GetAllocator()->template Convert<list_entry<T>>(
        entry_->prior_ptr_);
    return *this;
  }

  /** Return the next iterator */
  HSHM_CROSS_FUN
  list_iterator_templ operator++(int) const {
    list_iterator_templ next_iter(*this);
    ++next_iter;
    return next_iter;
  }

  /** Return the prior iterator */
  HSHM_CROSS_FUN
  list_iterator_templ operator--(int) const {
    list_iterator_templ prior_iter(*this);
    --prior_iter;
    return prior_iter;
  }

  /** Return the iterator at count after this one */
  HSHM_CROSS_FUN
  list_iterator_templ operator+(size_t count) const {
    list_iterator_templ pos(*this);
    for (size_t i = 0; i < count; ++i) {
      ++pos;
    }
    return pos;
  }

  /** Return the iterator at count before this one */
  HSHM_CROSS_FUN
  list_iterator_templ operator-(size_t count) const {
    list_iterator_templ pos(*this);
    for (size_t i = 0; i < count; ++i) {
      --pos;
    }
    return pos;
  }

  /** Get the iterator at count after this one (in-place) */
  HSHM_CROSS_FUN
  void operator+=(size_t count) {
    list_iterator_templ pos = (*this) + count;
    entry_ = pos.entry_;
    entry_ptr_ = pos.entry_ptr_;
  }

  /** Get the iterator at count before this one (in-place) */
  HSHM_CROSS_FUN
  void operator-=(size_t count) {
    list_iterator_templ pos = (*this) - count;
    entry_ = pos.entry_;
    entry_ptr_ = pos.entry_ptr_;
  }

  /** Determine if two iterators are equal */
  HSHM_CROSS_FUN
  friend bool operator==(const list_iterator_templ &a,
                         const list_iterator_templ &b) {
    return (a.is_end() && b.is_end()) || (a.entry_ == b.entry_);
  }

  /** Determine if two iterators are inequal */
  HSHM_CROSS_FUN
  friend bool operator!=(const list_iterator_templ &a,
                         const list_iterator_templ &b) {
    return !(a.is_end() && b.is_end()) && (a.entry_ != b.entry_);
  }

  /** Determine whether this iterator is the end iterator */
  HSHM_CROSS_FUN
  bool is_end() const { return entry_ == nullptr; }

  /** Determine whether this iterator is the begin iterator */
  HSHM_CROSS_FUN
  bool is_begin() const {
    if (entry_) {
      return entry_->prior_ptr_.IsNull();
    } else {
      return false;
    }
  }
};

/**
 * MACROS used to simplify the list namespace
 * Used as inputs to the HIPC_CONTAINER_TEMPLATE
 * */
#define CLASS_NAME list
#define CLASS_NEW_ARGS T

/**
 * Doubly linked list implementation
 * */
template <typename T, HSHM_CLASS_TEMPL>
class list : public ShmContainer {
 public:
  HIPC_CONTAINER_TEMPLATE((CLASS_NAME), (CLASS_NEW_ARGS))
  OffsetPointer head_ptr_, tail_ptr_;
  size_t length_;

 public:
  /**====================================
   * Typedefs
   * ===================================*/

  /** forward iterator typedef */
  typedef list_iterator_templ<T, HSHM_CLASS_TEMPL_ARGS> iterator_t;
  /** const forward iterator typedef */
  typedef list_iterator_templ<T, HSHM_CLASS_TEMPL_ARGS> citerator_t;

 public:
  /**====================================
   * Default Constructor
   * ===================================*/

  /** Constructor. Default. */
  HSHM_CROSS_FUN
  explicit list() {
    init_shm_container(HSHM_MEMORY_MANAGER->GetDefaultAllocator<AllocT>());
    SetNull();
  }

  /** SHM constructor. Default. */
  HSHM_CROSS_FUN
  explicit list(const hipc::CtxAllocator<AllocT> &alloc) {
    init_shm_container(alloc);
    SetNull();
  }

  /**====================================
   * Copy Constructors
   * ===================================*/

  /** Copy constructor */
  HSHM_CROSS_FUN
  explicit list(const list &other) {
    init_shm_container(other.GetCtxAllocator());
    SetNull();
    shm_strong_copy_op<list>(other);
  }

  /** SHM copy constructor */
  HSHM_CROSS_FUN
  explicit list(const hipc::CtxAllocator<AllocT> &alloc, const list &other) {
    init_shm_container(alloc);
    SetNull();
    shm_strong_copy_op<list>(other);
  }

  /** SHM copy assignment operator */
  HSHM_CROSS_FUN
  list &operator=(const list &other) {
    if (this != &other) {
      shm_destroy();
      shm_strong_copy_op<list>(other);
    }
    return *this;
  }

  /** SHM copy constructor. From std::list */
  HSHM_CROSS_FUN
  explicit list(const hipc::CtxAllocator<AllocT> &alloc, std::list<T> &other) {
    init_shm_container(alloc);
    SetNull();
    shm_strong_copy_op<std::list<T>>(other);
  }

  /** SHM copy assignment operator. From std::list. */
  HSHM_CROSS_FUN
  list &operator=(const std::list<T> &other) {
    if (this != &other) {
      shm_destroy();
      shm_strong_copy_op<std::list<T>>(other);
    }
    return *this;
  }

  /** SHM copy constructor + operator main */
  template <typename ListT>
  HSHM_CROSS_FUN void shm_strong_copy_op(const ListT &other) {
    for (auto iter = other.cbegin(); iter != other.cend(); ++iter) {
      emplace_back(*iter);
    }
  }

  /**====================================
   * Move Constructors
   * ===================================*/

  /** Move constructor. */
  HSHM_CROSS_FUN
  list(list &&other) noexcept {
    shm_move_op<false>(other.GetCtxAllocator(), std::move(other));
  }

  /** SHM move constructor. */
  HSHM_CROSS_FUN
  list(const hipc::CtxAllocator<AllocT> &alloc, list &&other) noexcept {
    shm_move_op<false>(alloc, std::move(other));
  }

  /** SHM move assignment operator. */
  HSHM_CROSS_FUN
  list &operator=(list &&other) noexcept {
    if (this != &other) {
      shm_move_op<true>(GetCtxAllocator(), std::move(other));
    }
    return *this;
  }

  /** SHM move base */
  template <bool IS_ASSIGN>
  HSHM_CROSS_FUN void shm_move_op(const hipc::CtxAllocator<AllocT> &alloc,
                                  list &&other) {
    if constexpr (IS_ASSIGN) {
      shm_destroy();
    } else {
      init_shm_container(alloc);
    }
    if (GetAllocator() == other.GetAllocator()) {
      memcpy((void *)this, (void *)&other, sizeof(*this));
      other.SetNull();
    } else {
      shm_strong_copy_op<list>(other);
      other.shm_destroy();
    }
  }

  /**====================================
   * Destructor
   * ===================================*/

  /** SHM destructor.  */
  HSHM_CROSS_FUN
  void shm_destroy_main() { clear(); }

  /** Check if the list is empty */
  HSHM_CROSS_FUN
  bool IsNull() const { return length_ == 0; }

  /** Sets this list as empty */
  HSHM_CROSS_FUN
  void SetNull() {
    length_ = 0;
    head_ptr_.SetNull();
    tail_ptr_.SetNull();
  }

  /**====================================
   * list Methods
   * ===================================*/

  /** Construct an element at the back of the list */
  template <typename... Args>
  HSHM_CROSS_FUN void emplace_back(Args &&...args) {
    emplace(end(), std::forward<Args>(args)...);
  }

  /** Assign elements to vector using iterator */
  template <typename Iterator>
  HSHM_INLINE_CROSS_FUN void assign(Iterator first, Iterator last) {
    for (auto iter = first; iter != last; ++iter) {
      emplace_back(*iter);
    }
  }

  /** Construct an element at the beginning of the list */
  template <typename... Args>
  HSHM_CROSS_FUN void emplace_front(Args &&...args) {
    emplace(begin(), std::forward<Args>(args)...);
  }

  /** Construct an element at \a pos position in the list */
  template <typename... Args>
  HSHM_CROSS_FUN void emplace(iterator_t pos, Args &&...args) {
    OffsetPointer entry_ptr;
    auto entry = _create_entry(entry_ptr, std::forward<Args>(args)...);
    if (size() == 0) {
      entry->prior_ptr_.SetNull();
      entry->next_ptr_.SetNull();
      head_ptr_ = entry_ptr;
      tail_ptr_ = entry_ptr;
    } else if (pos.is_begin()) {
      entry->prior_ptr_.SetNull();
      entry->next_ptr_ = head_ptr_;
      auto head = GetAllocator()->template Convert<list_entry<T>>(tail_ptr_);
      head->prior_ptr_ = entry_ptr;
      head_ptr_ = entry_ptr;
    } else if (pos.is_end()) {
      entry->prior_ptr_ = tail_ptr_;
      entry->next_ptr_.SetNull();
      auto tail = GetAllocator()->template Convert<list_entry<T>>(tail_ptr_);
      tail->next_ptr_ = entry_ptr;
      tail_ptr_ = entry_ptr;
    } else {
      auto next = GetAllocator()->template Convert<list_entry<T>>(
          pos.entry_->next_ptr_);
      auto prior = GetAllocator()->template Convert<list_entry<T>>(
          pos.entry_->prior_ptr_);
      entry->next_ptr_ = pos.entry_->next_ptr_;
      entry->prior_ptr_ = pos.entry_->prior_ptr_;
      next->prior_ptr_ = entry_ptr;
      prior->next_ptr_ = entry_ptr;
    }
    ++length_;
  }

  /** Erase element with ID */
  HSHM_CROSS_FUN
  void erase(const T &entry) {
    auto iter = find(entry);
    erase(iter);
  }

  /** Erase the element at pos */
  HSHM_CROSS_FUN
  void erase(iterator_t pos) { erase(pos, pos + 1); }

  /** Erase all elements between first and last */
  HSHM_CROSS_FUN
  void erase(iterator_t first, iterator_t last) {
    if (first.is_end()) {
      return;
    }
    auto first_prior_ptr = first.entry_->prior_ptr_;
    auto pos = first;
    while (pos != last) {
      auto next = pos + 1;
      HSHM_DESTROY_AR(pos.entry_->data_)
      FullPtr<void, OffsetPointer> full_ptr(GetAllocator()->template Convert<void>(pos.entry_ptr_), pos.entry_ptr_);
      GetAllocator()->Free(GetMemCtx(), full_ptr);
      --length_;
      pos = next;
    }

    if (first_prior_ptr.IsNull()) {
      head_ptr_ = last.entry_ptr_;
    } else {
      auto first_prior =
          GetAllocator()->template Convert<list_entry<T>>(first_prior_ptr);
      first_prior->next_ptr_ = last.entry_ptr_;
    }

    if (last.entry_ptr_.IsNull()) {
      tail_ptr_ = first_prior_ptr;
    } else {
      last.entry_->prior_ptr_ = first_prior_ptr;
    }
  }

  /** Destroy all elements in the list */
  HSHM_CROSS_FUN
  void clear() { erase(begin(), end()); }

  /** Get the object at the front of the list */
  HSHM_CROSS_FUN
  T &front() { return *begin(); }

  /** Get the object at the back of the list */
  HSHM_CROSS_FUN
  T &back() { return *last(); }

  /** Get the number of elements in the list */
  HSHM_CROSS_FUN
  size_t size() const { return length_; }

  /** Find an element in this list */
  HSHM_CROSS_FUN
  iterator_t find(const T &entry) { return hshm::find(begin(), end(), entry); }

  /**====================================
   * Iterators
   * ===================================*/

  /** Forward iterator begin */
  HSHM_CROSS_FUN
  iterator_t begin() {
    if (size() == 0) {
      return end();
    }
    auto head = GetAllocator()->template Convert<list_entry<T>>(head_ptr_);
    return iterator_t(*this, head, head_ptr_);
  }

  /** Last iterator begin */
  HSHM_CROSS_FUN
  iterator_t last() {
    if (size() == 0) {
      return end();
    }
    auto tail = GetAllocator()->template Convert<list_entry<T>>(tail_ptr_);
    return iterator_t(*this, tail, tail_ptr_);
  }

  /** Forward iterator end */
  HSHM_CROSS_FUN
  iterator_t end() {
    return iterator_t(*this, nullptr, OffsetPointer::GetNull());
  }

  /** Constant forward iterator begin */
  HSHM_CROSS_FUN
  citerator_t cbegin() const {
    if (size() == 0) {
      return cend();
    }
    auto head = GetAllocator()->template Convert<list_entry<T>>(head_ptr_);
    return citerator_t(const_cast<list &>(*this), head, head_ptr_);
  }

  /** Constant forward iterator end */
  HSHM_CROSS_FUN
  citerator_t cend() const {
    return iterator_t(const_cast<list &>(*this), nullptr,
                      OffsetPointer::GetNull());
  }

  /**====================================
   * Serialization
   * ===================================*/

  /** Serialize */
  template <typename Ar>
  HSHM_CROSS_FUN void save(Ar &ar) const {
    save_list<Ar, hipc::list<T, HSHM_CLASS_TEMPL_ARGS>, T>(ar, *this);
  }

  /** Deserialize */
  template <typename Ar>
  HSHM_CROSS_FUN void load(Ar &ar) {
    load_list<Ar, hipc::list<T, HSHM_CLASS_TEMPL_ARGS>, T>(ar, *this);
  }

 private:
  template <typename... Args>
  HSHM_INLINE_CROSS_FUN list_entry<T> *_create_entry(OffsetPointer &p,
                                                     Args &&...args) {
    auto full_ptr =
        GetAllocator()->template AllocateObjs<list_entry<T>, OffsetPointer>(GetMemCtx(), 1);
    auto entry = full_ptr.ptr_;
    p = full_ptr.shm_;
    entry->data_.shm_init(GetCtxAllocator(), std::forward<Args>(args)...);
    return entry;
  }
};

}  // namespace hshm::ipc

namespace hshm {

template <typename T, HSHM_CLASS_TEMPL_WITH_PRIV_DEFAULTS>
using list = hipc::list<T, HSHM_CLASS_TEMPL_ARGS>;

}  // namespace hshm

#undef CLASS_NAME
#undef CLASS_NEW_ARGS

#endif  // HSHM_DATA_STRUCTURES__LIST_H_
