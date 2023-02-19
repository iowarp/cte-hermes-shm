/*
 * Copyright (C) 2022  SCS Lab <scslab@iit.edu>,
 * Luke Logan <llogan@hawk.iit.edu>,
 * Jaime Cernuda Garcia <jcernudagarcia@hawk.iit.edu>
 * Jay Lofstead <gflofst@sandia.gov>,
 * Anthony Kougkas <akougkas@iit.edu>,
 * Xian-He Sun <sun@iit.edu>
 *
 * This file is part of hermes
 *
 * hermes is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public
 * License along with this program.  If not, see
 * <http://www.gnu.org/licenses/>.
 */


#ifndef HERMES_DATA_STRUCTURES_THREAD_UNSAFE_SLIST_H_
#define HERMES_DATA_STRUCTURES_THREAD_UNSAFE_SLIST_H_

#include "hermes_shm/data_structures/data_structure.h"
#include "hermes_shm/data_structures/smart_ptr/manual_ptr.h"
#include "hermes_shm/data_structures/internal/shm_archive_or_t.h"

#include <list>

namespace hermes::ipc {

/** forward pointer for list */
template<typename T>
class slist;

/** represents an object within a list */
template<typename T>
struct slist_entry {
 public:
  OffsetPointer next_ptr_;
  ShmHeaderOrT<T> data_;

  /**
   * Constructor.
   * */
  template<typename ...Args>
  explicit slist_entry(Allocator *alloc, Args ...args)
  : data_(alloc, std::forward<Args>(args)...) {}

  /**
   * Returns the element stored in the list
   * */
  Ref<T> internal_ref(Allocator *alloc) {
    return Ref<T>(data_.internal_ref(alloc));
  }

  /** Destroys the shared-memory header*/
  void shm_destroy(Allocator *alloc) {
    data_->shm_destroy(alloc);
  }
};

/**
 * The list iterator
 * */
template<typename T, bool CONST_ITER>
struct slist_iterator_templ {
 public:
  typedef SHM_CONST_T_OR_T(slist<T>, CONST_ITER) ListT_Const;

 public:
  ListT_Const *list_;
  slist_entry<T> *entry_;
  OffsetPointer entry_ptr_;

  /** Default constructor */
  slist_iterator_templ() = default;

  /** Construct an iterator  */
  explicit slist_iterator_templ(ListT_Const *list)
  : list_(list), entry_(nullptr), entry_ptr_(OffsetPointer::GetNull()) {}

  /** Construct an iterator  */
  explicit slist_iterator_templ(ListT_Const *list,
                                slist_entry<T> *entry,
                                OffsetPointer entry_ptr)
    : list_(list), entry_(entry), entry_ptr_(entry_ptr) {}

  /** Copy constructor */
  slist_iterator_templ(const slist_iterator_templ &other) {
    list_ = other.list_;
    entry_ = other.entry_;
    entry_ptr_ = other.entry_ptr_;
  }

  /** Assign this iterator from another iterator */
  slist_iterator_templ& operator=(const slist_iterator_templ &other) {
    if (this != &other) {
      list_ = other.list_;
      entry_ = other.entry_;
      entry_ptr_ = other.entry_ptr_;
    }
    return *this;
  }

  /** Change the list pointer */
  void change_pointer(ListT_Const *other) {
    list_ = other;
  }

  /** Get the object the iterator points to */
  Ref<T> operator*() {
    return entry_->internal_ref(list_->GetAllocator());
  }

  /** Get the object the iterator points to */
  const Ref<T> operator*() const {
    return entry_->internal_ref(list_->GetAllocator());
  }

  /** Get the next iterator (in place) */
  slist_iterator_templ& operator++() {
    if (is_end()) { return *this; }
    entry_ptr_ = entry_->next_ptr_;
    entry_ = list_->alloc_->template
      Convert<slist_entry<T>>(entry_->next_ptr_);
    return *this;
  }

  /** Get the prior iterator (in place) */
  slist_iterator_templ& operator--() {
    if (is_end() || is_begin()) { return *this; }
    entry_ptr_ = entry_->prior_ptr_;
    entry_ = list_->alloc_->template
      Convert<slist_entry<T>>(entry_->prior_ptr_);
    return *this;
  }

  /** Return the next iterator */
  slist_iterator_templ operator++(int) const {
    slist_iterator_templ next_iter(*this);
    ++next_iter;
    return next_iter;
  }

  /** Return the prior iterator */
  slist_iterator_templ operator--(int) const {
    slist_iterator_templ prior_iter(*this);
    --prior_iter;
    return prior_iter;
  }

  /** Return the iterator at count after this one */
  slist_iterator_templ operator+(size_t count) const {
    slist_iterator_templ pos(*this);
    for (size_t i = 0; i < count; ++i) {
      ++pos;
    }
    return pos;
  }

  /** Return the iterator at count before this one */
  slist_iterator_templ operator-(size_t count) const {
    slist_iterator_templ pos(*this);
    for (size_t i = 0; i < count; ++i) {
      --pos;
    }
    return pos;
  }

  /** Get the iterator at count after this one (in-place) */
  void operator+=(size_t count) {
    slist_iterator_templ pos = (*this) + count;
    entry_ = pos.entry_;
    entry_ptr_ = pos.entry_ptr_;
  }

  /** Get the iterator at count before this one (in-place) */
  void operator-=(size_t count) {
    slist_iterator_templ pos = (*this) - count;
    entry_ = pos.entry_;
    entry_ptr_ = pos.entry_ptr_;
  }

  /** Determine if two iterators are equal */
  friend bool operator==(const slist_iterator_templ &a,
                         const slist_iterator_templ &b) {
    return (a.is_end() && b.is_end()) || (a.entry_ == b.entry_);
  }

  /** Determine if two iterators are inequal */
  friend bool operator!=(const slist_iterator_templ &a,
                         const slist_iterator_templ &b) {
    return !(a.is_end() && b.is_end()) && (a.entry_ != b.entry_);
  }

  /** Create the end iterator */
  static slist_iterator_templ const end() {
    static slist_iterator_templ end_iter(nullptr);
    return end_iter;
  }

  /** Determine whether this iterator is the end iterator */
  bool is_end() const {
    return list_ == nullptr;
  }

  /** Determine whether this iterator is the begin iterator */
  bool is_begin() const {
    if (list_ && entry_) {
      return entry_ptr_ == list_->header_->head_ptr_;
    } else {
      return false;
    }
  }
};

/** forward iterator typedef */
template<typename T>
using slist_iterator = slist_iterator_templ<T, false>;

/** const forward iterator typedef */
template<typename T>
using slist_citerator = slist_iterator_templ<T, true>;


/**
 * MACROS used to simplify the list namespace
 * Used as inputs to the SHM_CONTAINER_TEMPLATE
 * */
#define CLASS_NAME slist
#define TYPED_CLASS slist<T>
#define TYPED_HEADER ShmHeader<slist<T>>

/**
 * The list shared-memory header
 * */
template<typename T>
struct ShmHeader<TYPED_CLASS> : public ShmBaseHeader {
  OffsetPointer head_ptr_, tail_ptr_;
  size_t length_;

  ShmHeader() = default;

  ShmHeader(const ShmHeader &other) {
    head_ptr_ = other.head_ptr_;
    tail_ptr_ = other.tail_ptr_;
    length_ = other.length_;
  }
};

/**
 * Doubly linked list implementation
 * */
template<typename T>
class slist : public ShmContainer {
 public:
  SHM_CONTAINER_TEMPLATE((CLASS_NAME), (TYPED_CLASS), (TYPED_HEADER))

 public:
  ////////////////////////////
  /// SHM Overrides
  ////////////////////////////

  /** Default constructor */
  slist() = default;

  /** Initialize list in shared memory */
  void shm_init_main(TYPED_HEADER *header,
                     Allocator *alloc) {
    shm_init_allocator(alloc);
    shm_init_header(header);
    header_->length_ = 0;
    header_->head_ptr_.SetNull();
    header_->tail_ptr_.SetNull();
  }

  /** Copy from std::list */
  void shm_init_main(TYPED_HEADER *header,
                     Allocator *alloc, std::list<T> &other) {
    shm_init_allocator(alloc);
    shm_init_header(header);
    for (auto &entry : other) {
      emplace_back(entry);
    }
  }

  /** Destroy all shared memory allocated by the list */
  void shm_destroy_main() {
    clear();
  }

  /** Store into shared memory */
  void shm_serialize_main() const {}

  /** Load from shared memory */
  void shm_deserialize_main() {}

  /** Move constructor */
  void shm_weak_move_main(TYPED_HEADER *header,
                          Allocator *alloc, slist &other) {
    shm_init_allocator(alloc);
    shm_init_header(header);
    *header_ = *(other.header_);
  }

  /** Copy constructor */
  void shm_strong_copy_main(TYPED_HEADER *header,
                            Allocator *alloc, const slist &other) {
    shm_init_allocator(alloc);
    shm_init_header(header);
    for (auto iter = other.cbegin(); iter != other.cend(); ++iter) {
      emplace_back(**iter);
    }
  }

  ////////////////////////////
  /// SList Operators
  ////////////////////////////

  /** Construct an element at the back of the list */
  template<typename... Args>
  inline void emplace_back(Args&&... args) {
    emplace(end(), std::forward<Args>(args)...);
  }

  /** Construct an element at the beginning of the list */
  template<typename... Args>
  inline void emplace_front(Args&&... args) {
    emplace(begin(), std::forward<Args>(args)...);
  }

  /** Construct an element at \a pos position in the list */
  template<typename ...Args>
  void emplace(slist_iterator<T> pos, Args&&... args) {
    OffsetPointer entry_ptr;
    auto entry = _create_entry(entry_ptr, std::forward<Args>(args)...);
    if (size() == 0) {
      entry->next_ptr_.SetNull();
      header_->head_ptr_ = entry_ptr;
      header_->tail_ptr_ = entry_ptr;
    } else if (pos.is_begin()) {
      entry->next_ptr_ = header_->head_ptr_;
      header_->head_ptr_ = entry_ptr;
    } else if (pos.is_end()) {
      entry->next_ptr_.SetNull();
      auto tail = alloc_->template
        Convert<slist_entry<T>>(header_->tail_ptr_);
      tail->next_ptr_ = entry_ptr;
      header_->tail_ptr_ = entry_ptr;
    } else {
      auto prior = find_prior(entry);
      prior->next_ptr_ = entry_ptr;
      entry->next_ptr_ = pos.entry_->next_ptr_;
    }
    ++header_->length_;
  }

  /** Erase the element at pos */
  void erase(slist_iterator<T> pos) {
    erase(pos, pos+1);
  }

  /** Erase all elements between first and last */
  void erase(slist_iterator<T> first,
             slist_iterator<T> last) {
    auto first_prior = find_prior(first.entry_);
    auto pos = first;
    while (pos != last) {
      auto next = pos + 1;
      Allocator::DestructObj<slist_entry<T>>(*pos.entry_);
      alloc_->Free(pos.entry_ptr_);
      --header_->length_;
      pos = next;
    }

    if (first_prior == nullptr) {
      header_->head_ptr_ = last.entry_ptr_;
    } else {
      first_prior->next_ptr_ = last.entry_ptr_;
    }

    if (last.entry_ptr_.IsNull()) {
      header_->tail_ptr_ = alloc_->template
        Convert<void, OffsetPointer>(first_prior);
    }
  }

  /** Destroy all elements in the list */
  void clear() {
    erase(begin(), end());
  }

  /** Get the object at the front of the list */
  inline Ref<T> front() {
    return *begin();
  }

  /** Get the object at the back of the list */
  inline Ref<T> back() {
    return *end();
  }

  /** Get the number of elements in the list */
  inline size_t size() const {
    if (!IsNull()) {
      return header_->length_;
    }
    return 0;
  }

 private:
  slist_entry<T>* find_prior(slist_entry<T> *entry) {
    slist_entry<T>* pos = alloc_->template
      Convert<slist_entry<T>>(header_->head_ptr_);
    while (pos) {
      slist_entry<T>* next = alloc_->template
        Convert<slist_entry<T>>(pos->next_ptr_);
      if (next == entry) {
        return pos;
      }
      pos = entry;
    }
    return nullptr;
  }

 public:

  ////////////////////////////
  /// Iterators
  ////////////////////////////

  /** Forward iterator begin */
  inline slist_iterator<T> begin() {
    if (size() == 0) { return end(); }
    auto head = alloc_->template
      Convert<slist_entry<T>>(header_->head_ptr_);
    return slist_iterator<T>(this, head, header_->head_ptr_);
  }

  /** Forward iterator end */
  static slist_iterator<T> const end() {
    return slist_iterator<T>::end();
  }

  /** Constant forward iterator begin */
  inline slist_citerator<T> cbegin() const {
    if (size() == 0) { return cend(); }
    auto head = alloc_->template
      Convert<slist_entry<T>>(header_->head_ptr_);
    return slist_citerator<T>(this, head, header_->head_ptr_);
  }

  /** Constant forward iterator end */
  static slist_citerator<T> const cend() {
    return slist_citerator<T>::end();
  }

 private:
  template<typename ...Args>
  inline slist_entry<T>* _create_entry(OffsetPointer &p, Args&& ...args) {
    auto entry = alloc_->template
      AllocateConstructObjs<slist_entry<T>>(1, p, std::forward<Args>(args)...);
    return entry;
  }
};

}  // namespace hermes::ipc

#undef CLASS_NAME
#undef TYPED_CLASS
#undef TYPED_HEADER

#endif  // HERMES_DATA_STRUCTURES_THREAD_UNSAFE_SLIST_H_
