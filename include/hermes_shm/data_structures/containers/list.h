//
// Created by llogan on 10/17/24.
//

#ifndef HERMES_SHM_INCLUDE_HERMES_SHM_DATA_STRUCTURES_CONTAINERS_LIST_H_
#define HERMES_SHM_INCLUDE_HERMES_SHM_DATA_STRUCTURES_CONTAINERS_LIST_H_

#include "hermes_shm/memory/memory_manager_.h"
#include "hermes_shm/data_structures/serialization/serialize_common.h"

namespace hshm {

/** forward pointer for list */
template<typename T>
class list;

/** represents an object within a list */
template<typename T>
struct list_entry {
 public:
  list_entry *next_ptr_, *prior_ptr_;
  T data_;
};

/**
 * The list iterator
 * */
template<typename T>
struct list_iterator_templ {
 public:
  /**< A shm reference to the containing list object. */
  list<T> *list_;
  /**< A pointer to the entry in shared memory */
  list_entry<T> *entry_;

  /** Default constructor */
  HSHM_CROSS_FUN
  list_iterator_templ() = default;

  /** Construct an iterator  */
  HSHM_CROSS_FUN
  explicit list_iterator_templ(list<T> &list,
                               list_entry<T> *entry)
  : list_(&list), entry_(entry) {}

  /** Copy constructor */
  HSHM_CROSS_FUN
  list_iterator_templ(const list_iterator_templ &other) {
    list_ = other.list_;
    entry_ = other.entry_;
  }

  /** Assign this iterator from another iterator */
  HSHM_CROSS_FUN
  list_iterator_templ& operator=(const list_iterator_templ &other) {
    if (this != &other) {
      list_ = other.list_;
      entry_ = other.entry_;
    }
    return *this;
  }

  /** Get the object the iterator points to */
  HSHM_CROSS_FUN
  T& operator*() {
    return entry_->data_;
  }

  /** Get the object the iterator points to */
  HSHM_CROSS_FUN
  const T& operator*() const {
    return entry_->data_;
  }

  /** Get the next iterator (in place) */
  HSHM_CROSS_FUN
  list_iterator_templ& operator++() {
    if (is_end()) { return *this; }
    entry_ = list_->GetAllocator()->template
        Convert<list_entry<T>>(entry_->next_ptr_);
    return *this;
  }

  /** Get the prior iterator (in place) */
  HSHM_CROSS_FUN
  list_iterator_templ& operator--() {
    if (is_end() || is_begin()) { return *this; }
    entry_ = list_->GetAllocator()->template
        Convert<list_entry<T>>(entry_->prior_ptr_);
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
  }

  /** Get the iterator at count before this one (in-place) */
  HSHM_CROSS_FUN
  void operator-=(size_t count) {
    list_iterator_templ pos = (*this) - count;
    entry_ = pos.entry_;
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
  bool is_end() const {
    return entry_ == nullptr;
  }

  /** Determine whether this iterator is the begin iterator */
  HSHM_CROSS_FUN
  bool is_begin() const {
    if (entry_) {
      return entry_->prior_ptr_ == nullptr;
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
#define TYPED_CLASS list<T>

/**
 * Doubly linked list implementation
 * */
template<typename T>
class list {
 public:
  HSHM_CONTAINER_BASE_TEMPLATE
  list_entry<T> *head_ptr_, *tail_ptr_;
  size_t length_;

 public:
  /**====================================
   * Typedefs
   * ===================================*/

  /** forward iterator typedef */
  typedef list_iterator_templ<T> iterator_t;
  /** const forward iterator typedef */
  typedef list_iterator_templ<T> citerator_t;

 public:
  /**====================================
   * Default Constructor
   * ===================================*/

  /** SHM constructor. Default. */
  HSHM_CROSS_FUN
  explicit list() {
    SetNull();
  }

  /** SHM constructor. Default. */
  HSHM_CROSS_FUN
  explicit list(Allocator *alloc) {
    init_private_container(alloc);
    SetNull();
  }

  /**====================================
   * Copy Constructors
   * ===================================*/

  /** SHM copy constructor */
  HSHM_CROSS_FUN
  explicit list(Allocator *alloc,
                const list &other) {
    init_private_container(alloc);
    SetNull();
    shm_strong_copy_construct_and_op<list>(other);
  }

  /** SHM copy assignment operator */
  HSHM_CROSS_FUN
  list& operator=(const list &other) {
    if (this != &other) {
      clear();
      shm_strong_copy_construct_and_op<list>(other);
    }
    return *this;
  }

  /** SHM copy constructor. From std::list */
  HSHM_CROSS_FUN
  explicit list(Allocator *alloc,
                std::list<T> &other) {
    init_private_container(alloc);
    SetNull();
    shm_strong_copy_construct_and_op<std::list<T>>(other);
  }

  /** SHM copy assignment operator. From std::list. */
  HSHM_CROSS_FUN
  list& operator=(const std::list<T> &other) {
    if (this != &other) {
      clear();
      shm_strong_copy_construct_and_op<std::list<T>>(other);
    }
    return *this;
  }

  /** SHM copy constructor + operator main */
  template<typename ListT>
  HSHM_CROSS_FUN
  void shm_strong_copy_construct_and_op(const ListT &other) {
    for (auto iter = other.cbegin(); iter != other.cend(); ++iter) {
      emplace_back(*iter);
    }
  }

  /**====================================
   * Move Constructors
   * ===================================*/

  /** SHM move constructor. */
  HSHM_CROSS_FUN
  list(Allocator *alloc, list &&other) noexcept {
    init_private_container(alloc);
    if (GetAllocator() == other.GetAllocator()) {
      memcpy((void*) this, (void *) &other, sizeof(*this));
      other = nullptr;
    } else {
      shm_strong_copy_construct_and_op<list>(other);
      other.clear();
    }
  }

  /** SHM move assignment operator. */
  HSHM_CROSS_FUN
  list& operator=(list &&other) noexcept {
    if (this != &other) {
      clear();
      if (GetAllocator() == other.GetAllocator()) {
        memcpy((void *) this, (void *) &other, sizeof(*this));
        other.SetNull();
      } else {
        shm_strong_copy_construct_and_op<list>(other);
        other.clear();
      }
    }
    return *this;
  }

  /**====================================
   * Destructor
   * ===================================*/

  /** SHM destructor.  */
  HSHM_CROSS_FUN
  void clear_main() {
    clear();
  }

  /** Check if the list is empty */
  HSHM_CROSS_FUN
  bool IsNull() const {
    return length_ == 0;
  }

  /** Sets this list as empty */
  HSHM_CROSS_FUN
  void SetNull() {
    length_ = 0;
    head_ptr_ = nullptr;
    tail_ptr_ = nullptr;
  }

  /**====================================
   * list Methods
   * ===================================*/

  /** Construct an element at the back of the list */
  template<typename... Args>
  HSHM_CROSS_FUN
  void emplace_back(Args&&... args) {
    emplace(end(), std::forward<Args>(args)...);
  }

  /** Construct an element at the beginning of the list */
  template<typename... Args>
  HSHM_CROSS_FUN
  void emplace_front(Args&&... args) {
    emplace(begin(), std::forward<Args>(args)...);
  }

  /** Construct an element at \a pos position in the list */
  template<typename ...Args>
  HSHM_CROSS_FUN
  void emplace(iterator_t pos, Args&&... args) {
    auto entry = _create_entry(std::forward<Args>(args)...);
    if (size() == 0) {
      entry->prior_ptr_ = nullptr;
      entry->next_ptr_ = nullptr;
      head_ptr_ = entry;
      tail_ptr_ = entry;
    } else if (pos.is_begin()) {
      entry->prior_ptr_ = nullptr;
      entry->next_ptr_ = head_ptr_;
      auto head = GetAllocator()->template
          Convert<list_entry<T>>(tail_ptr_);
      head->prior_ptr_ = entry;
      head_ptr_ = entry;
    } else if (pos.is_end()) {
      entry->prior_ptr_ = tail_ptr_;
      entry->next_ptr_ = nullptr;
      auto tail = GetAllocator()->template
          Convert<list_entry<T>>(tail_ptr_);
      tail->next_ptr_ = entry;
      tail_ptr_ = entry;
    } else {
      auto next = GetAllocator()->template
          Convert<list_entry<T>>(pos.entry_->next_ptr_);
      auto prior = GetAllocator()->template
          Convert<list_entry<T>>(pos.entry_->prior_ptr_);
      entry->next_ptr_ = pos.entry_->next_ptr_;
      entry->prior_ptr_ = pos.entry_->prior_ptr_;
      next->prior_ptr_ = entry;
      prior->next_ptr_ = entry;
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
  void erase(iterator_t pos) {
    erase(pos, pos+1);
  }

  /** Erase all elements between first and last */
  HSHM_CROSS_FUN
  void erase(iterator_t first,
             iterator_t last) {
    if (first.is_end()) { return; }
    auto first_prior_ptr = first.entry_->prior_ptr_;
    auto pos = first;
    while (pos != last) {
      auto next = pos + 1;
      GetAllocator()->Free(pos.entry_);
      --length_;
      pos = next;
    }

    if (first_prior_ptr == nullptr) {
      head_ptr_ = last.entry_;
    } else {
      auto first_prior = GetAllocator()->template
          Convert<list_entry<T>>(first_prior_ptr);
      first_prior->next_ptr_ = last.entry_;
    }

    if (last.entry_ == nullptr) {
      tail_ptr_ = first_prior_ptr;
    } else {
      last.entry_->prior_ptr_ = first_prior_ptr;
    }
  }

  /** Destroy all elements in the list */
  HSHM_CROSS_FUN
  void clear() {
    erase(begin(), end());
  }

  /** Get the object at the front of the list */
  HSHM_CROSS_FUN
  T& front() {
    return *begin();
  }

  /** Get the object at the back of the list */
  HSHM_CROSS_FUN
  T& back() {
    return *last();
  }

  /** Get the number of elements in the list */
  HSHM_CROSS_FUN
  size_t size() const {
    return length_;
  }

  /** Find an element in this list */
  HSHM_CROSS_FUN
  iterator_t find(const T &entry) {
    return hshm::find(begin(), end(), entry);
  }

  /**====================================
   * Iterators
   * ===================================*/

  /** Forward iterator begin */
  HSHM_CROSS_FUN
  iterator_t begin() {
    if (size() == 0) { return end(); }
    return iterator_t(*this, head_ptr_);
  }

  /** Last iterator begin */
  HSHM_CROSS_FUN
  iterator_t last() {
    if (size() == 0) { return end(); }
    return iterator_t(*this, tail_ptr_);
  }

  /** Forward iterator end */
  HSHM_CROSS_FUN
  iterator_t end() {
    return iterator_t(*this, nullptr);
  }

  /** Constant forward iterator begin */
  HSHM_CROSS_FUN
  citerator_t cbegin() const {
    if (size() == 0) { return cend(); }
    return citerator_t(const_cast<list&>(*this),
                       head_ptr_);
  }

  /** Constant forward iterator end */
  HSHM_CROSS_FUN
  citerator_t cend() const {
    return iterator_t(const_cast<list&>(*this), nullptr);
  }

  /**====================================
  * Serialization
  * ===================================*/

  /** Serialize */
  template <typename Ar>
  HSHM_CROSS_FUN
  void save(Ar &ar) const {
    save_list<Ar, hshm::list<T>, T>(ar, *this);
  }

  /** Deserialize */
  template <typename Ar>
  HSHM_CROSS_FUN
  void load(Ar &ar) {
    load_list<Ar, hshm::list<T>, T>(ar, *this);
  }

 private:
  template<typename ...Args>
  HSHM_INLINE_CROSS_FUN list_entry<T>* _create_entry(
      Args&& ...args) {
    auto entry = GetAllocator()->template
        AllocateObjs<list_entry<T>>(1);
    HSHM_MAKE_AR(entry->data_, GetAllocator(), std::forward<Args>(args)...)
    return entry;
  }
};

}  // namespace hshm

#endif  // HERMES_SHM_INCLUDE_HERMES_SHM_DATA_STRUCTURES_CONTAINERS_LIST_H_
