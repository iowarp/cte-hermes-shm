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

#ifndef HSHM_DATA_STRUCTURES_LOCKLESS_VECTOR_H_
#define HSHM_DATA_STRUCTURES_LOCKLESS_VECTOR_H_

#include <vector>

#include "hermes_shm/data_structures/internal/shm_internal.h"
#include "hermes_shm/data_structures/serialization/serialize_common.h"

namespace hshm::ipc {

/** forward pointer for vector_base */
template <typename T, HSHM_CLASS_TEMPL_WITH_DEFAULTS>
class vector_base;

/**
 * The vector_base iterator implementation
 * */
template <typename T, bool FORWARD_ITER, HSHM_CLASS_TEMPL>
struct vector_iterator_templ {
 public:
  vector_base<T, HSHM_CLASS_TEMPL_ARGS> *vec_;
  i64 i_;

  /** Default constructor */
  HSHM_INLINE_CROSS_FUN vector_iterator_templ() = default;

  /** Construct an iterator (called from vector_base class) */
  template <typename SizeT>
  HSHM_INLINE_CROSS_FUN explicit vector_iterator_templ(
      vector_base<T, HSHM_CLASS_TEMPL_ARGS> *vec, SizeT i)
      : vec_(vec), i_(static_cast<i64>(i)) {}

  /** Construct an iterator (called from iterator) */
  HSHM_INLINE_CROSS_FUN explicit vector_iterator_templ(
      vector_base<T, HSHM_CLASS_TEMPL_ARGS> *vec, i64 i)
      : vec_(vec), i_(i) {}

  /** Copy constructor */
  HSHM_INLINE_CROSS_FUN vector_iterator_templ(
      const vector_iterator_templ &other)
      : vec_(other.vec_), i_(other.i_) {}

  /** Copy assignment operator  */
  HSHM_INLINE_CROSS_FUN vector_iterator_templ &operator=(
      const vector_iterator_templ &other) {
    if (this != &other) {
      vec_ = other.vec_;
      i_ = other.i_;
    }
    return *this;
  }

  /** Move constructor */
  HSHM_INLINE_CROSS_FUN vector_iterator_templ(
      vector_iterator_templ &&other) noexcept {
    vec_ = other.vec_;
    i_ = other.i_;
  }

  /** Move assignment operator  */
  HSHM_INLINE_CROSS_FUN vector_iterator_templ &operator=(
      vector_iterator_templ &&other) noexcept {
    if (this != &other) {
      vec_ = other.vec_;
      i_ = other.i_;
    }
    return *this;
  }

  /** Dereference the iterator */
  HSHM_INLINE_CROSS_FUN T &operator*() { return vec_->data_ar()[i_].get_ref(); }

  /** Dereference the iterator */
  HSHM_INLINE_CROSS_FUN const T &operator*() const {
    return vec_->data_ar()[i_].get_ref();
  }

  /** Increment iterator in-place */
  HSHM_INLINE_CROSS_FUN vector_iterator_templ &operator++() {
    if constexpr (FORWARD_ITER) {
      ++i_;
    } else {
      --i_;
    }
    return *this;
  }

  /** Decrement iterator in-place */
  HSHM_INLINE_CROSS_FUN vector_iterator_templ &operator--() {
    if constexpr (FORWARD_ITER) {
      --i_;
    } else {
      ++i_;
    }
    return *this;
  }

  /** Create the next iterator */
  HSHM_INLINE_CROSS_FUN vector_iterator_templ operator++(int) const {
    vector_iterator_templ next_iter(*this);
    ++next_iter;
    return next_iter;
  }

  /** Create the prior iterator */
  HSHM_INLINE_CROSS_FUN vector_iterator_templ operator--(int) const {
    vector_iterator_templ prior_iter(*this);
    --prior_iter;
    return prior_iter;
  }

  /** Increment iterator by \a count and return */
  HSHM_INLINE_CROSS_FUN vector_iterator_templ operator+(size_t count) const {
    if constexpr (FORWARD_ITER) {
      return vector_iterator_templ(vec_, i_ + count);
    } else {
      return vector_iterator_templ(vec_, i_ - count);
    }
  }

  /** Decrement iterator by \a count and return */
  HSHM_INLINE_CROSS_FUN vector_iterator_templ operator-(size_t count) const {
    if constexpr (FORWARD_ITER) {
      return vector_iterator_templ(vec_, i_ - count);
    } else {
      return vector_iterator_templ(vec_, i_ + count);
    }
  }

  /** Difference between two iterators */
  HSHM_INLINE_CROSS_FUN friend i64 operator-(const vector_iterator_templ &a,
                                             const vector_iterator_templ &b) {
    return (a.i_ - b.i_);
  }

  /** Increment iterator by \a count in-place */
  HSHM_INLINE_CROSS_FUN void operator+=(size_t count) {
    if constexpr (FORWARD_ITER) {
      i_ += count;
    } else {
      i_ -= count;
    }
  }

  /** Decrement iterator by \a count in-place */
  HSHM_INLINE_CROSS_FUN void operator-=(size_t count) {
    if constexpr (FORWARD_ITER) {
      i_ -= count;
    } else {
      i_ += count;
    }
  }

  /** Check if two iterators are equal */
  HSHM_INLINE_CROSS_FUN friend bool operator==(const vector_iterator_templ &a,
                                               const vector_iterator_templ &b) {
    return (a.i_ == b.i_);
  }

  /** Check if two iterators are inequal */
  HSHM_INLINE_CROSS_FUN friend bool operator!=(const vector_iterator_templ &a,
                                               const vector_iterator_templ &b) {
    return (a.i_ != b.i_);
  }

  /** Less than operator */
  HSHM_INLINE_CROSS_FUN friend bool operator<(const vector_iterator_templ &a,
                                              const vector_iterator_templ &b) {
    return (a.i_ < b.i_);
  }

  /** Greater than operator */
  HSHM_INLINE_CROSS_FUN friend bool operator>(const vector_iterator_templ &a,
                                              const vector_iterator_templ &b) {
    return (a.i_ > b.i_);
  }

  /** Less than or equal operator */
  HSHM_INLINE_CROSS_FUN friend bool operator<=(const vector_iterator_templ &a,
                                               const vector_iterator_templ &b) {
    return (a.i_ <= b.i_);
  }

  /** Greater than or equal operator */
  HSHM_INLINE_CROSS_FUN friend bool operator>=(const vector_iterator_templ &a,
                                               const vector_iterator_templ &b) {
    return (a.i_ >= b.i_);
  }

  /** Set this iterator to end */
  HSHM_INLINE_CROSS_FUN void set_end() {
    if constexpr (FORWARD_ITER) {
      i_ = vec_->size();
    } else {
      i_ = -1;
    }
  }

  /** Set this iterator to begin */
  HSHM_INLINE_CROSS_FUN void set_begin() {
    if constexpr (FORWARD_ITER) {
      i_ = 0;
    } else {
      i_ = vec_->size() - 1;
    }
  }

  /** Determine whether this iterator is the begin iterator */
  HSHM_INLINE_CROSS_FUN bool is_begin() const {
    if constexpr (FORWARD_ITER) {
      return (i_ == 0);
    } else {
      return (i_ == vec_->template size<i64>() - 1);
    }
  }

  /** Determine whether this iterator is the end iterator */
  HSHM_INLINE_CROSS_FUN bool is_end() const {
    if constexpr (FORWARD_ITER) {
      return i_ >= vec_->template size<i64>();
    } else {
      return i_ == -1;
    }
  }
};

/**
 * MACROS used to simplify the vector_base namespace
 * Used as inputs to the HIPC_CONTAINER_TEMPLATE
 * */
#define CLASS_NAME vector_base
#define CLASS_NEW_ARGS T

/**
 * The vector_base class
 * */
template <typename T, HSHM_CLASS_TEMPL>
class vector_base : public ShmContainer {
 public:
  HIPC_CONTAINER_TEMPLATE((CLASS_NAME), (CLASS_NEW_ARGS))

 public:
  /**====================================
   * Typedefs
   * ===================================*/

  /** forwrard iterator */
  typedef vector_iterator_templ<T, true, HSHM_CLASS_TEMPL_ARGS> iterator_t;
  /** reverse iterator */
  typedef vector_iterator_templ<T, false, HSHM_CLASS_TEMPL_ARGS> riterator_t;
  /** const iterator */
  typedef vector_iterator_templ<T, true, HSHM_CLASS_TEMPL_ARGS> citerator_t;
  /** const reverse iterator */
  typedef vector_iterator_templ<T, false, HSHM_CLASS_TEMPL_ARGS> criterator_t;

 public:
  /**====================================
   * Variables
   * ===================================*/
  OffsetPointer vec_ptr_;
  size_t max_length_, length_;

 public:
  /**====================================
   * Default Constructor
   * ===================================*/

  /** SHM constructor. Default. */
  HSHM_CROSS_FUN
  explicit vector_base() {
    init_shm_container(HSHM_MEMORY_MANAGER->GetDefaultAllocator<AllocT>());
    SetNull();
  }

  /** SHM constructor. Default. */
  HSHM_CROSS_FUN
  explicit vector_base(const hipc::CtxAllocator<AllocT> &alloc) {
    init_shm_container(alloc);
    SetNull();
  }

  /** Constructor. Resize + construct. */
  template <typename... Args>
  HSHM_CROSS_FUN explicit vector_base(size_t length, Args &&...args) {
    shm_init(HSHM_MEMORY_MANAGER->GetDefaultAllocator<AllocT>(), length,
             std::forward<Args>(args)...);
  }

  /** SHM constructor. Resize + construct. */
  template <typename... Args>
  HSHM_CROSS_FUN explicit vector_base(const hipc::CtxAllocator<AllocT> &alloc,
                                 size_t length, Args &&...args) {
    shm_init(alloc, length, std::forward<Args>(args)...);
  }

  /** Constructor */
  template <typename... Args>
  HSHM_CROSS_FUN void shm_init(const CtxAllocator<AllocT> &tls_alloc,
                               size_t length, Args &&...args) {
    init_shm_container(tls_alloc);
    SetNull();
    resize(length, std::forward<Args>(args)...);
  }

  /**====================================
   * Copy Constructors
   * ===================================*/

  /** Copy constructor. From vector_base. */
  HSHM_CROSS_FUN
  explicit vector_base(const vector_base &other) {
    init_shm_container(other.GetCtxAllocator());
    SetNull();
    shm_strong_copy_main<vector_base<T, HSHM_CLASS_TEMPL_ARGS>>(other);
  }

  /** SHM copy constructor. From vector_base. */
  HSHM_CROSS_FUN
  explicit vector_base(const hipc::CtxAllocator<AllocT> &alloc,
                  const vector_base &other) {
    init_shm_container(alloc);
    SetNull();
    shm_strong_copy_main<vector_base<T, HSHM_CLASS_TEMPL_ARGS>>(other);
  }

  /** SHM copy assignment operator. From vector_base. */
  HSHM_CROSS_FUN
  vector_base &operator=(const vector_base &other) {
    if (this != &other) {
      shm_destroy();
      shm_strong_copy_main<vector_base>(other);
    }
    return *this;
  }

  /** Copy constructor. From std::vector */
  HSHM_HOST_FUN
  explicit vector_base(const std::vector<T> &other) {
    init_shm_container(other.GetCtxAllocator());
    SetNull();
    shm_strong_copy_main<std::vector<T>>(other);
  }

  /** SHM copy constructor. From std::vector */
  HSHM_HOST_FUN
  explicit vector_base(const hipc::CtxAllocator<AllocT> &alloc,
                  const std::vector<T> &other) {
    init_shm_container(alloc);
    SetNull();
    shm_strong_copy_main<std::vector<T>>(other);
  }

  /** SHM copy assignment operator. From std::vector */
  HSHM_HOST_FUN
  vector_base &operator=(const std::vector<T> &other) {
    shm_destroy();
    shm_strong_copy_main<std::vector<T>>(other);
    return *this;
  }

  /** The main copy operation  */
  template <typename VectorT>
  HSHM_CROSS_FUN void shm_strong_copy_main(const VectorT &other) {
    reserve(other.size());
    if constexpr (std::is_pod<T>() && !IS_SHM_ARCHIVEABLE(T)) {
      memcpy(data(), other.data(), other.size() * sizeof(T));
      length_ = other.size();
    } else {
      for (auto iter = other.cbegin(); iter != other.cend(); ++iter) {
        emplace_back((*iter));
      }
    }
  }

  /**====================================
   * Move Constructors
   * ===================================*/

  /** Move constructor. */
  HSHM_CROSS_FUN
  vector_base(vector_base &&other) {
    shm_move_op<false>(HSHM_MEMORY_MANAGER->GetDefaultAllocator<AllocT>(),
                       std::move(other));
  }

  /** SHM move constructor. */
  HSHM_CROSS_FUN
  vector_base(const hipc::CtxAllocator<AllocT> &alloc, vector_base &&other) {
    shm_move_op<false>(alloc, std::move(other));
  }

  /** SHM move assignment operator. */
  HSHM_CROSS_FUN
  vector_base &operator=(vector_base &&other) noexcept {
    if (this != &other) {
      shm_move_op<true>(other.GetCtxAllocator(), std::move(other));
    }
    return *this;
  }

  /** SHM move assignment operator. */
  template <bool IS_ASSIGN>
  HSHM_CROSS_FUN void shm_move_op(const hipc::CtxAllocator<AllocT> &alloc,
                                  vector_base &&other) noexcept {
    if constexpr (!IS_ASSIGN) {
      init_shm_container(alloc);
    }
    if (GetAllocator() == other.GetAllocator()) {
      memcpy((void *)this, (void *)&other, sizeof(*this));
      other.SetNull();
    } else {
      shm_strong_copy_main<vector_base>(other);
      other.shm_destroy();
    }
  }

  /**====================================
   * Destructor
   * ===================================*/

  /** Check if null */
  HSHM_INLINE_CROSS_FUN
  bool IsNull() const { return vec_ptr_.IsNull(); }

  /** Make null */
  HSHM_INLINE_CROSS_FUN
  void SetNull() {
    length_ = 0;
    max_length_ = 0;
    vec_ptr_.SetNull();
  }

  /** Destroy all shared memory allocated by the vector */
  HSHM_INLINE_CROSS_FUN
  void shm_destroy_main() {
    erase(begin(), end());
    CtxAllocator<AllocT> alloc = GetCtxAllocator();
    FullPtr<void, OffsetPointer> full_ptr(alloc->template Convert<void>(vec_ptr_), vec_ptr_);
    alloc->Free(alloc.ctx_, full_ptr);
  }

  /**====================================
   * Vector Operations
   * ===================================*/

  /**
   * Convert to std::vector
   * */
  HSHM_INLINE_HOST_FUN
  std::vector<T> vec() {
    std::vector<T> v;
    v.reserve(size());
    for (T &entry : *this) {
      v.emplace_back(entry);
    }
    return v;
  }

  /**
   * Reserve space in the vector to emplace elements. Does not
   * change the size of the list.
   *
   * @param length the maximum size the vector can get before a growth occurs
   * @param args the arguments to construct
   * */
  template <typename... Args>
  HSHM_INLINE_CROSS_FUN void reserve(size_t length, Args &&...args) {
    if (length == 0) {
      return;
    }
    grow_vector(data_ar(), length, false, std::forward<Args>(args)...);
  }

  /**
   * Reserve space in the vector to emplace elements. Changes the
   * size of the list.
   *
   * @param length the maximum size the vector can get before a growth occurs
   * @param args the arguments used to construct the vector elements
   * */
  template <typename... Args>
  HSHM_CROSS_FUN void resize(size_t length, Args &&...args) {
    if (length == 0) {
      length_ = 0;
      return;
    }
    grow_vector(data_ar(), length, true, std::forward<Args>(args)...);
    length_ = length;
  }

  /** Index the vector at position i */
  HSHM_INLINE_CROSS_FUN
  T &operator[](const size_t i) { return data_ar()[i].get_ref(); }

  /** Index the vector at position i */
  HSHM_INLINE_CROSS_FUN
  const T &operator[](const size_t i) const { return data_ar()[i].get_ref(); }

  /** Get first element of vector */
  HSHM_INLINE_CROSS_FUN
  T &front() { return (*this)[0]; }

  /** Get last element of vector */
  HSHM_INLINE_CROSS_FUN
  T &back() { return (*this)[size() - 1]; }

  /** Get first element of vector */
  HSHM_INLINE_CROSS_FUN
  const T &front() const { return (*this)[0]; }

  /** Get last element of vector */
  HSHM_INLINE_CROSS_FUN
  const T &back() const { return (*this)[size() - 1]; }

  /** Pop element at back of vector  */
  HSHM_INLINE_CROSS_FUN
  void pop_back() {
    if (length_ == 0) return;
    hipc::Allocator::DestructObj(back());
    --length_;
  }

  /** Construct an element at the back of the vector */
  template <typename... Args>
  HSHM_CROSS_FUN void emplace_back(Args &&...args) {
    delay_ar<T> *vec = data_ar();
    if (length_ == max_length_) {
      vec = grow_vector(vec, 0, false);
    }
    vec[length_].shm_init(GetCtxAllocator(), std::forward<Args>(args)...);
    ++length_;
  }

  /** Assign elements to vector using iterator */
  template <typename Iterator>
  HSHM_INLINE_CROSS_FUN void assign(Iterator first, Iterator last) {
    for (auto iter = first; iter != last; ++iter) {
      emplace_back(*iter);
    }
  }

  /** Assign elements to vector using iterator up to maximum size */
  template <typename Iterator>
  HSHM_INLINE_CROSS_FUN void assign(Iterator first, Iterator last,
                                    int max_count) {
    for (auto iter = first; iter != last && size() < max_count; ++iter) {
      emplace_back(*iter);
    }
  }

  /** Construct an element in the front of the vector */
  template <typename... Args>
  HSHM_INLINE_CROSS_FUN void emplace_front(Args &&...args) {
    emplace(begin(), std::forward<Args>(args)...);
  }

  /** Construct an element at an arbitrary position in the vector */
  template <typename... Args>
  HSHM_CROSS_FUN void emplace(iterator_t pos, Args &&...args) {
    if (pos.is_end()) {
      emplace_back(std::forward<Args>(args)...);
      return;
    }
    delay_ar<T> *vec = data_ar();
    if (length_ == max_length_) {
      vec = grow_vector(vec, 0, false);
    }
    shift_right(pos);
    vec[pos.i_].shm_init(GetCtxAllocator(), std::forward<Args>(args)...);
    ++length_;
  }

  /** Replace an element at a position */
  template <typename... Args>
  HSHM_INLINE_CROSS_FUN void replace(iterator_t pos, Args &&...args) {
    if (pos.is_end()) {
      return;
    }
    delay_ar<T> *vec = data_ar();
    hipc::Allocator::DestructObj((*this)[(size_t)pos.i_]);
    vec[pos.i_].shm_init(GetCtxAllocator(), std::forward<Args>(args)...);
  }

  /** Delete the element at \a pos position */
  HSHM_INLINE_CROSS_FUN
  void erase(iterator_t pos) {
    if (pos.is_end()) return;
    shift_left(pos, 1);
    length_ -= 1;
  }

  /** Delete elements between first and last  */
  HSHM_INLINE_CROSS_FUN
  void erase(iterator_t first, iterator_t last) {
    i64 last_i;
    if (first.is_end()) return;
    if (last.is_end()) {
      last_i = size();
    } else {
      last_i = last.i_;
    }
    size_t count = (size_t)(last_i - first.i_);
    if (count == 0) return;
    shift_left(first, count);
    length_ -= count;
  }

  /** Delete all elements from the vector */
  HSHM_INLINE_CROSS_FUN
  void clear() { erase(begin(), end()); }

  /** Get the size of the vector */
  template <typename SizeT = size_t>
  HSHM_INLINE_CROSS_FUN SizeT size() const {
    return static_cast<SizeT>(length_);
  }

  /** Get the max size of the vector */
  template <typename SizeT = size_t>
  HSHM_INLINE_CROSS_FUN SizeT capacity() const {
    return static_cast<SizeT>(max_length_);
  }

  /** Get the data in the vector */
  HSHM_INLINE_CROSS_FUN void *data() {
    return reinterpret_cast<void *>(data_ar());
  }

  /** Get constant pointer to the data */
  HSHM_INLINE_CROSS_FUN void *data() const {
    return reinterpret_cast<void *>(data_ar());
  }

  /** Retreives a pointer to the internal array */
  HSHM_INLINE_CROSS_FUN delay_ar<T> *data_ar() {
    return GetAllocator()->template Convert<delay_ar<T>>(vec_ptr_);
  }

  /** Retreives a pointer to the array */
  HSHM_INLINE_CROSS_FUN delay_ar<T> *data_ar() const {
    return GetAllocator()->template Convert<delay_ar<T>>(vec_ptr_);
  }

  /**====================================
   * Internal Operations
   * ===================================*/
 private:
  /**
   * Grow a vector to a new size.
   *
   * @param vec the C-style array of elements to grow
   * @param max_length the new length of the vector. If 0, the current size
   * of the vector will be multiplied by a constant.
   * @param args the arguments used to construct the elements of the vector
   * */
  template <typename... Args>
  HSHM_CROSS_FUN delay_ar<T> *grow_vector(delay_ar<T> *vec, size_t max_length,
                                          bool resize, Args &&...args) {
    // Grow vector by 25%
    if (max_length == 0) {
      max_length = 5 * max_length_ / 4;
      if (max_length <= max_length_ + 10) {
        max_length += 10;
      }
    }
    if (max_length < max_length_) {
      return nullptr;
    }

    // Allocate new shared-memory vec
    delay_ar<T> *new_vec;
    CtxAllocator<AllocT> alloc = GetCtxAllocator();
    if constexpr (std::is_pod<T>() && !IS_SHM_ARCHIVEABLE(T)) {
      // Use reallocate for well-behaved objects
      FullPtr<delay_ar<T>, OffsetPointer> vec_full_ptr(
          alloc->template Convert<delay_ar<T>>(vec_ptr_), vec_ptr_);
      auto result_ptr = alloc->template ReallocateObjs<delay_ar<T>>(
          alloc.ctx_, vec_full_ptr, max_length);
      new_vec = result_ptr.ptr_;
      vec_ptr_ = result_ptr.shm_;
    } else {
      // Use std::move for unpredictable objects
      auto full_ptr = alloc->template AllocateObjs<delay_ar<T>, OffsetPointer>(alloc.ctx_,
                                                          max_length);
      new_vec = full_ptr.ptr_;
      OffsetPointer new_p = full_ptr.shm_;
      for (size_t i = 0; i < length_; ++i) {
        T &old_entry = (*this)[i];
        new_vec[i].shm_init(alloc, std::move(old_entry));
      }
      if (!vec_ptr_.IsNull()) {
        FullPtr<void, OffsetPointer> old_full_ptr(alloc->template Convert<void>(vec_ptr_), vec_ptr_);
        alloc->Free(alloc.ctx_, old_full_ptr);
      }
      vec_ptr_ = new_p;
    }
    if (new_vec == nullptr) {
      HSHM_THROW_ERROR(OUT_OF_MEMORY, max_length * sizeof(delay_ar<T>),
                       alloc->GetCurrentlyAllocatedSize());
    }
    if (resize) {
      for (size_t i = length_; i < max_length; ++i) {
        new_vec[i].shm_init(alloc, std::forward<Args>(args)...);
      }
    }

    // Update vector header
    max_length_ = max_length;
    return new_vec;
  }

  /**
   * Shift every element starting at "pos" to the left by count. Any element
   * who would be shifted before "pos" will be deleted.
   *
   * @param pos the starting position
   * @param count the amount to shift left by
   * */
  HSHM_INLINE_CROSS_FUN void shift_left(const iterator_t pos,
                                        size_t count = 1) {
    delay_ar<T> *vec = data_ar();
    for (size_t i = 0; i < count; ++i) {
      HSHM_DESTROY_AR(vec[pos.i_ + i])
    }
    auto dst = vec + pos.i_;
    auto src = dst + count;
    for (auto i = pos.i_ + count; i < size(); ++i) {
      memcpy((void *)dst, (void *)src, sizeof(delay_ar<T>));
      dst += 1;
      src += 1;
    }
  }

  /**
   * Shift every element starting at "pos" to the right by count. Increases
   * the total number of elements of the vector by "count". Does not modify
   * the size parameter of the vector, this is done elsewhere.
   *
   * @param pos the starting position
   * @param count the amount to shift right by
   * */
  HSHM_INLINE_CROSS_FUN void shift_right(const iterator_t pos,
                                         size_t count = 1) {
    auto src = data_ar() + size() - 1;
    auto dst = src + count;
    auto sz = static_cast<i64>(size());
    for (auto i = sz - 1; i >= pos.i_; --i) {
      memcpy((void *)dst, (void *)src, sizeof(delay_ar<T>));
      dst -= 1;
      src -= 1;
    }
  }

  /**====================================
   * Iterators
   * ===================================*/
 public:
  /** Beginning of the forward iterator */
  HSHM_INLINE_CROSS_FUN iterator_t begin() { return iterator_t(this, 0); }

  /** End of the forward iterator */
  HSHM_INLINE_CROSS_FUN iterator_t end() { return iterator_t(this, size()); }

  /** Beginning of the constant forward iterator */
  HSHM_INLINE_CROSS_FUN citerator_t cbegin() const {
    return citerator_t(const_cast<vector_base *>(this), 0);
  }

  /** End of the forward iterator */
  HSHM_INLINE_CROSS_FUN citerator_t cend() const {
    return citerator_t(const_cast<vector_base *>(this), size<i64>());
  }

  /** Beginning of the reverse iterator */
  HSHM_INLINE_CROSS_FUN riterator_t rbegin() {
    return riterator_t(this, size<i64>() - 1);
  }

  /** End of the reverse iterator */
  HSHM_INLINE_CROSS_FUN riterator_t rend() {
    return citerator_t(this, (i64)-1);
  }

  /** Beginning of the constant reverse iterator */
  HSHM_INLINE_CROSS_FUN criterator_t crbegin() const {
    return criterator_t(const_cast<vector_base *>(this), size<i64>() - 1);
  }

  /** End of the constant reverse iterator */
  HSHM_INLINE_CROSS_FUN criterator_t crend() const {
    return criterator_t(const_cast<vector_base *>(this), (i64)-1);
  }

  /** Lets Thallium know how to serialize an hipc::vector_base. */
  template <typename Ar>
  HSHM_CROSS_FUN void save(Ar &ar) const {
    save_vec<Ar, hipc::vector_base<T, HSHM_CLASS_TEMPL_ARGS>, T>(ar, *this);
  }

  /** Lets Thallium know how to deserialize an hipc::vector_base. */
  template <typename Ar>
  HSHM_CROSS_FUN void load(Ar &ar) {
    load_vec<Ar, hipc::vector_base<T, HSHM_CLASS_TEMPL_ARGS>, T>(ar, *this);
  }
};

template <typename T, HSHM_CLASS_TEMPL_WITH_IPC_DEFAULTS>
using vector = vector_base<T, HSHM_CLASS_TEMPL_ARGS>;

}  // namespace hshm::ipc

namespace hshm {

template <typename T, HSHM_CLASS_TEMPL_WITH_PRIV_DEFAULTS>
using vector = hipc::vector_base<T, HSHM_CLASS_TEMPL_ARGS>;

}  // namespace hshm

#undef CLASS_NAME
#undef CLASS_NEW_ARGS

#endif  // HSHM_DATA_STRUCTURES_LOCKLESS_VECTOR_H_
