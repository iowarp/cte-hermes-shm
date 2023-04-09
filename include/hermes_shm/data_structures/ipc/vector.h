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


#ifndef HERMES_DATA_STRUCTURES_LOCKLESS_VECTOR_H_
#define HERMES_DATA_STRUCTURES_LOCKLESS_VECTOR_H_

#include "hermes_shm/data_structures/ipc/internal/shm_internal.h"

#include <vector>

namespace hshm::ipc {

/** forward pointer for vector_templ */
template<typename T, bool FIXED>
class vector_templ;

/**
 * The vector_templ iterator implementation
 * */
template<typename T, bool FIXED, bool FORWARD_ITER>
struct vector_iterator_templ {
 public:
  hipc::Ref<vector_templ<T, FIXED>> vec_;
  off64_t i_;

  /** Default constructor */
  HSHM_ALWAYS_INLINE vector_iterator_templ() = default;

  /** Construct an iterator (called from vector class) */
  template<typename SizeT>
  HSHM_ALWAYS_INLINE explicit vector_iterator_templ(
    const ShmDeserialize<vector_templ<T, FIXED>> &vec, SizeT i)
  : vec_(vec), i_(static_cast<off64_t>(i)) {}

  /** Construct an iterator (called from iterator) */
  HSHM_ALWAYS_INLINE explicit vector_iterator_templ(
    const hipc::Ref<vector_templ<T, FIXED>> &vec, off64_t i)
  : vec_(vec), i_(i) {}

  /** Copy constructor */
  HSHM_ALWAYS_INLINE vector_iterator_templ(const vector_iterator_templ &other)
  : vec_(other.vec_), i_(other.i_) {}

  /** Copy assignment operator  */
  HSHM_ALWAYS_INLINE vector_iterator_templ&
  operator=(const vector_iterator_templ &other) {
    if (this != &other) {
      vec_ = other.vec_;
      i_ = other.i_;
    }
    return *this;
  }

  /** Move constructor */
  HSHM_ALWAYS_INLINE vector_iterator_templ(vector_iterator_templ &&other) {
    vec_ = other.vec_;
    i_ = other.i_;
  }

  /** Move assignment operator  */
  HSHM_ALWAYS_INLINE vector_iterator_templ&
  operator=(vector_iterator_templ &&other) {
    if (this != &other) {
      vec_ = other.vec_;
      i_ = other.i_;
    }
    return *this;
  }

  /** Dereference the iterator */
  HSHM_ALWAYS_INLINE Ref<T> operator*() {
    return Ref<T>(vec_->data_ar()[i_], vec_->GetAllocator());
  }

  /** Dereference the iterator */
  HSHM_ALWAYS_INLINE const Ref<T> operator*() const {
    return Ref<T>(vec_->data_ar()[i_], vec_->GetAllocator());
  }

  /** Increment iterator in-place */
  HSHM_ALWAYS_INLINE vector_iterator_templ& operator++() {
    if constexpr(FORWARD_ITER) {
      ++i_;
    } else {
      --i_;
    }
    return *this;
  }

  /** Decrement iterator in-place */
  HSHM_ALWAYS_INLINE vector_iterator_templ& operator--() {
    if (is_begin() || is_end()) { return *this; }
    if constexpr(FORWARD_ITER) {
      --i_;
    } else {
      ++i_;
    }
    return *this;
  }

  /** Create the next iterator */
  HSHM_ALWAYS_INLINE vector_iterator_templ operator++(int) const {
    vector_iterator_templ next_iter(*this);
    ++next_iter;
    return next_iter;
  }

  /** Create the prior iterator */
  HSHM_ALWAYS_INLINE vector_iterator_templ operator--(int) const {
    vector_iterator_templ prior_iter(*this);
    --prior_iter;
    return prior_iter;
  }

  /** Increment iterator by \a count and return */
  HSHM_ALWAYS_INLINE vector_iterator_templ operator+(size_t count) const {
    if constexpr(FORWARD_ITER) {
      return vector_iterator_templ(vec_, i_ + count);
    } else {
      return vector_iterator_templ(vec_, i_ - count);
    }
  }

  /** Decrement iterator by \a count and return */
  HSHM_ALWAYS_INLINE vector_iterator_templ operator-(size_t count) const {
    if constexpr(FORWARD_ITER) {
      return vector_iterator_templ(vec_, i_ - count);
    } else {
      return vector_iterator_templ(vec_, i_ + count);
    }
  }

  /** Increment iterator by \a count in-place */
  HSHM_ALWAYS_INLINE void operator+=(size_t count) {
    if constexpr(FORWARD_ITER) {
      i_ += count;
    } else {
      i_ -= count;
    }
  }

  /** Decrement iterator by \a count in-place */
  HSHM_ALWAYS_INLINE void operator-=(size_t count) {
    if constexpr(FORWARD_ITER) {
      i_ -= count;
    } else {
      i_ += count;
    }
  }

  /** Check if two iterators are equal */
  HSHM_ALWAYS_INLINE friend bool operator==(const vector_iterator_templ &a,
                         const vector_iterator_templ &b) {
    return (a.i_ == b.i_);
  }

  /** Check if two iterators are inequal */
  HSHM_ALWAYS_INLINE friend bool operator!=(const vector_iterator_templ &a,
                         const vector_iterator_templ &b) {
    return (a.i_ != b.i_);
  }

  /** Set this iterator to end */
  HSHM_ALWAYS_INLINE void set_end() {
    if constexpr(FORWARD_ITER) {
      i_ = vec_->size();
    } else {
      i_ = -1;
    }
  }

  /** Set this iterator to begin */
  HSHM_ALWAYS_INLINE void set_begin() {
    if constexpr(FORWARD_ITER) {
      i_ = 0;
    } else {
      i_ = vec_->size() - 1;
    }
  }

  /** Determine whether this iterator is the begin iterator */
  HSHM_ALWAYS_INLINE bool is_begin() const {
    if constexpr(FORWARD_ITER) {
      return (i_ == 0);
    } else {
      return (i_ == (int64_t)vec_->size() - 1);
    }
  }

  /** Determine whether this iterator is the end iterator */
  HSHM_ALWAYS_INLINE bool is_end() const {
    if constexpr(FORWARD_ITER) {
      return i_ >= (int64_t)vec_->size();
    } else {
      return i_ == -1;
    }
  }
};

/**
 * MACROS used to simplify the vector_templ namespace
 * Used as inputs to the SHM_CONTAINER_TEMPLATE
 * */
#define CLASS_NAME vector_templ
#define TYPED_CLASS vector_templ<T, FIXED>
#define TYPED_HEADER ShmHeader<vector_templ<T, FIXED>>

/**
 * The vector_templ shared-memory header
 * */
template<typename T, bool FIXED>
struct ShmHeader<TYPED_CLASS> {
  SHM_CONTAINER_HEADER_TEMPLATE(ShmHeader)
  AtomicPointer vec_ptr_;
  size_t max_length_, length_;

  /** Strong copy operation */
  void strong_copy(const ShmHeader &other) {
    vec_ptr_ = other.vec_ptr_;
    max_length_ = other.max_length_;
    length_ = other.length_;
  }
};

/**
 * The vector_templ class
 * */
template<typename T, bool FIXED>
class vector_templ : public ShmContainer {
 public:
  SHM_CONTAINER_TEMPLATE((CLASS_NAME), (TYPED_CLASS), (TYPED_HEADER))

 public:
  /**====================================
   * Typedefs
   * ===================================*/

  /** forwrard iterator */
  typedef vector_iterator_templ<T, FIXED, true>  iterator_t;
  /** reverse iterator */
  typedef vector_iterator_templ<T, FIXED, false> riterator_t;
  /** const iterator */
  typedef vector_iterator_templ<T, FIXED, true>  citerator_t;
  /** const reverse iterator */
  typedef vector_iterator_templ<T, FIXED, false> criterator_t;

 public:
  /**====================================
   * Variables
   * ===================================*/
  ShmArchive<T> *cache_;

 public:
  /**====================================
   * Default Constructor
   * ===================================*/

  /** SHM constructor. Default. */
  explicit vector_templ(TYPED_HEADER *header, Allocator *alloc) {
    shm_init_header(header, alloc);
    SetNull();
  }

  /** SHM constructor. Resize + construct. */
  template<typename ...Args>
  explicit vector_templ(TYPED_HEADER *header, Allocator *alloc,
                        size_t length, Args&& ...args) {
    shm_init_header(header, alloc);
    SetNull();
    resize(length, std::forward<Args>(args)...);
  }

  /**====================================
   * Copy Constructors
   * ===================================*/

  /** SHM copy constructor. From vector_templ. */
  explicit vector_templ(TYPED_HEADER *header, Allocator *alloc,
                        const vector_templ &other) {
    shm_init_header(header, alloc);
    SetNull();
    shm_strong_copy_main<vector_templ<T, FIXED>>(other);
  }

  /** SHM copy assignment operator. From vector_templ. */
  vector_templ& operator=(const vector_templ &other) {
    if (this != &other) {
      shm_destroy();
      shm_strong_copy_main<vector_templ>(other);
    }
    return *this;
  }

  /** SHM copy constructor. From std::vector */
  explicit vector_templ(TYPED_HEADER *header, Allocator *alloc,
                        const std::vector<T> &other) {
    shm_init_header(header, alloc);
    SetNull();
    shm_strong_copy_main<std::vector<T>>(other);
  }

  /** SHM copy assignment operator. From std::vector */
  vector_templ& operator=(const std::vector<T> &other) {
    shm_destroy();
    shm_strong_copy_main<std::vector<T>>(other);
    return *this;
  }

  /** The main copy operation  */
  template<typename VectorT>
  void shm_strong_copy_main(const VectorT &other) {
    reserve(other.size());
    if constexpr(std::is_pod<T>() && !IS_SHM_ARCHIVEABLE(T)) {
      memcpy(data(), other.data(),
             other.size() * sizeof(T));
      header_->length_ = other.size();
      shm_deserialize_main();
    } else {
      for (auto iter = other.cbegin(); iter != other.cend(); ++iter) {
        if constexpr(IS_SHM_ARCHIVEABLE(VectorT)) {
          emplace_back((**iter));
        } else {
          emplace_back((*iter));
        }
      }
    }
  }

  /**====================================
   * Move Constructors
   * ===================================*/

  /** SHM move constructor. */
  vector_templ(TYPED_HEADER *header, Allocator *alloc, vector_templ &&other) {
    shm_init_header(header, alloc);
    if (alloc_ == other.alloc_) {
      memcpy((void *) header_, (void *) other.header_, sizeof(*header_));
      shm_deserialize_main();
      other.SetNull();
    } else {
      shm_strong_copy_main<vector_templ>(other);
      other.shm_destroy();
    }
  }

  /** SHM move assignment operator. */
  vector_templ& operator=(vector_templ &&other) noexcept {
    if (this != &other) {
      shm_destroy();
      if (alloc_ == other.alloc_) {
        memcpy((void *) header_, (void *) other.header_, sizeof(*header_));
        shm_deserialize_main();
        other.SetNull();
      } else {
        shm_strong_copy_main<vector_templ>(other);
        other.shm_destroy();
      }
    }
    return *this;
  }

  /**====================================
   * Destructor
   * ===================================*/

  /** Check if null */
  HSHM_ALWAYS_INLINE bool IsNull() const {
    return header_->vec_ptr_.IsNull();
  }

  /** Make null */
  HSHM_ALWAYS_INLINE void SetNull() {
    header_->length_ = 0;
    header_->max_length_ = 0;
    header_->vec_ptr_.SetNull();
  }

  /** Destroy all shared memory allocated by the vector_templ */
  void shm_destroy_main() {
    erase(begin(), end());
    alloc_->Free(header_->vec_ptr_);
  }

  /**====================================
   * SHM Deserialization
   * ===================================*/

  /** Load from shared memory */
  void shm_deserialize_main() {
    if constexpr(FIXED) {
      cache_ = alloc_->template
        Convert<ShmArchive<T>>(header_->vec_ptr_);
    }
  }

  /**====================================
   * Vector Operations
   * ===================================*/

  /**
   * Convert to std::vector
   * */
  std::vector<T> vec() {
    std::vector<T> v;
    v.reserve(size());
    for (hipc::Ref<T> entry : *this) {
      v.emplace_back(*entry);
    }
    return v;
  }

  /**
   * Reserve space in the vector_templ to emplace elements. Does not
   * change the size of the list.
   *
   * @param length the maximum size the vector_templ can get before a growth occurs
   * @param args the arguments to construct
   * */
  template<typename ...Args>
  void reserve(size_t length, Args&& ...args) {
    if (length == 0) { return; }
    grow_vector(data_ar(), length, false, std::forward<Args>(args)...);
  }

  /**
   * Reserve space in the vector_templ to emplace elements. Changes the
   * size of the list.
   *
   * @param length the maximum size the vector_templ can get before a growth occurs
   * @param args the arguments used to construct the vector_templ elements
   * */
  template<typename ...Args>
  void resize(size_t length, Args&& ...args) {
    if (length == 0) {
      header_->length_ = 0;
      return;
    }
    grow_vector(data_ar(), length, true, std::forward<Args>(args)...);
    header_->length_ = length;
  }

  /** Index the vector_templ at position i */
  HSHM_ALWAYS_INLINE hipc::Ref<T> operator[](const size_t i) {
    ShmArchive<T> *vec = data_ar();
    return hipc::Ref<T>(vec[i], alloc_);
  }

  /** Index the vector_templ at position i */
  HSHM_ALWAYS_INLINE const hipc::Ref<T> operator[](const size_t i) const {
    ShmArchive<T> *vec = data_ar();
    return hipc::Ref<T>(vec[i], alloc_);
  }

  /** Get first element of vector_templ */
  HSHM_ALWAYS_INLINE hipc::Ref<T> front() {
    return (*this)[0];
  }

  /** Get last element of vector_templ */
  HSHM_ALWAYS_INLINE hipc::Ref<T> back() {
    return (*this)[size() - 1];
  }

  /** Construct an element at the back of the vector_templ */
  template<typename... Args>
  void emplace_back(Args&& ...args) {
    ShmArchive<T> *vec = data_ar();
    if (header_->length_ == header_->max_length_) {
      vec = grow_vector(vec, 0, false);
    }
    make_ref<T>(vec[header_->length_], alloc_, std::forward<Args>(args)...);
    ++header_->length_;
  }

  /** Construct an element in the front of the vector_templ */
  template<typename ...Args>
  HSHM_ALWAYS_INLINE void emplace_front(Args&& ...args) {
    emplace(begin(), std::forward<Args>(args)...);
  }

  /** Construct an element at an arbitrary position in the vector_templ */
  template<typename ...Args>
  void emplace(iterator_t pos, Args&&... args) {
    if (pos.is_end()) {
      emplace_back(std::forward<Args>(args)...);
      return;
    }
    ShmArchive<T> *vec = data_ar();
    if (header_->length_ == header_->max_length_) {
      vec = grow_vector(vec, 0, false);
    }
    shift_right(pos);
    make_ref<T>(vec[pos.i_], alloc_, std::forward<Args>(args)...);
    ++header_->length_;
  }

  /** Replace an element at a position */
  template<typename ...Args>
  void replace(iterator_t pos, Args&&... args) {
    if (pos.is_end()) {
      return;
    }
    ShmArchive<T> *vec = data_ar();
    (*this)[pos.i_].shm_destroy();
    make_ref<T>(vec[pos.i_], alloc_, std::forward<Args>(args)...);
  }

  /** Delete the element at \a pos position */
  void erase(iterator_t pos) {
    if (pos.is_end()) return;
    shift_left(pos, 1);
    header_->length_ -= 1;
  }

  /** Delete elements between first and last  */
  void erase(iterator_t first, iterator_t last) {
    size_t last_i;
    if (first.is_end()) return;
    if (last.is_end()) {
      last_i = size();
    } else {
      last_i = last.i_;
    }
    size_t count = last_i - first.i_;
    if (count == 0) return;
    shift_left(first, count);
    header_->length_ -= count;
  }

  /** Delete all elements from the vector_templ */
  HSHM_ALWAYS_INLINE void clear() {
    erase(begin(), end());
  }

  /** Get the size of the vector_templ */
  template<typename SizeT=size_t>
  HSHM_ALWAYS_INLINE SizeT size() const {
    return static_cast<SizeT>(header_->length_);
  }

  /** Get the data in the vector_templ */
  HSHM_ALWAYS_INLINE void* data() {
    return reinterpret_cast<void*>(data_ar());
  }

  /** Get constant pointer to the data */
  HSHM_ALWAYS_INLINE void* data() const {
    return reinterpret_cast<void*>(data_ar());
  }

  /** Retreives a pointer to the internal array */
  HSHM_ALWAYS_INLINE ShmArchive<T>* data_ar() {
    if constexpr(FIXED) {
      return cache_;
    } else {
      return alloc_->template
        Convert<ShmArchive<T>>(header_->vec_ptr_);
    }
  }

  /** Retreives a pointer to the array */
  HSHM_ALWAYS_INLINE ShmArchive<T>* data_ar() const {
    if constexpr(FIXED) {
      return cache_;
    } else {
      return alloc_->template
        Convert<ShmArchive<T>>(header_->vec_ptr_);
    }
  }

  /**====================================
   * Internal Operations
   * ===================================*/
 private:
  /**
   * Grow a vector_templ to a new size.
   *
   * @param vec the C-style array of elements to grow
   * @param max_length the new length of the vector_templ. If 0, the current size
   * of the vector_templ will be multiplied by a constant.
   * @param args the arguments used to construct the elements of the vector_templ
   * */
  template<typename ...Args>
  ShmArchive<T>* grow_vector(ShmArchive<T> *vec, size_t max_length,
                             bool resize, Args&& ...args) {
    // Grow vector_templ by 25%
    if (max_length == 0) {
      max_length = 5 * header_->max_length_ / 4;
      if (max_length <= header_->max_length_ + 10) {
        max_length += 10;
      }
    }
    if (max_length < header_->max_length_) {
      return nullptr;
    }

    // Allocate new shared-memory vec
    ShmArchive<T> *new_vec;
    if constexpr(std::is_pod<T>() && !IS_SHM_ARCHIVEABLE(T)) {
      // Use reallocate for well-behaved objects
      new_vec = alloc_->template
        ReallocateObjs<ShmArchive<T>>(header_->vec_ptr_, max_length);
    } else {
      // Use std::move for unpredictable objects
      Pointer new_p;
      new_vec = alloc_->template
        AllocateObjs<ShmArchive<T>>(max_length, new_p);
      for (size_t i = 0; i < header_->length_; ++i) {
        hipc::Ref<T> old_entry = (*this)[i];
        hipc::Ref<T> new_entry = make_ref<T>(new_vec[i], alloc_,
                                             std::move(*old_entry));
      }
      if (!header_->vec_ptr_.IsNull()) {
        alloc_->Free(header_->vec_ptr_);
      }
      header_->vec_ptr_ = new_p;
    }
    if (new_vec == nullptr) {
      throw OUT_OF_MEMORY.format("vector_templ::emplace_back",
                                 max_length*sizeof(ShmArchive<T>));
    }
    if (resize) {
      for (size_t i = header_->length_; i < max_length; ++i) {
        hipc::make_ref<T>(new_vec[i], alloc_, std::forward<Args>(args)...);
      }
    }

    // Update vector_templ header
    header_->max_length_ = max_length;
    if constexpr(FIXED) {
      shm_deserialize_main();
    }

    return new_vec;
  }

  /**
   * Shift every element starting at "pos" to the left by count. Any element
   * who would be shifted before "pos" will be deleted.
   *
   * @param pos the starting position
   * @param count the amount to shift left by
   * */
  void shift_left(const iterator_t pos, size_t count = 1) {
    ShmArchive<T> *vec = data_ar();
    for (size_t i = 0; i < count; ++i) {
      hipc::Ref<T>(vec[pos.i_ + i], alloc_).shm_destroy();
    }
    auto dst = vec + pos.i_;
    auto src = dst + count;
    for (auto i = pos.i_ + count; i < size(); ++i) {
      memcpy((void*)dst, (void*)src, sizeof(ShmArchive<T>));
      dst += 1; src += 1;
    }
  }

  /**
   * Shift every element starting at "pos" to the right by count. Increases
   * the total number of elements of the vector_templ by "count". Does not modify
   * the size parameter of the vector_templ, this is done elsewhere.
   *
   * @param pos the starting position
   * @param count the amount to shift right by
   * */
  void shift_right(const iterator_t pos, size_t count = 1) {
    auto src = data_ar() + size() - 1;
    auto dst = src + count;
    auto sz = static_cast<off64_t>(size());
    for (auto i = sz - 1; i >= pos.i_; --i) {
      memcpy((void*)dst, (void*)src, sizeof(ShmArchive<T>));
      dst -= 1; src -= 1;
    }
  }

  /**====================================
   * Iterators
   * ===================================*/
 public:
  /** Beginning of the forward iterator */
  iterator_t begin() {
    return iterator_t(GetShmDeserialize(), 0);
  }

  /** End of the forward iterator */
  iterator_t end() {
    return iterator_t(GetShmDeserialize(), size());
  }

  /** Beginning of the constant forward iterator */
  citerator_t cbegin() const {
    return citerator_t(GetShmDeserialize(), 0);
  }

  /** End of the forward iterator */
  citerator_t cend() const {
    return citerator_t(GetShmDeserialize(), size());
  }

  /** Beginning of the reverse iterator */
  riterator_t rbegin() {
    return riterator_t(GetShmDeserialize(), size<off64_t>() - 1);
  }

  /** End of the reverse iterator */
  riterator_t rend() {
    return citerator_t(GetShmDeserialize(), (off64_t)-1);
  }

  /** Beginning of the constant reverse iterator */
  criterator_t crbegin() const {
    return criterator_t(GetShmDeserialize(), size<off64_t>() - 1);
  }

  /** End of the constant reverse iterator */
  criterator_t crend() const {
    return criterator_t(GetShmDeserialize(), (off64_t)-1);
  }
};

/** Global definition of variable-sized vector */
template<typename T>
using vector = vector_templ<T, false>;

/** Global definition of a fixed-size array */
template<typename T>
using array = vector_templ<T, true>;

}  // namespace hshm::ipc

#undef CLASS_NAME
#undef TYPED_CLASS
#undef TYPED_HEADER

#endif  // HERMES_DATA_STRUCTURES_LOCKLESS_VECTOR_H_
