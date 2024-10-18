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


#ifndef HERMES_DATA_STRUCTURES_LOCKLESS_STRING_H_
#define HERMES_DATA_STRUCTURES_LOCKLESS_STRING_H_

#include "hermes_shm/data_structures/ipc/internal/shm_internal.h"
#include "hermes_shm/data_structures/containers/charbuf.h"
#include "hermes_shm/data_structures/serialization/serialize_common.h"
#include <string>

namespace hshm::ipc {

/** forward declaration for string */
template<size_t SSO>
class string_templ;

/**
 * MACROS used to simplify the string namespace
 * Used as inputs to the HIPC_CONTAINER_TEMPLATE
 * */
#define CLASS_NAME string_templ
#define TYPED_CLASS string_templ<SSO>

/** string shared-memory header */
template<size_t SSO>
struct ShmHeader<string_templ<SSO>> {
  HIPC_CONTAINER_HEADER_TEMPLATE(ShmHeader)
  size_t length_;
  char sso_[SSO];
  Pointer text_;

  /** Strong copy operation */
  HSHM_CROSS_FUN
  void strong_copy(const ShmHeader &other) {
    length_ = other.length_;
    text_ = other.text_;
    if (length_ < SSO) {
      memcpy(sso_, other.sso_, other.length_ + 1);
    }
  }
};

/**
 * A string of characters.
 * */
template<size_t SSO>
class string_templ : public ShmContainer {
 public:
  HIPC_CONTAINER_TEMPLATE((CLASS_NAME), (TYPED_CLASS))

 public:
  size_t length_;
  Pointer text_;
  char sso_[SSO];

 public:
  /**====================================
   * Default Constructor
   * ===================================*/

  /** SHM Constructor. Default. */
  HSHM_CROSS_FUN
  explicit string_templ(Allocator *alloc) {
    init_shm_container(alloc);
    SetNull();
  }

  /**====================================
   * Emplace Constructors
   * ===================================*/

  /** SHM Constructor. Just allocate space. */
  HSHM_CROSS_FUN
  explicit string_templ(Allocator *alloc,
                        size_t length) {
    init_shm_container(alloc);
    _create_str(length);
  }

  /**====================================
   * Copy Constructors
   * ===================================*/

  /** SHM Constructor. From const char* */
  HSHM_CROSS_FUN
  explicit string_templ(Allocator *alloc,
                        const char *text) {
    init_shm_container(alloc);
    size_t length = strlen(text);
    _create_str(text, length);
  }

  /** SHM Constructor. From const char* and size */
  HSHM_CROSS_FUN
  explicit string_templ(Allocator *alloc,
                        const char *text, size_t length) {
    init_shm_container(alloc);
    _create_str(text, length);
  }

  /** SHM Constructor. From std::string */
  HSHM_CROSS_FUN
  explicit string_templ(Allocator *alloc,
                        const std::string &text) {
    init_shm_container(alloc);
    _create_str(text.data(), text.size());
  }

  /** SHM copy assignment operator. From std::string. */
  HSHM_CROSS_FUN
  string_templ& operator=(const std::string &other) {
    shm_destroy();
    _create_str(other.data(), other.size());
    return *this;
  }

  /** SHM Constructor. From std::string */
  HSHM_CROSS_FUN
  explicit string_templ(Allocator *alloc,
                        const hshm::charbuf &text) {
    init_shm_container(alloc);
    _create_str(text.data(), text.size());
  }

  /** SHM copy assignment operator. From std::string. */
  HSHM_CROSS_FUN
  string_templ& operator=(const hshm::charbuf &other) {
    shm_destroy();
    _create_str(other.data(), other.size());
    return *this;
  }

  /** SHM copy constructor. From string. */
  HSHM_CROSS_FUN
  explicit string_templ(Allocator *alloc,
                        const string_templ &other) {
    init_shm_container(alloc);
    _create_str(other.data(), other.size());
  }

  /** SHM copy assignment operator. From string. */
  HSHM_CROSS_FUN
  string_templ& operator=(const string_templ &other) {
    if (this != &other) {
      shm_destroy();
      _create_str(other.data(), other.size());
    }
    return *this;
  }

  /**====================================
   * Move Constructors
   * ===================================*/

  /** Strong copy operation */
  HSHM_INLINE_CROSS_FUN
  void strong_copy(const string_templ &other) {
    length_ = other.length_;
    text_ = other.text_;
    if (length_ < SSO) {
      memcpy(sso_, other.sso_, other.length_ + 1);
    }
  }

  /** SHM move constructor. */
  HSHM_CROSS_FUN
  string_templ(Allocator *alloc, string_templ &&other) {
    init_shm_container(alloc);
    if (GetAllocator() == other.GetAllocator()) {
      strong_copy(other);
      other.SetNull();
    } else {
      _create_str(other.data(), other.size());
      other.shm_destroy();
    }
  }

  /** SHM move assignment operator. */
  HSHM_CROSS_FUN
  string_templ& operator=(string_templ &&other) noexcept {
    if (this != &other) {
      shm_destroy();
      if (GetAllocator() == other.GetAllocator()) {
        strong_copy(other);
        other.SetNull();
      } else {
        _create_str(other.data(), other.size());
        other.shm_destroy();
      }
    }
    return *this;
  }

  /**====================================
   * Destructors
   * ===================================*/

  /** Check if this string is NULL */
  HSHM_INLINE_CROSS_FUN bool IsNull() const {
    return length_ == 0;
  }

  /** Set this string to NULL */
  HSHM_INLINE_CROSS_FUN void SetNull() {
    text_.SetNull();
    length_ = 0;
  }

  /** Destroy the shared-memory data. */
  HSHM_INLINE_CROSS_FUN void shm_destroy_main() {
    if (size() >= SSO) {
      GetAllocator()->Free(text_);
    }
  }

  /**====================================
   * String Operations
   * ===================================*/

  /** Get character at index i in the string */
  HSHM_INLINE_CROSS_FUN char& operator[](size_t i) {
    return data()[i];
  }

  /** Get character at index i in the string */
  HSHM_INLINE_CROSS_FUN const char& operator[](size_t i) const {
    return data()[i];
  }

  /** Hash function */
  HSHM_CROSS_FUN size_t Hash() const {
    return string_hash<string_templ<SSO>>(*this);
  }

  /** Convert into a std::string */
  HSHM_INLINE_CROSS_FUN std::string str() const {
    return {c_str(), length_};
  }

  /** Get the size of the current string */
  HSHM_INLINE_CROSS_FUN size_t size() const {
    return length_;
  }

  /** Get a constant reference to the C-style string */
  HSHM_INLINE_CROSS_FUN const char* c_str() const {
    return data();
  }

  /** Get a constant reference to the C-style string */
  HSHM_INLINE_CROSS_FUN const char* data() const {
    if (length_ < SSO) {
      return sso_;
    } else {
      return GetAllocator()->template Convert<char, Pointer>(text_);
    }
  }

  /** Get a mutable reference to the C-style string */
  HSHM_INLINE_CROSS_FUN char* data() {
    if (length_ < SSO) {
      return sso_;
    } else {
      return GetAllocator()->template Convert<char, Pointer>(text_);
    }
  }

  /** Resize this string */
  HSHM_CROSS_FUN
  void resize(size_t new_size) {
    if (IsNull()) {
      _create_str(new_size);
    } else if (new_size > size()) {
      GetAllocator()->template Reallocate<Pointer>(text_, new_size);
      length_ = new_size;
    } else {
      length_ = new_size;
    }
  }

  /**====================================
   * Serialization
   * ===================================*/

  /** Serialize */
  template <typename Ar>
  HSHM_CROSS_FUN
  void save(Ar &ar) const {
    save_string<Ar, string_templ>(ar, *this);
  }

  /** Deserialize */
  template <typename A>
  HSHM_CROSS_FUN
  void load(A &ar) {
    load_string<A, string_templ>(ar, *this);
  }

  /**====================================
   * Comparison Operations
   * ===================================*/

  HSHM_CROSS_FUN
  int _strncmp(const char *a, size_t len_a,
               const char *b, size_t len_b) const {
    if (len_a != len_b) {
      return int((int64_t)len_a - (int64_t)len_b);
    }
    for (size_t i = 0; i < len_a; ++i) {
      if (a[i] != b[i]) {
        return a[i] - b[i];
      }
    }
    return 0;
  }

#define HERMES_STR_CMP_OPERATOR(op) \
  HSHM_CROSS_FUN bool operator op(const char *other) const { \
    return _strncmp(data(), size(), other, strlen(other)) op 0; \
  } \
  HSHM_CROSS_FUN bool operator op(const std::string &other) const { \
    return _strncmp(data(), size(), other.data(), other.size()) op 0; \
  } \
  HSHM_CROSS_FUN bool operator op(const string_templ &other) const { \
    return _strncmp(data(), size(), other.data(), other.size()) op 0; \
  }

  HERMES_STR_CMP_OPERATOR(==)  // NOLINT
  HERMES_STR_CMP_OPERATOR(!=)  // NOLINT
  HERMES_STR_CMP_OPERATOR(<)  // NOLINT
  HERMES_STR_CMP_OPERATOR(>)  // NOLINT
  HERMES_STR_CMP_OPERATOR(<=)  // NOLINT
  HERMES_STR_CMP_OPERATOR(>=)  // NOLINT
#undef HERMES_STR_CMP_OPERATOR

 private:
  HSHM_INLINE_CROSS_FUN void _create_str(size_t length) {
    if (length < SSO) {
      // NOTE(llogan): less than and not equal because length doesn't
      // account for trailing 0.
    } else {
      text_ = GetAllocator()->Allocate(length + 1);
    }
    length_ = length;
  }
  HSHM_INLINE_CROSS_FUN void _create_str(const char *text, size_t length) {
    _create_str(length);
    char *str = data();
    memcpy(str, text, length);
    str[length] = 0;
  }
};

/** Our default SSO value */
typedef string_templ<32> string;

/** Consider the string as an uniterpreted set of bytes */
typedef string charbuf;

}  // namespace hshm::ipc

/** std::hash function for string */
namespace std {
template<size_t SSO>
struct hash<hshm::ipc::string_templ<SSO>> {
  HSHM_CROSS_FUN size_t operator()(const hshm::ipc::string_templ<SSO> &text) const {
    return text.Hash();
  }
};
}  // namespace std

/** hshm::hash function for string */
namespace hshm {
template<size_t SSO>
struct hash<hshm::ipc::string_templ<SSO>> {
  HSHM_CROSS_FUN size_t operator()(const hshm::ipc::string_templ<SSO> &text) const {
    return text.Hash();
  }
};
}  // namespace hshm

#undef CLASS_NAME
#undef TYPED_CLASS

#endif  // HERMES_DATA_STRUCTURES_LOCKLESS_STRING_H_
