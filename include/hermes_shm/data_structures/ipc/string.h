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
#include <string>

namespace hshm::ipc {

/** forward declaration for string */
template<size_t SSO>
class string_templ;

/**
 * MACROS used to simplify the string namespace
 * Used as inputs to the SHM_CONTAINER_TEMPLATE
 * */
#define CLASS_NAME string_templ
#define TYPED_CLASS string_templ<SSO>
#define TYPED_HEADER ShmHeader<string_templ<SSO>>

/** string shared-memory header */
template<size_t SSO>
struct ShmHeader<string_templ<SSO>> {
  SHM_CONTAINER_HEADER_TEMPLATE(ShmHeader)
  size_t length_;
  char sso_[SSO];
  Pointer text_;

  /** Strong copy operation */
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
  SHM_CONTAINER_TEMPLATE((CLASS_NAME), (TYPED_CLASS), (TYPED_HEADER))

 public:
  char *text_;

 public:
  /**====================================
   * Default Constructor
   * ===================================*/

  /** SHM Constructor. Default. */
  explicit string_templ(TYPED_HEADER *header, Allocator *alloc) {
    shm_init_header(header, alloc);
    SetNull();
  }

  /**====================================
   * Emplace Constructors
   * ===================================*/

  /** SHM Constructor. Just allocate space. */
  explicit string_templ(TYPED_HEADER *header, Allocator *alloc,
                  size_t length) {
    shm_init_header(header, alloc);
    _create_str(length);
  }

  /**====================================
   * Copy Constructors
   * ===================================*/

  /** SHM Constructor. From const char* */
  explicit string_templ(TYPED_HEADER *header, Allocator *alloc,
                  const char *text) {
    shm_init_header(header, alloc);
    size_t length = strlen(text);
    _create_str(text, length);
  }

  /** SHM Constructor. From const char* and size */
  explicit string_templ(TYPED_HEADER *header, Allocator *alloc,
                  const char *text, size_t length) {
    shm_init_header(header, alloc);
    _create_str(text, length);
  }

  /** SHM Constructor. From std::string */
  explicit string_templ(TYPED_HEADER *header, Allocator *alloc,
                  const std::string &text) {
    shm_init_header(header, alloc);
    _create_str(text.data(), text.size());
  }

  /** SHM copy assignment operator. From std::string. */
  string_templ& operator=(const std::string &other) {
    shm_destroy();
    _create_str(other.data(), other.size());
    return *this;
  }

  /** SHM Constructor. From std::string */
  explicit string_templ(TYPED_HEADER *header, Allocator *alloc,
                  const hshm::charbuf &text) {
    shm_init_header(header, alloc);
    _create_str(text.data(), text.size());
  }

  /** SHM copy assignment operator. From std::string. */
  string_templ& operator=(const hshm::charbuf &other) {
    shm_destroy();
    _create_str(other.data(), other.size());
    return *this;
  }

  /** SHM copy constructor. From string. */
  explicit string_templ(TYPED_HEADER *header, Allocator *alloc,
                  const string_templ &other) {
    shm_init_header(header, alloc);
    _create_str(other.data(), other.size());
  }

  /** SHM copy assignment operator. From string. */
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

  /** SHM move constructor. */
  string_templ(TYPED_HEADER *header, Allocator *alloc, string_templ &&other) {
    shm_init_header(header, alloc);
    if (alloc_ == other.alloc_) {
      (*header_) = (*other.header_);
      shm_deserialize_main();
      other.SetNull();
    } else {
      _create_str(other.data(), other.size());
      other.shm_destroy();
    }
  }

  /** SHM move assignment operator. */
  string_templ& operator=(string_templ &&other) noexcept {
    if (this != &other) {
      shm_destroy();
      if (alloc_ == other.alloc_) {
        (*header_) = (*other.header_);
        shm_deserialize_main();
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
  bool IsNull() const {
    return header_->length_ == 0;
  }

  /** Set this string to NULL */
  void SetNull() {
    header_->text_.SetNull();
    header_->length_ = 0;
  }

  /** Destroy the shared-memory data. */
  void shm_destroy_main() {
    if (size() >= SSO) {
      alloc_->Free(header_->text_);
    }
  }

  /**====================================
   * SHM Deserialization
   * ===================================*/

  /** Load from shared memory */
  void shm_deserialize_main() {
    if (!IsNull()) {
      if (size() < SSO) {
        text_ = header_->sso_;
      } else {
        text_ = alloc_->template
          Convert<char>(header_->text_);
      }
    }
  }

  /**====================================
   * String Operations
   * ===================================*/

  /** Get character at index i in the string */
  char& operator[](size_t i) const {
    return text_[i];
  }

  /** Convert into a std::string */
  std::string str() const {
    return {text_, header_->length_};
  }

  /** Get the size of the current string */
  size_t size() const {
    return header_->length_;
  }

  /** Get a constant reference to the C-style string */
  char* c_str() const {
    return text_;
  }

  /** Get a constant reference to the C-style string */
  char* data() const {
    return text_;
  }

  /** Get a mutable reference to the C-style string */
  char* data() {
    return text_;
  }

  /** Resize this string */
  void resize(size_t new_size) {
    if (IsNull()) {
      _create_str(new_size);
    } else if (new_size > size()) {
      text_ = alloc_->ReallocatePtr<char, Pointer>(header_->text_, new_size);
      header_->length_ = new_size;
    } else {
      header_->length_ = new_size;
    }
  }

  /**====================================
   * Comparison Operations
   * ===================================*/

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
  bool operator op(const char *other) const { \
    return _strncmp(data(), size(), other, strlen(other)) op 0; \
  } \
  bool operator op(const std::string &other) const { \
    return _strncmp(data(), size(), other.data(), other.size()) op 0; \
  } \
  bool operator op(const string_templ &other) const { \
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
  inline void _create_str(size_t length) {
    if (length < SSO) {
      // NOTE(llogan): less than and not equal because length doesn't
      // account for trailing 0.
      text_ = header_->sso_;
    } else {
      text_ = alloc_->AllocatePtr<char>(length + 1, header_->text_);
    }
    header_->length_ = length;
  }
  inline void _create_str(const char *text, size_t length) {
    _create_str(length);
    memcpy(text_, text, length);
    text_[length] = 0;
  }
};

/** Our default SSO value */
typedef string_templ<32> string;

/** Consider the string as an uniterpreted set of bytes */
typedef string charbuf;

}  // namespace hshm::ipc

namespace std {

/** Hash function for string */
template<size_t SSO>
struct hash<hshm::ipc::string_templ<SSO>> {
  size_t operator()(const hshm::ipc::string_templ<SSO> &text) const {
    size_t sum = 0;
    for (size_t i = 0; i < text.size(); ++i) {
      auto shift = static_cast<size_t>(i % sizeof(size_t));
      auto c = static_cast<size_t>((unsigned char)text[i]);
      sum = 31*sum + (c << shift);
    }
    return sum;
  }
};

}  // namespace std

#undef CLASS_NAME
#undef TYPED_CLASS
#undef TYPED_HEADER

#endif  // HERMES_DATA_STRUCTURES_LOCKLESS_STRING_H_
