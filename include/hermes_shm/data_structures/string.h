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

#include "internal/shm_internal.h"
#include <string>

namespace hermes_shm::ipc {

/** forward declaration for string */
class string;

/**
 * MACROS used to simplify the string namespace
 * Used as inputs to the SHM_CONTAINER_TEMPLATE
 * */
#define CLASS_NAME string
#define TYPED_CLASS string
#define TYPED_HEADER ShmHeader<string>

/** string shared-memory header */
template<>
struct ShmHeader<string> : public ShmBaseHeader {
  SHM_CONTAINER_HEADER_TEMPLATE(ShmHeader)
  size_t length_;
  Pointer text_;

  /** Strong copy operation */
  void strong_copy(const ShmHeader &other) {
    length_ = other.length_;
    text_ = other.text_;
  }
};

/**
 * A string of characters.
 * */
class string : public ShmContainer {
 public:
  SHM_CONTAINER_TEMPLATE((CLASS_NAME), (TYPED_CLASS), (TYPED_HEADER))

 public:
  char *text_;

 public:
  /**====================================
   * Shm Overrides
   * ===================================*/

  /** SHM Constructor. Default. */
  void shm_init() {
    SetNull();
  }

  /** SHM Constructor. From const char* */
  void shm_init(const char *text) {
    size_t length = strlen(text);
    _create_str(text, length);
  }

  /** SHM Constructor. From const char* and size */
  void shm_init(const char *text, size_t length) {
    _create_str(text, length);
  }

  /** SHM Constructor. From std::string */
  void shm_init(const std::string &text) {
    _create_str(text.data(), text.size());
  }

  /** Internal move operator. */
  void shm_weak_move_main(string &&other) {
      memcpy((void*)header_, (void*)other.header_, sizeof(*header_));
      shm_deserialize_main();
      other.SetNull();
  }

  /** Internal copy operator */
  void shm_strong_copy_main(const string &other) {
    _create_str(other.data(), other.size());
  }

  /** Destroy the shared-memory data. */
  void shm_destroy_main() {
    alloc_->Free(header_->text_);
  }

  /** Load from shared memory */
  void shm_deserialize_main() {
    if (!IsNull()) {
      text_ = alloc_->template
        Convert<char>(header_->text_);
    } else {
      text_ = nullptr;
    }
  }

  /** Check if this string is NULL */
  bool IsNull() const {
    return header_ == nullptr || header_->text_.IsNull();
  }

  /** Set this string to NULL */
  void SetNull() {
    header_->text_.SetNull();
    header_->length_ = 0;
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
  char* data_mutable() {
    return text_;
  }

  /**====================================
   * Comparison Operations
   * ===================================*/

  int _strncmp(const char *a, size_t len_a,
               const char *b, size_t len_b) const {
    if (len_a != len_b) {
      return int((int64_t)len_a - (int64_t)len_b);
    }
    int sum = 0;
    for (size_t i = 0; i < len_a; ++i) {
      sum += a[i] - b[i];
    }
    return sum;
  }

#define HERMES_STR_CMP_OPERATOR(op) \
  bool operator op(const char *other) const { \
    return _strncmp(data(), size(), other, strlen(other)) op 0; \
  } \
  bool operator op(const std::string &other) const { \
    return _strncmp(data(), size(), other.data(), other.size()) op 0; \
  } \
  bool operator op(const string &other) const { \
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
    text_ = alloc_->AllocatePtr<char>(length + 1, header_->text_);
    header_->length_ = length;
  }
  inline void _create_str(const char *text, size_t length) {
    _create_str(length);
    memcpy(text_, text, length);
    text_[length] = 0;
  }
};

/** Consider the string as an uniterpreted set of bytes */
typedef string charbuf;

}  // namespace hermes_shm::ipc

namespace std {

/** Hash function for string */
template<>
struct hash<hermes_shm::ipc::string> {
  size_t operator()(const hermes_shm::ipc::string &text) const {
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
