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
#include "hermes_shm/data_structures/ipc/string.h"
#include "hermes_shm/data_structures/serialization/serialize_common.h"
#include <string>

namespace hshm::ipc {

/** forward declaration for string */
template<size_t SSO, HSHM_CLASS_TEMPL_WITH_DEFAULTS>
class string_templ;

/**
 * MACROS used to simplify the string namespace
 * Used as inputs to the HIPC_CONTAINER_TEMPLATE
 * */
#define CLASS_NAME string_templ
#define CLASS_NEW_ARGS SSO

/**
 * A string of characters.
 * */
template<size_t SSO, HSHM_CLASS_TEMPL>
class string_templ : public ShmContainer {
 public:
  HIPC_CONTAINER_TEMPLATE((CLASS_NAME), (CLASS_NEW_ARGS))

 public:
  size_t length_;
  Pointer text_;
  char sso_[SSO];

 public:
  /**====================================
   * Default Constructor
   * ===================================*/

  /** Constructor. Default. */
  HSHM_CROSS_FUN
  explicit string_templ() {
    init_shm_container(HERMES_MEMORY_MANAGER->GetDefaultAllocator<AllocT>());
    SetNull();
  }

  /** SHM Constructor. Default. */
  HSHM_CROSS_FUN
  explicit string_templ(const hipc::CtxAllocator<AllocT> &alloc) {
    init_shm_container(alloc);
    SetNull();
  }

  /**====================================
   * Emplace Constructors
   * ===================================*/

  /** SHM Constructor. Just allocate space. */
  HSHM_CROSS_FUN
  explicit string_templ(size_t length) {
    init_shm_container(HERMES_MEMORY_MANAGER->GetDefaultAllocator<AllocT>());
    _create_str(length);
  }

  /** SHM Constructor. Just allocate space. */
  HSHM_CROSS_FUN
  explicit string_templ(const hipc::CtxAllocator<AllocT> &alloc,
                        size_t length) {
    init_shm_container(alloc);
    _create_str(length);
  }

  /**====================================
   * Copy Constructors
   * ===================================*/

  /** Constructor. From const char* */
  HSHM_CROSS_FUN
  explicit string_templ(const char *text) {
    shm_strong_copy_op<false, false>(
        HERMES_MEMORY_MANAGER->GetDefaultAllocator<AllocT>(), text, 0);
  }

  /** SHM Constructor. From const char* */
  HSHM_CROSS_FUN
  explicit string_templ(const hipc::CtxAllocator<AllocT> &alloc,
                        const char *text) {
    shm_strong_copy_op<false, false>(
          alloc, text, 0);
  }

  /** Constructor. From const char* and size */
  HSHM_CROSS_FUN
  explicit string_templ(const char *text, size_t length) {
    shm_strong_copy_op<false, true>(
      HERMES_MEMORY_MANAGER->GetDefaultAllocator<AllocT>(), text, length);
  }

  /** SHM Constructor. From const char* and size */
  HSHM_CROSS_FUN
  explicit string_templ(const hipc::CtxAllocator<AllocT> &alloc,
                        const char *text, size_t length) {
    shm_strong_copy_op<false, true>(
      alloc, text, length);
  }

  /** Copy constructor. From string_templ. */
  HSHM_CROSS_FUN
  explicit string_templ(const string_templ &other) {
    shm_strong_copy_op<false, true>(
        other.GetCtxAllocator(), other.data(), other.size());
  }

  /** SHM copy constructor. From string_templ. */
  HSHM_CROSS_FUN
  explicit string_templ(const hipc::CtxAllocator<AllocT> &alloc,
                        const string_templ &other) {
    shm_strong_copy_op<false, true>(
            alloc, other.data(), other.size());
  }

  /** SHM copy assignment operator. From string_templ. */
  HSHM_CROSS_FUN
  string_templ& operator=(const string_templ &other) {
    if (this != &other) {
      shm_strong_copy_op<true, true>(
        GetCtxAllocator(), other.data(), other.size());
    }
    return *this;
  }

  /** Constructor. From std::string */
  HSHM_CROSS_FUN
  explicit string_templ(const std::string &other) {
    shm_strong_copy_op<false, true>(
        HERMES_MEMORY_MANAGER->GetDefaultAllocator<AllocT>(), other.data(), other.size());
  }

  /** SHM Constructor. From std::string */
  HSHM_HOST_FUN
  explicit string_templ(const hipc::CtxAllocator<AllocT> &alloc,
                        const std::string &other) {
    shm_strong_copy_op<false, true>(
      alloc, other.data(), other.size());
  }

  /** SHM copy assignment operator. From std::string. */
  HSHM_HOST_FUN
  string_templ& operator=(const std::string &other) {
    shm_strong_copy_op<true, true>(
      GetCtxAllocator(), other.data(), other.size());
    return *this;
  }

  /** Strong copy operation */
  template<bool IS_ASSIGN, bool HAS_LENGTH>
  void shm_strong_copy_op(const hipc::CtxAllocator<AllocT> &alloc,
                          const char *text,
                          size_t length) {
    if constexpr (IS_ASSIGN) {
      shm_destroy();
    } else {
      init_shm_container(alloc);
    }
    if constexpr (!HAS_LENGTH) {
      length = strlen(text);
    }
    _create_str(text, length);
  }

  /** Strong copy */
  HSHM_INLINE_CROSS_FUN
  void strong_copy(const string_templ &other) {
    length_ = other.length_;
    text_ = other.text_;
    if (length_ < SSO) {
      memcpy(sso_, other.sso_, other.length_ + 1);
    }
  }

  /**====================================
   * Move Constructors
   * ===================================*/

  /** Move constructor. */
  HSHM_CROSS_FUN
  string_templ(string_templ &&other) {
    shm_move_op<false>(other.GetCtxAllocator(), std::move(other));
  }

  /** SHM move constructor. */
  HSHM_CROSS_FUN
  string_templ(const hipc::CtxAllocator<AllocT> &alloc, string_templ &&other) {
    shm_move_op<false>(alloc, std::move(other));
  }

  /** SHM move assignment operator. */
  HSHM_CROSS_FUN
  string_templ& operator=(string_templ &&other) noexcept {
    if (this != &other) {
      shm_move_op<true>(GetCtxAllocator(), std::move(other));
    }
    return *this;
  }

  /** SHM move operator. */
  template<bool IS_ASSIGN>
  HSHM_CROSS_FUN
  void shm_move_op(const hipc::CtxAllocator<AllocT> &alloc,
                   string_templ &&other) noexcept {
    if constexpr (IS_ASSIGN) {
      shm_destroy();
    } else {
      init_shm_container(alloc);
    }
    if (GetCtxAllocator() == other.GetCtxAllocator()) {
      strong_copy(other);
      other.SetNull();
    } else {
      _create_str(other.data(), other.size());
      other.shm_destroy();
    }
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
      GetAllocator()->Free(GetMemCtx(), text_);
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
    return string_hash<string_templ>(*this);
  }

  /** Convert into a std::string */
  HSHM_INLINE_HOST_FUN std::string str() const {
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
    } else if (length_ < SSO && new_size < SSO) {
      length_ = new_size;
    } else if (length_ < SSO && new_size >= SSO) {
      LPointer<char> text = GetAllocator()->template AllocateLocalPtr<char>(
          GetMemCtx(), new_size);
      text_ = text.shm_;
      memcpy(text.ptr_, sso_, length_);
    } else if (new_size > size()) {
      GetAllocator()->template Reallocate<Pointer>(
          GetMemCtx(), text_, new_size);
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
  template <typename Ar>
  HSHM_CROSS_FUN
  void load(Ar &ar) {
    load_string<Ar, string_templ>(ar, *this);
  }

  /**====================================
   * Comparison Operations
   * ===================================*/

#define HERMES_STR_CMP_OPERATOR(op) \
  bool operator TYPE_UNWRAP(op)(const char *other) const { \
    return hshm::strncmp(data(), size(), other, hshm::strlen(other)) op 0; \
  } \
  bool operator op(const std::string &other) const { \
    return hshm::strncmp(data(), size(), other.data(), other.size()) op 0; \
  } \
  bool operator op(const string_templ &other) const { \
    return hshm::strncmp(data(), size(), other.data(), other.size()) op 0; \
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
      text_ = GetAllocator()->Allocate(GetMemCtx(), length + 1);
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
using string = string_templ<32>;

/** Consider the string as an uniterpreted set of bytes */
using charbuf = string;

}  // namespace hshm::ipc

namespace hshm {

template<size_t SSO, HSHM_CLASS_TEMPL_WITH_PRIV_DEFAULTS>
using string_templ = ipc::string_templ<SSO, HSHM_CLASS_TEMPL_ARGS>;

}  // namespace hshm

/** std::hash function for string */
namespace std {
template <size_t SSO, HSHM_CLASS_TEMPL>
struct hash<hshm::ipc::string_templ<SSO, HSHM_CLASS_TEMPL_ARGS>> {
  HSHM_CROSS_FUN size_t operator()(
      const hshm::ipc::string_templ<SSO, HSHM_CLASS_TEMPL_ARGS> &text) const {
    return text.Hash();
  }
};
}  // namespace std

/** hshm::hash function for string */
namespace hshm {
template <size_t SSO, HSHM_CLASS_TEMPL>
struct hash<hshm::ipc::string_templ<SSO, HSHM_CLASS_TEMPL_ARGS>> {
  HSHM_CROSS_FUN size_t operator()(
      const hshm::ipc::string_templ<SSO, HSHM_CLASS_TEMPL_ARGS> &text) const {
    return text.Hash();
  }
};
}  // namespace hshm

#undef CLASS_NAME
#undef CLASS_NEW_ARGS

#endif  // HERMES_DATA_STRUCTURES_LOCKLESS_STRING_H_
