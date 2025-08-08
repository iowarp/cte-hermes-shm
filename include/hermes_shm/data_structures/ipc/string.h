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

#ifndef HSHM_DATA_STRUCTURES_LOCKLESS_STRING_H_
#define HSHM_DATA_STRUCTURES_LOCKLESS_STRING_H_

#include <string>

#include "chararr.h"
#include "hermes_shm/data_structures/internal/shm_internal.h"
#include "hermes_shm/data_structures/serialization/serialize_common.h"

namespace hshm::ipc {

class StringFlags {
 public:
  CLS_CONST u32 kWrap = BIT_OPT(u32, 0);
};
#define HSHM_STRING_SSO 31

/** forward declaration for string */
template <size_t SSO, u32 FLAGS, HSHM_CLASS_TEMPL_WITH_DEFAULTS>
class string_templ;

/**
 * MACROS used to simplify the string namespace
 * Used as inputs to the HIPC_CONTAINER_TEMPLATE
 * */
#define CLASS_NAME string_templ
#define CLASS_NEW_ARGS SSO, FLAGS

/**
 * A string of bytes.
 * */
template <size_t SSO, u32 FLAGS, HSHM_CLASS_TEMPL>
class string_templ : public ShmContainer {
 public:
  HIPC_CONTAINER_TEMPLATE((CLASS_NAME), (CLASS_NEW_ARGS))

 public:
  size_t length_, max_length_;
  union {
    Pointer text_;
    char *data_;
  };
  char sso_[SSO + 1];
  bool is_wrap_;

 public:
  /**====================================
   * Default Constructor
   * ===================================*/

  /** Constructor. Default. */
  HSHM_CROSS_FUN
  explicit string_templ() {
    init_shm_container(HSHM_MEMORY_MANAGER->GetDefaultAllocator<AllocT>());
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
    init_shm_container(HSHM_MEMORY_MANAGER->GetDefaultAllocator<AllocT>());
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

  /**
   * const char* constructors
   */

  /** Constructor. From const char* */
  HSHM_CROSS_FUN
  string_templ(const char *text) {
    shm_strong_or_weak_copy_op<false, false>(
        HSHM_MEMORY_MANAGER->GetDefaultAllocator<AllocT>(), text, 0);
  }

  /** SHM Constructor. From const char* */
  HSHM_CROSS_FUN
  explicit string_templ(const hipc::CtxAllocator<AllocT> &alloc,
                        const char *text) {
    shm_strong_or_weak_copy_op<false, false>(alloc, text, 0);
  }

  /** Constructor. From const char* and size */
  HSHM_CROSS_FUN
  explicit string_templ(const char *text, size_t length) {
    shm_strong_or_weak_copy_op<false, true>(
        HSHM_MEMORY_MANAGER->GetDefaultAllocator<AllocT>(), text, length);
  }

  /** SHM Constructor. From const char* and size */
  HSHM_CROSS_FUN
  explicit string_templ(const hipc::CtxAllocator<AllocT> &alloc,
                        const char *text, size_t length) {
    shm_strong_or_weak_copy_op<false, true>(alloc, text, length);
  }

  /** SHM copy assignment operator. From const char*. */
  HSHM_CROSS_FUN string_templ &operator=(const char *other) {
    if ((char *)this != (char *)&other) {
      shm_strong_or_weak_copy_op<false, false>(GetCtxAllocator(), other, 0);
    }
    return *this;
  }

  /**
   * std::string copy constructors
   */

  /** Copy constructor. From std::string. */
  HSHM_HOST_FUN string_templ(const std::string &other) {
    shm_strong_or_weak_copy_op<false, true>(
        HSHM_MEMORY_MANAGER->GetDefaultAllocator<AllocT>(), other.data(),
        other.size());
  }

  /** SHM copy constructor. From std::string. */
  HSHM_HOST_FUN explicit string_templ(const hipc::CtxAllocator<AllocT> &alloc,
                                      const std::string &other) {
    shm_strong_or_weak_copy_op<false, true>(alloc, other.data(), other.size());
  }

  /** SHM copy assignment operator. From std::string. */
  HSHM_HOST_FUN string_templ &operator=(const std::string &other) {
    if (this != reinterpret_cast<const string_templ *>(&other)) {
      shm_strong_or_weak_copy_op<true, true>(GetCtxAllocator(), other.data(),
                                             other.size());
    }
    return *this;
  }

  /**
   * Templated string_templ copy constructors
   */

  /** Copy constructor. From any string_templ. */
  template <size_t SSO1, u32 FLAGS1, HSHM_CLASS_TEMPL2>
  HSHM_CROSS_FUN explicit string_templ(
      const string_templ<SSO1, FLAGS1, HSHM_CLASS_TEMPL_ARGS2> &other) {
    shm_strong_or_weak_copy_op<false, true>(other.GetCtxAllocator(),
                                            other.data(), other.size());
  }

  /** SHM copy constructor. From any string_templ. */
  template <size_t SSO1, u32 FLAGS1, HSHM_CLASS_TEMPL2>
  HSHM_CROSS_FUN explicit string_templ(
      const hipc::CtxAllocator<AllocT> &alloc,
      const string_templ<SSO1, FLAGS1, HSHM_CLASS_TEMPL_ARGS2> &other) {
    shm_strong_or_weak_copy_op<false, true>(alloc, other.data(), other.size());
  }

  /** SHM copy assignment operator. From any string_templ. */
  template <size_t SSO1, u32 FLAGS1, HSHM_CLASS_TEMPL2>
  HSHM_CROSS_FUN string_templ &operator=(
      const string_templ<SSO1, FLAGS1, HSHM_CLASS_TEMPL_ARGS2> &other) {
    if ((char *)this != (char *)&other) {
      shm_strong_or_weak_copy_op<true, true>(GetCtxAllocator(), other.data(),
                                             other.size());
    }
    return *this;
  }

  /**
   * This string_templ copy constructors
   */

  /** Copy constructor. From this string_templ. */
  HSHM_CROSS_FUN explicit string_templ(const string_templ &other) {
    shm_strong_or_weak_copy_op<false, true>(other.GetCtxAllocator(),
                                            other.data(), other.size());
  }

  /** SHM copy constructor. From this string_templ. */
  HSHM_INLINE_CROSS_FUN explicit string_templ(
      const hipc::CtxAllocator<AllocT> &alloc, const string_templ &other) {
    shm_strong_or_weak_copy_op<false, true>(alloc, other.data(), other.size());
  }

  /** SHM copy assignment operator. From this string_templ. */
  HSHM_CROSS_FUN string_templ &operator=(const string_templ &other) {
    if ((char *)this != (char *)&other) {
      shm_strong_or_weak_copy_op<true, true>(GetCtxAllocator(), other.data(),
                                             other.size());
    }
    return *this;
  }

  /**
   * Core copy operations
   */

  /** Strong or weak copy operation */
  template <bool IS_ASSIGN, bool HAS_LENGTH>
  HSHM_CROSS_FUN void shm_strong_or_weak_copy_op(
      const hipc::CtxAllocator<AllocT> &alloc, const char *text,
      size_t length) {
    if constexpr (FLAGS & StringFlags::kWrap) {
      shm_weak_wrap_op<IS_ASSIGN, HAS_LENGTH>(alloc, text, length);
    } else {
      shm_strong_copy_op<IS_ASSIGN, HAS_LENGTH>(alloc, text, length);
    }
  }

  /** Weak wrap operation */
  template <bool IS_ASSIGN, bool HAS_LENGTH>
  HSHM_CROSS_FUN void shm_weak_wrap_op(const hipc::CtxAllocator<AllocT> &alloc,
                                       const char *text, size_t length) {
    if constexpr (IS_ASSIGN) {
      shm_destroy();
    } else {
      init_shm_container(alloc);
    }
    is_wrap_ = true;
    data_ = const_cast<char *>(text);
    if constexpr (!HAS_LENGTH) {
      length = strlen(text);
    }
    length_ = length;
    max_length_ = length_;
  }

  /** Strong copy operation */
  template <bool IS_ASSIGN, bool HAS_LENGTH>
  HSHM_CROSS_FUN void shm_strong_copy_op(
      const hipc::CtxAllocator<AllocT> &alloc, const char *text,
      size_t length) {
    is_wrap_ = false;
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
  string_templ &operator=(string_templ &&other) noexcept {
    if (this != &other) {
      shm_move_op<true>(GetCtxAllocator(), std::move(other));
    }
    return *this;
  }

  /** SHM move operator. */
  template <bool IS_ASSIGN>
  HSHM_CROSS_FUN void shm_move_op(const hipc::CtxAllocator<AllocT> &alloc,
                                  string_templ &&other) noexcept {
    if constexpr (!IS_ASSIGN) {
      init_shm_container(alloc);
    }
    if (is_wrap_ || GetAllocator() == other.GetAllocator()) {
      move_copy(other);
      other.SetNull();
    } else {
      _create_str(other.data(), other.size());
      other.shm_destroy();
    }
  }

  /** Move copy */
  HSHM_INLINE_CROSS_FUN
  void move_copy(const string_templ &other) {
    length_ = other.length_;
    max_length_ = other.max_length_;
    text_ = other.text_;
    is_wrap_ = other.is_wrap_;
    if (length_ <= SSO && !is_wrap_) {
      memcpy(sso_, other.sso_, other.length_ + 1);
    }
  }

  /**====================================
   * Destructors
   * ===================================*/

  /** Check if this string is NULL */
  HSHM_INLINE_CROSS_FUN bool IsNull() const { return length_ == 0; }

  /** Set this string to NULL */
  HSHM_INLINE_CROSS_FUN void SetNull() {
    text_.SetNull();
    length_ = 0;
    is_wrap_ = false;
  }

  /** Destroy the shared-memory data. */
  HSHM_INLINE_CROSS_FUN void shm_destroy_main() {
    if (max_length_ > SSO && !is_wrap_) {
      FullPtr<void> full_ptr(GetAllocator()->template Convert<void>(text_), text_);
      GetAllocator()->template Free<void>(GetMemCtx(), full_ptr);
    }
  }

  /**====================================
   * String Operations
   * ===================================*/

  /** Get character at index i in the string */
  HSHM_INLINE_CROSS_FUN char &operator[](size_t i) { return data()[i]; }

  /** Get character at index i in the string */
  HSHM_INLINE_CROSS_FUN const char &operator[](size_t i) const {
    return data()[i];
  }

  /** Hash function */
  HSHM_CROSS_FUN size_t Hash() const {
    return string_hash<string_templ>(*this);
  }

  /** Convert into a std::string */
  HSHM_INLINE_HOST_FUN std::string str() const { return {c_str(), length_}; }

  /** Get the size of the current string */
  HSHM_INLINE_CROSS_FUN size_t size() const { return length_; }

  /** Empty */
  bool empty() const {
    return size() == 0;
  }

  /** Get a constant reference to the C-style string */
  HSHM_INLINE_CROSS_FUN const char *c_str() const { return data(); }

  /** Get a constant reference to the C-style string */
  HSHM_INLINE_CROSS_FUN const char *data() const {
    if constexpr (FLAGS & StringFlags::kWrap) {
      if (is_wrap_) {
        return data_;
      }
    }
    if (length_ <= SSO) {
      return sso_;
    } else {
      return GetAllocator()->template Convert<char, Pointer>(text_);
    }
  }

  /** Get a mutable reference to the C-style string */
  HSHM_INLINE_CROSS_FUN char *data() {
    if constexpr (FLAGS & StringFlags::kWrap) {
      if (is_wrap_) {
        return data_;
      }
    }
    if (length_ <= SSO) {
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
      return;
    }
    size_t orig_length = length_;
    char *orig_data = data();
    Pointer orig_text = text_;
    length_ = new_size;

    // WRAP cases
    if constexpr (FLAGS & StringFlags::kWrap) {
      if (is_wrap_) {
        if (new_size < max_length_) {
        } else {
          _create_str(orig_data, orig_length, new_size);
        }
        return;
      }
    }

    // SSO cases
    if (orig_length <= SSO) {
      if (new_size <= SSO) {
      } else {
        _create_str(orig_data, orig_length, new_size);
      }
      return;
    }

    // Buffer cases
    if (new_size > orig_length) {
      // Make the current buffer larger
      FullPtr<void> old_full_ptr(GetAllocator()->template Convert<void>(text_), text_);
      auto new_full_ptr = GetAllocator()->template Reallocate<void, Pointer>(GetMemCtx(), old_full_ptr,
                                                   new_size);
      text_ = new_full_ptr.shm_;
      max_length_ = new_size;
    } else if (new_size <= SSO) {
      // Free current buffer & use SSO
      _create_str(orig_data, orig_length, new_size);
      FullPtr<void> full_ptr(GetAllocator()->template Convert<void>(orig_text), orig_text);
      GetAllocator()->template Free<void>(GetMemCtx(), full_ptr);
    }
  }

  /**====================================
   * Serialization
   * ===================================*/

  /** Serialize */
  template <typename Ar>
  HSHM_CROSS_FUN void save(Ar &ar) const {
    save_string<Ar, string_templ>(ar, *this);
  }

  /** Deserialize */
  template <typename Ar>
  HSHM_CROSS_FUN void load(Ar &ar) {
    load_string<Ar, string_templ>(ar, *this);
  }

  /** ostream */
  friend std::ostream &operator<<(std::ostream &os, const string_templ &str) {
    os << str.str();
    return os;
  }

  /**====================================
   * Comparison Operations
   * ===================================*/

#define HSHM_STR_CMP_OPERATOR(op)                                              \
  bool operator TYPE_UNWRAP(op)(const char *other) const {                     \
    return hshm::strncmp(data(), size(), other, hshm::strlen(other)) op 0;     \
  }                                                                            \
  bool operator op(const std::string &other) const {                           \
    return hshm::strncmp(data(), size(), other.data(), other.size()) op 0;     \
  }                                                                            \
  template <size_t SSO1, u32 FLAGS1, HSHM_CLASS_TEMPL2>                        \
  bool operator op(                                                            \
      const string_templ<SSO1, FLAGS1, HSHM_CLASS_TEMPL_ARGS2> &other) const { \
    return hshm::strncmp(data(), size(), other.data(), other.size()) op 0;     \
  }

  HSHM_STR_CMP_OPERATOR(==)  // NOLINT
  HSHM_STR_CMP_OPERATOR(!=)  // NOLINT
  HSHM_STR_CMP_OPERATOR(<)   // NOLINT
  HSHM_STR_CMP_OPERATOR(>)   // NOLINT
  HSHM_STR_CMP_OPERATOR(<=)  // NOLINT
  HSHM_STR_CMP_OPERATOR(>=)  // NOLINT
#undef HSHM_STR_CMP_OPERATOR

 private:
  HSHM_INLINE_CROSS_FUN void _create_str(size_t length) {
    is_wrap_ = false;
    if (length <= SSO) {
      length_ = length;
      max_length_ = SSO;
    } else {
      auto full_ptr = GetAllocator()->template Allocate<void>(GetMemCtx(), length);
      text_ = full_ptr.shm_;
      length_ = length;
      max_length_ = length;
    }
  }

  HSHM_INLINE_CROSS_FUN void _create_str(const char *text, size_t length) {
    _create_str(length);
    char *str = data();
    memcpy(str, text, length);
    // str[length] = 0;
  }

  HSHM_INLINE_CROSS_FUN void _create_str(const char *text, size_t orig_length,
                                         size_t new_length) {
    _create_str(new_length);
    char *str = data();
    memcpy(str, text, orig_length < new_length ? orig_length : new_length);
  }
};

using string = string_templ<HSHM_STRING_SSO, 0>;
using charbuf = string;

}  // namespace hshm::ipc

namespace hshm {

template <size_t SSO, u32 FLAGS, HSHM_CLASS_TEMPL_WITH_PRIV_DEFAULTS>
using string_templ = ipc::string_templ<SSO, FLAGS, HSHM_CLASS_TEMPL_ARGS>;

using string = string_templ<HSHM_STRING_SSO, 0>;
using charbuf = string;
using charwrap = ipc::string_templ<HSHM_STRING_SSO, hipc::StringFlags::kWrap>;

}  // namespace hshm

/** std::hash function for string */
namespace std {
template <size_t SSO, hshm::u32 FLAGS, HSHM_CLASS_TEMPL>
struct hash<hshm::ipc::string_templ<SSO, FLAGS, HSHM_CLASS_TEMPL_ARGS>> {
  size_t operator()(
      const hshm::ipc::string_templ<SSO, FLAGS, HSHM_CLASS_TEMPL_ARGS> &text)
      const {
    return text.Hash();
  }
};
}  // namespace std

/** hshm::hash function for string */
namespace hshm {
template <size_t SSO, u32 FLAGS, HSHM_CLASS_TEMPL>
struct hash<hshm::ipc::string_templ<SSO, FLAGS, HSHM_CLASS_TEMPL_ARGS>> {
  HSHM_CROSS_FUN size_t operator()(
      const hshm::ipc::string_templ<SSO, FLAGS, HSHM_CLASS_TEMPL_ARGS> &text)
      const {
    return text.Hash();
  }
};
}  // namespace hshm

#undef CLASS_NAME
#undef CLASS_NEW_ARGS

#endif  // HSHM_DATA_STRUCTURES_LOCKLESS_STRING_H_
