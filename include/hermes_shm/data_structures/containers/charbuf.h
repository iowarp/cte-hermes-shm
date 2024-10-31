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

#ifndef HERMES_INCLUDE_HERMES_TYPES_CHARBUF_H_
#define HERMES_INCLUDE_HERMES_TYPES_CHARBUF_H_

#include "hermes_shm/types/real_number.h"
#include "hermes_shm/memory/memory_manager_.h"
#include "hermes_shm/data_structures/serialization/serialize_common.h"
#include "hermes_shm/data_structures/containers/internal/hshm_container.h"
#include "string_common.h"
#include <string>

namespace hshm {

/** An uninterpreted array of bytes */
struct charbuf {
  /**====================================
   * Variables & Types
   * ===================================*/
  HSHM_CONTAINER_BASE_TEMPLATE
  char *data_; /**< The pointer to data */
  size_t size_; /**< The size of data */
  size_t total_size_; /**< The true size of data buffer */
  bool destructable_;  /**< Whether or not this container owns data */

  /**====================================
   * Default Constructor
   * ===================================*/

  /** Default constructor */
  HSHM_INLINE_CROSS_FUN charbuf()
  : alloc_(nullptr), data_(nullptr), size_(0),
    total_size_(0), destructable_(false) {}

  /**====================================
   * Destructor
   * ===================================*/

  /** Destructor */
  HSHM_INLINE_CROSS_FUN ~charbuf() { Free(); }

  /**====================================
   * Emplace Constructors
   * ===================================*/

  /** Size-based constructor */
  HSHM_INLINE_CROSS_FUN explicit charbuf(size_t size) {
    Allocate(HERMES_MEMORY_MANAGER->GetDefaultAllocator(), size);
  }

  /** Allocator + Size-based constructor */
  HSHM_INLINE_CROSS_FUN explicit charbuf(hipc::Allocator *alloc, size_t size) {
    Allocate(alloc, size);
  }

  /**====================================
  * Reference Constructors
  * ===================================*/

  /** Reference constructor. From char* + size */
  HSHM_INLINE_CROSS_FUN explicit charbuf(char *data, size_t size)
  : alloc_(nullptr), data_(data), size_(size),
    total_size_(size), destructable_(false) {}

  /**
   * Reference constructor. From const char*
   * We assume that the data will not be modified by the user, but
   * we must cast away the const anyway.
   * */
  HSHM_INLINE_CROSS_FUN explicit charbuf(const char *data, size_t size)
  : alloc_(nullptr), data_(const_cast<char*>(data)),
    size_(size), total_size_(size), destructable_(false) {}

  /**====================================
   * Copy Constructors
   * ===================================*/

  /** Copy constructor. From std::string. */
  HSHM_INLINE_CROSS_FUN explicit charbuf(const std::string &data) {
    Allocate(HERMES_MEMORY_MANAGER->GetDefaultAllocator(), data.size());
    memcpy(data_, data.data(), data.size());
  }

  /** Copy constructor. From charbuf. */
  HSHM_INLINE_CROSS_FUN charbuf(const charbuf &other) {
    if (!Allocate(HERMES_MEMORY_MANAGER->GetDefaultAllocator(),
                  other.size())) {
      return;
    }
    memcpy(data_, other.data(), size());
  }

  /** Copy assignment operator */
  HSHM_INLINE_CROSS_FUN charbuf& operator=(const charbuf &other) {
    if (this != &other) {
      Free();
      if (!Allocate(HERMES_MEMORY_MANAGER->GetDefaultAllocator(),
                    other.size())) {
        return *this;
      }
      memcpy(data_, other.data(), size());
    }
    return *this;
  }

  /**====================================
   * Move Constructors
   * ===================================*/

  /** Move constructor */
  HSHM_CROSS_FUN charbuf(charbuf &&other) {
    alloc_ = other.alloc_;
    data_ = other.data_;
    size_ = other.size_;
    total_size_ = other.total_size_;
    destructable_ = other.destructable_;
    other.size_ = 0;
    other.total_size_ = 0;
    other.destructable_ = false;
  }

  /** Move assignment operator */
  HSHM_CROSS_FUN charbuf& operator=(charbuf &&other) noexcept {
    if (this != &other) {
      Free();
      alloc_ = other.alloc_;
      data_ = other.data_;
      size_ = other.size_;
      total_size_ = other.total_size_;
      destructable_ = other.destructable_;
      other.size_ = 0;
      other.total_size_ = 0;
      other.destructable_ = false;
    }
    return *this;
  }

  /**====================================
   * Methods
   * ===================================*/

  /** Destroy and resize */
  HSHM_CROSS_FUN void resize(size_t new_size) {
    if (new_size <= total_size_) {
      size_ = new_size;
      return;
    }
    if (alloc_ == nullptr) {
      alloc_ = HERMES_MEMORY_MANAGER->GetDefaultAllocator();
    }
    if (destructable_) {
      data_ = alloc_->ReallocatePtr<char>(
          ThreadId::GetNull(), data_, new_size);
    } else {
      data_ = alloc_->AllocatePtr<char>(
          ThreadId::GetNull(), new_size);
    }
    destructable_ = true;
    size_ = new_size;
    total_size_ = new_size;
  }

  /** Reference data */
  HSHM_INLINE_CROSS_FUN char* data() {
    return data_;
  }

  /** Reference data */
  HSHM_INLINE_CROSS_FUN char* data() const {
    return data_;
  }

  /** Reference size */
  HSHM_INLINE_CROSS_FUN size_t size() const {
    return size_;
  }

  /** Convert to std::string */
  HSHM_INLINE_HOST_FUN const std::string str() const {
    return std::string(data(), size());
  }

  /**====================================
   * Operators
   * ===================================*/

  /** Index operator */
  HSHM_INLINE_CROSS_FUN char& operator[](size_t idx) {
      return data_[idx];
  }

  /** Const index operator */
  HSHM_INLINE_CROSS_FUN const char& operator[](size_t idx) const {
    return data_[idx];
  }

  /** Hash function */
  HSHM_CROSS_FUN size_t Hash() const {
    return string_hash<hshm::charbuf>(*this);
  }

  /**====================================
   * Serialization
   * ===================================*/

  /** Serialize */
  template <typename Ar>
  HSHM_CROSS_FUN void save(Ar &ar) const {
    save_string<Ar, charbuf>(ar, *this);
  }

  /** Deserialize */
  template <typename Ar>
  HSHM_CROSS_FUN void load(Ar &ar) {
    load_string<Ar, charbuf>(ar, *this);
  }

  /**====================================
   * Comparison Operators
   * ===================================*/
#define HERMES_STR_CMP_OPERATOR(op) \
  bool operator TYPE_UNWRAP(op)(const char *other) const { \
    return hshm::strncmp(data(), size(), other, hshm::strlen(other)) op 0; \
  } \
  bool operator op(const std::string &other) const { \
  return hshm::strncmp(data(), size(), other.data(), other.size()) op 0; \
  } \
  bool operator op(const charbuf &other) const { \
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
  /**====================================
   * Internal functions
   * ===================================*/

  /** Allocate charbuf */
  HSHM_CROSS_FUN bool Allocate(hipc::Allocator *alloc, size_t size) {
    hipc::OffsetPointer p;
    if (size == 0) {
      alloc_ = nullptr;
      data_ = nullptr;
      size_ = 0;
      total_size_ = 0;
      destructable_ = false;
      return false;
    }
    alloc_ = alloc;
    data_ = alloc->AllocatePtr<char>(
        hshm::ThreadId::GetNull(), size, p);
    size_ = size;
    total_size_ = size;
    destructable_ = true;
    return true;
  }

  /** Explicitly free the charbuf */
  HSHM_CROSS_FUN void Free() {
    if (destructable_ && data_ && total_size_) {
      alloc_->FreePtr<char>(hshm::ThreadId::GetNull(), data_);
    }
  }
};

typedef charbuf string;

}  // namespace hshm

/** std::hash function for string */
namespace std {
template<>
struct hash<hshm::charbuf> {
  HSHM_CROSS_FUN size_t operator()(const hshm::charbuf &text) const {
    return text.Hash();
  }
};
}  // namespace std

/** hshm::hash function for string */
namespace hshm {
template<>
struct hash<hshm::charbuf> {
  HSHM_CROSS_FUN size_t operator()(const hshm::charbuf &text) const {
    return text.Hash();
  }
};
}  // namespace hshm

#endif  // HERMES_INCLUDE_HERMES_TYPES_CHARBUF_H_
