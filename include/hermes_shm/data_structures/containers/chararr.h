//
// Created by llogan on 10/16/24.
//

#ifndef HERMES_SHM_INCLUDE_HERMES_SHM_DATA_STRUCTURES_CONTAINERS_chararr_H_
#define HERMES_SHM_INCLUDE_HERMES_SHM_DATA_STRUCTURES_CONTAINERS_chararr_H_

#include "hermes_shm/constants/macros.h"

namespace hshm {

template<int LENGTH=4096>
class chararr {
 public:
  char buf_[LENGTH];
  int length_;

 public:
  /**====================================
   * Basic Constructors
   * ===================================*/
  /** Default constructor */
  HSHM_CROSS_FUN
  chararr() = default;

  /** Size-based constructor */
  HSHM_INLINE_CROSS_FUN explicit chararr(size_t size) {
    resize(length_);
  }

  /**====================================
   * Copy Constructors
   * ===================================*/
  /** Construct from const char* */
  HSHM_CROSS_FUN
  chararr(const char *data) {
    length_ = strnlen(data, LENGTH);
    memcpy(buf_, data, length_);
  }

  /** Construct from sized char* */
  HSHM_CROSS_FUN
  chararr(const char *data, size_t length) {
    length_ = strnlen(data, LENGTH);
    memcpy(buf_, data, length);
  }

  /** Construct from std::string */
  HSHM_CROSS_FUN
  chararr(const std::string &data) {
    length_ = data.size();
    memcpy(buf_, data.data(), length_);
  }

  /** Construct from chararr */
  HSHM_CROSS_FUN
  chararr(const chararr &data) {
    length_ = data.size();
    memcpy(buf_, data.data(), length_);
  }

  /** Copy assignment operator */
  HSHM_INLINE_CROSS_FUN chararr& operator=(const chararr &other) {
    if (this != &other) {
      length_ = other.size();
      memcpy(buf_, other.data(), length_);
    }
    return *this;
  }

  /**====================================
   * Move Constructors
   * ===================================*/

  /** Move constructor */
  HSHM_CROSS_FUN chararr(chararr &&other) {
    length_ = other.length_;
    memcpy(buf_, other.buf_, length_);
  }

  /** Move assignment operator */
  HSHM_CROSS_FUN chararr& operator=(chararr &&other) noexcept {
    if (this != &other) {
      length_ = other.length_;
      memcpy(buf_, other.buf_, length_);
    }
    return *this;
  }

  /**====================================
   * Methods
   * ===================================*/

  /** Destroy and resize */
  HSHM_CROSS_FUN void resize(size_t new_size) {
    length_ = new_size;
  }

  /** Reference data */
  HSHM_INLINE_CROSS_FUN char* data() {
    return buf_;
  }

  /** Reference data */
  HSHM_INLINE_CROSS_FUN const char* data() const {
    return buf_;
  }

  /** Reference size */
  HSHM_INLINE_CROSS_FUN size_t size() const {
    return length_;
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
    return buf_[idx];
  }

  /** Const index operator */
  HSHM_INLINE_CROSS_FUN const char& operator[](size_t idx) const {
    return buf_[idx];
  }

  /**====================================
   * Serialization
   * ===================================*/

  /** Serialize */
  template <typename Ar>
  HSHM_CROSS_FUN void save(Ar &ar) const {
    save_string<Ar, chararr>(ar, *this);
  }

  /** Deserialize */
  template <typename Ar>
  HSHM_CROSS_FUN void load(Ar &ar) {
    load_string<Ar, chararr>(ar, *this);
  }

  /**====================================
   * Comparison Operators
   * ===================================*/

  HSHM_INLINE_CROSS_FUN int _strncmp(const char *a, size_t len_a,
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
  bool operator TYPE_UNWRAP(op)(const char *other) const { \
    return _strncmp(data(), size(), other, strlen(other)) op 0; \
  } \
  bool operator op(const std::string &other) const { \
    return _strncmp(data(), size(), other.data(), other.size()) op 0; \
  } \
  bool operator op(const chararr &other) const { \
    return _strncmp(data(), size(), other.data(), other.size()) op 0; \
  }

  HERMES_STR_CMP_OPERATOR(==)  // NOLINT
  HERMES_STR_CMP_OPERATOR(!=)  // NOLINT
  HERMES_STR_CMP_OPERATOR(<)  // NOLINT
  HERMES_STR_CMP_OPERATOR(>)  // NOLINT
  HERMES_STR_CMP_OPERATOR(<=)  // NOLINT
  HERMES_STR_CMP_OPERATOR(>=)  // NOLINT
};

}  // namespace hshm

#endif  // HERMES_SHM_INCLUDE_HERMES_SHM_DATA_STRUCTURES_CONTAINERS_chararr_H_
