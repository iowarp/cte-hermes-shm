//
// Created by llogan on 10/16/24.
//

#ifndef HSHM_SHM_INCLUDE_HSHM_SHM_DATA_STRUCTURES_CONTAINERS_chararr_templ_H_
#define HSHM_SHM_INCLUDE_HSHM_SHM_DATA_STRUCTURES_CONTAINERS_chararr_templ_H_

#include "hermes_shm/constants/macros.h"
#include "hermes_shm/data_structures/serialization/serialize_common.h"
#include "string_common.h"

namespace hshm::ipc {

template <int LENGTH, bool WithNull, int FULL_LENGTH = LENGTH + WithNull>
class chararr_templ {
 public:
  char buf_[FULL_LENGTH];
  int length_;

 public:
  /**====================================
   * Basic Constructors
   * ===================================*/
  /** Default constructor */
  HSHM_CROSS_FUN
  chararr_templ() = default;

  /** Size-based constructor */
  HSHM_INLINE_CROSS_FUN explicit chararr_templ(size_t size) { resize(size); }

  /**====================================
   * Copy Constructors
   * ===================================*/
  /** Construct from const char* */
  HSHM_CROSS_FUN
  chararr_templ(const char *data) {
    length_ = hshm::strnlen(data, LENGTH);
    memcpy(buf_, data, length_);
    if constexpr (WithNull) {
      buf_[length_] = '\0';
    }
  }

  /** Construct from sized char* */
  HSHM_CROSS_FUN
  chararr_templ(const char *data, size_t length) {
    if (length > LENGTH) {
      length_ = LENGTH;
    }
    length_ = length;
    memcpy(buf_, data, length);
    if constexpr (WithNull) {
      buf_[length_] = '\0';
    }
  }

  /** Construct from std::string */
  HSHM_HOST_FUN
  chararr_templ(const std::string &data) {
    length_ = data.size();
    memcpy(buf_, data.data(), length_);
    if constexpr (WithNull) {
      buf_[length_] = '\0';
    }
  }

  /** Construct from chararr_templ */
  HSHM_CROSS_FUN
  chararr_templ(const chararr_templ &data) {
    length_ = data.size();
    memcpy(buf_, data.data(), length_);
    if constexpr (WithNull) {
      buf_[length_] = '\0';
    }
  }

  /** Copy assignment operator */
  HSHM_INLINE_CROSS_FUN chararr_templ &operator=(const chararr_templ &other) {
    if (this != &other) {
      length_ = other.size();
      memcpy(buf_, other.data(), length_);
      if constexpr (WithNull) {
        buf_[length_] = '\0';
      }
    }
    return *this;
  }

  /**====================================
   * Move Constructors
   * ===================================*/

  /** Move constructor */
  HSHM_CROSS_FUN chararr_templ(chararr_templ &&other) {
    length_ = other.length_;
    memcpy(buf_, other.buf_, length_);
    if constexpr (WithNull) {
      buf_[length_] = '\0';
    }
  }

  /** Move assignment operator */
  HSHM_CROSS_FUN chararr_templ &operator=(chararr_templ &&other) noexcept {
    if (this != &other) {
      length_ = other.length_;
      memcpy(buf_, other.buf_, length_);
      if constexpr (WithNull) {
        buf_[length_] = '\0';
      }
    }
    return *this;
  }

  /**====================================
   * Methods
   * ===================================*/

  /** Destroy and resize */
  HSHM_CROSS_FUN void resize(size_t new_size) { length_ = new_size; }

  /** Reference data */
  HSHM_INLINE_CROSS_FUN char *data() { return buf_; }

  /** Reference data */
  HSHM_INLINE_CROSS_FUN const char *data() const { return buf_; }

  /** Reference data */
  HSHM_INLINE_CROSS_FUN char *c_str() { return buf_; }

  /** Reference data */
  HSHM_INLINE_CROSS_FUN const char *c_str() const { return buf_; }

  /** Reference size */
  HSHM_INLINE_CROSS_FUN size_t size() const { return length_; }

  /** Convert to std::string */
  HSHM_INLINE_HOST_FUN const std::string str() const {
    return std::string(data(), size());
  }

  /**====================================
   * Operators
   * ===================================*/

  /** Index operator */
  HSHM_INLINE_CROSS_FUN char &operator[](size_t idx) { return buf_[idx]; }

  /** Const index operator */
  HSHM_INLINE_CROSS_FUN const char &operator[](size_t idx) const {
    return buf_[idx];
  }

  /** Hash function */
  HSHM_CROSS_FUN size_t Hash() const {
    return string_hash<chararr_templ>(*this);
  }

  /**====================================
   * Serialization
   * ===================================*/

  /** Serialize */
  template <typename Ar>
  void save(Ar &ar) const {
    hipc::save_string<Ar, chararr_templ>(ar, *this);
  }

  /** Deserialize */
  template <typename Ar>
  void load(Ar &ar) {
    hipc::load_string<Ar, chararr_templ>(ar, *this);
  }

  /**====================================
   * Comparison Operators
   * ===================================*/

#define HSHM_STR_CMP_OPERATOR(op)                                          \
  HSHM_CROSS_FUN                                                           \
  bool operator TYPE_UNWRAP(op)(const char *other) const {                 \
    return hshm::strncmp(data(), size(), other, hshm::strlen(other)) op 0; \
  }                                                                        \
  HSHM_HOST_FUN                                                            \
  bool operator op(const std::string &other) const {                       \
    return hshm::strncmp(data(), size(), other.data(), other.size()) op 0; \
  }                                                                        \
  HSHM_CROSS_FUN                                                           \
  bool operator op(const chararr_templ &other) const {                     \
    return hshm::strncmp(data(), size(), other.data(), other.size()) op 0; \
  }

  HSHM_STR_CMP_OPERATOR(==)  // NOLINT
  HSHM_STR_CMP_OPERATOR(!=)  // NOLINT
  HSHM_STR_CMP_OPERATOR(<)   // NOLINT
  HSHM_STR_CMP_OPERATOR(>)   // NOLINT
  HSHM_STR_CMP_OPERATOR(<=)  // NOLINT
  HSHM_STR_CMP_OPERATOR(>=)  // NOLINT
#undef HSHM_STR_CMP_OPERATOR
};

#if HSHM_IS_HOST
typedef chararr_templ<4095, true> chararr;
#else
typedef chararr_templ<31, true> chararr;
#endif

}  // namespace hshm::ipc

namespace hshm {
template <int LENGTH, bool WithNull>
using chararr_templ = hshm::ipc::chararr_templ<LENGTH, WithNull>;
using hshm::ipc::chararr;
}  // namespace hshm

/** std::hash function for string */
namespace std {
template <int LENGTH, bool WithNull>
struct hash<hshm::chararr_templ<LENGTH, WithNull>> {
  size_t operator()(const hshm::chararr_templ<LENGTH, WithNull> &text) const {
    return text.Hash();
  }
};
}  // namespace std

/** hshm::hash function for string */
namespace hshm {
template <int LENGTH, bool WithNull>
struct hash<hshm::chararr_templ<LENGTH, WithNull>> {
  HSHM_CROSS_FUN size_t
  operator()(const hshm::chararr_templ<LENGTH, WithNull> &text) const {
    return text.Hash();
  }
};
}  // namespace hshm

#endif  // HSHM_SHM_INCLUDE_HSHM_SHM_DATA_STRUCTURES_CONTAINERS_chararr_templ_H_
