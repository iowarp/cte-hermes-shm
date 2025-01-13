//
// Created by llogan on 10/17/24.
//

#ifndef HSHM_SHM_INCLUDE_HSHM_SHM_DATA_STRUCTURES_CONTAINERS_HASH_H_
#define HSHM_SHM_INCLUDE_HSHM_SHM_DATA_STRUCTURES_CONTAINERS_HASH_H_

#include <cstddef>

#include "hermes_shm/constants/macros.h"

namespace hshm {

/** General hash template */
template <typename T>
class hash;

/** String hash function */
template <typename StringT>
HSHM_CROSS_FUN size_t string_hash(const StringT &text) {
  size_t sum = 0;
  for (size_t i = 0; i < text.size(); ++i) {
    auto shift = static_cast<size_t>(i % sizeof(size_t));
    auto c = static_cast<size_t>((unsigned char)text[i]);
    sum = 31 * sum + (c << shift);
  }
  return sum;
}

/** Pointer hash function */
template <typename T>
struct hash<T *> {
  HSHM_CROSS_FUN size_t operator()(T *const &ptr) const {
    return reinterpret_cast<size_t>(ptr);
  }
};

/** Integer hash function */
template <typename T>
HSHM_INLINE_CROSS_FUN static size_t number_hash(const T &val) {
  if constexpr (sizeof(T) == 1) {
    return static_cast<size_t>(val);
  } else if constexpr (sizeof(T) == 2) {
    return static_cast<size_t>(val);
  } else if constexpr (sizeof(T) == 4) {
    return static_cast<size_t>(val);
  } else if constexpr (sizeof(T) == 8) {
    return static_cast<size_t>(val);
  } else {
    return 0;
  }
}

/** HSHM integer hash */
#define HSHM_INTEGER_HASH(T)                                  \
  template <>                                                 \
  struct hash<T> {                                            \
    HSHM_CROSS_FUN size_t operator()(const T &number) const { \
      return number_hash(number);                             \
    }                                                         \
  };

HSHM_INTEGER_HASH(bool);
HSHM_INTEGER_HASH(char);
HSHM_INTEGER_HASH(short);
HSHM_INTEGER_HASH(int);
HSHM_INTEGER_HASH(long);
HSHM_INTEGER_HASH(long long);
HSHM_INTEGER_HASH(float);
HSHM_INTEGER_HASH(double);

HSHM_INTEGER_HASH(unsigned char);
HSHM_INTEGER_HASH(unsigned short);
HSHM_INTEGER_HASH(unsigned int);
HSHM_INTEGER_HASH(unsigned long);
HSHM_INTEGER_HASH(unsigned long long);

}  // namespace hshm

#endif  // HSHM_SHM_INCLUDE_HSHM_SHM_DATA_STRUCTURES_CONTAINERS_HASH_H_
