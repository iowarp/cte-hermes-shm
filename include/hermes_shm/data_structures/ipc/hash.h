//
// Created by llogan on 10/17/24.
//

#ifndef HERMES_SHM_INCLUDE_HERMES_SHM_DATA_STRUCTURES_CONTAINERS_HASH_H_
#define HERMES_SHM_INCLUDE_HERMES_SHM_DATA_STRUCTURES_CONTAINERS_HASH_H_

namespace hshm {

/** General hash template */
template<typename T>
class hash;

/** String hash function */
template<typename StringT>
HSHM_CROSS_FUN size_t string_hash(const StringT &text) {
  size_t sum = 0;
  for (size_t i = 0; i < text.size(); ++i) {
    auto shift = static_cast<size_t>(i % sizeof(size_t));
    auto c = static_cast<size_t>((unsigned char)text[i]);
    sum = 31*sum + (c << shift);
  }
  return sum;
}

/** Integer hash function */
template<typename T>
HSHM_INLINE_CROSS static size_t number_hash(const T &val) {
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
#define HERMES_INTEGER_HASH(T) \
template<> \
struct hash<T> { \
  HSHM_CROSS_FUN size_t operator()(const T &number) const { return number_hash(number); } \
};

HERMES_INTEGER_HASH(bool);
HERMES_INTEGER_HASH(char);
HERMES_INTEGER_HASH(short);
HERMES_INTEGER_HASH(int);
HERMES_INTEGER_HASH(long);
HERMES_INTEGER_HASH(long long);
HERMES_INTEGER_HASH(float);
HERMES_INTEGER_HASH(double);

HERMES_INTEGER_HASH(unsigned char);
HERMES_INTEGER_HASH(unsigned short);
HERMES_INTEGER_HASH(unsigned int);
HERMES_INTEGER_HASH(unsigned long);
HERMES_INTEGER_HASH(unsigned long long);

}  // namespace hshm

#endif //HERMES_SHM_INCLUDE_HERMES_SHM_DATA_STRUCTURES_CONTAINERS_HASH_H_
