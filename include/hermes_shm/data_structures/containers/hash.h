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

}

#endif //HERMES_SHM_INCLUDE_HERMES_SHM_DATA_STRUCTURES_CONTAINERS_HASH_H_
