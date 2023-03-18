//
// Created by lukemartinlogan on 3/17/23.
//

#ifndef HERMES_SHM_INCLUDE_HERMES_SHM_DATA_STRUCTURES_IPC_CONVERTERS_H_
#define HERMES_SHM_INCLUDE_HERMES_SHM_DATA_STRUCTURES_IPC_CONVERTERS_H_

#include "hermes_shm/data_structures/data_structure.h"
#include <vector>
#include <list>

namespace hermes_shm {

/** Convert an iterable object into a vector */
template<typename T, typename SharedT>
std::vector<T> to_stl_vector(const SharedT &other) {
  std::vector<T> vec;
  vec.reserve(other.size());
  for (hipc::Ref<T> obj : other) {
    vec.emplace_back(*obj);
  }
  return vec;
}

/** Convert an iterable object into a list */
template<typename T, typename SharedT>
std::list<T> to_stl_list(const SharedT &other) {
  std::list<T> vec;
  for (hipc::Ref<T> obj : other) {
    vec.emplace_back(*obj);
  }
  return vec;
}

}  // namespace hermes_shm

#endif //HERMES_SHM_INCLUDE_HERMES_SHM_DATA_STRUCTURES_IPC_CONVERTERS_H_
