//
// Created by lukemartinlogan on 8/12/23.
//

#ifndef HERMES_SHM_INCLUDE_HERMES_SHM_DATA_STRUCTURES_SERIALIZATION_SERIALIZE_VECTOR_H_
#define HERMES_SHM_INCLUDE_HERMES_SHM_DATA_STRUCTURES_SERIALIZATION_SERIALIZE_VECTOR_H_

#include <stddef.h>

/** Serialize a generic vector */
template <typename A, typename ContainerT, typename T>
void save_vec(A &ar, ContainerT &obj) {
  ar << obj.size();
  for (auto iter = obj.cbegin(); iter != obj.cend(); ++iter) {
    ar << (*iter);
  }
}
/** Deserialize a generic vector */
template <typename A, typename ContainerT, typename T>
void load_vec(A &ar, ContainerT &obj) {
  size_t size;
  ar >> size;
  obj.resize(size);
  for (size_t i = 0; i < size; ++i) {
    ar >> (obj[i]);
  }
}

/** Serialize a generic list */
template <typename A, typename ContainerT, typename T>
void save_list(A &ar, ContainerT &obj) {
  ar << obj.size();
  for (auto iter = obj.cbegin(); iter != obj.cend(); ++iter) {
    ar << *(*iter);
  }
}
/** Deserialize a generic list */
template <typename A, typename ContainerT, typename T>
void load_list(A &ar, ContainerT &obj) {
  size_t size;
  ar >> size;
  for (int i = 0; i < size; ++i) {
    obj->emplace_back();
    auto last = obj->back();
    ar >> (*last);
  }
}

/** Serialize a generic string. */
template <typename A, typename StringT>
void save_string(A &ar, StringT &text) {
  ar << text.size();
  ar.write(text.data(), text.size());
}
/** Deserialize a generic string. */
template <typename A, typename StringT>
void load_string(A &ar, StringT &text) {
  size_t size;
  ar >> size;
  text.resize(size);
  ar.read(text.data(), text.size());
}

#endif // HERMES_SHM_INCLUDE_HERMES_SHM_DATA_STRUCTURES_SERIALIZATION_SERIALIZE_VECTOR_H_
