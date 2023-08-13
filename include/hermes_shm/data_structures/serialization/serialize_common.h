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

#ifndef HERMES_SHM_SERIALIZE_COMMON_H_
#define HERMES_SHM_SERIALIZE_COMMON_H_

#include <stddef.h>
#include <cereal/archives/binary.hpp>

template<typename Ar, typename T>
void write_binary(Ar &ar, const T *data, size_t size) {
  ar(cereal::binary_data(data, size));
}
template<typename Ar, typename T>
void read_binary(Ar &ar, T *data, size_t size) {
  ar(cereal::binary_data(data, size));
}

/** Serialize a generic string. */
template <typename Ar, typename StringT>
void save_string(Ar &ar, const StringT &text) {
  ar << text.size();
  // ar.write(text.data(), text.size());
  write_binary(ar, text.data(), text.size());
}
/** Deserialize a generic string. */
template <typename Ar, typename StringT>
void load_string(Ar &ar, StringT &text) {
  size_t size;
  ar >> size;
  text.resize(size);
  read_binary(ar, text.data(), text.size());
}

/** Serialize a generic vector */
template <typename Ar, typename ContainerT, typename T>
void save_vec(Ar &ar, const ContainerT &obj) {
  if constexpr(std::is_same_v<char, T>) {
    write_binary(ar, (char*)obj.data(), obj.size() * sizeof(T));
  } else {
    ar << obj.size();
    for (auto iter = obj.cbegin(); iter != obj.cend(); ++iter) {
      ar << (*iter);
    }
  }
}
/** Deserialize a generic vector */
template <typename Ar, typename ContainerT, typename T>
void load_vec(Ar &ar, ContainerT &obj) {
  size_t size;
  ar >> size;
  obj.resize(size);
  if constexpr(std::is_same_v<char, T>) {
    read_binary(ar, (char*)obj.data(), obj.size() * sizeof(T));
  } else {
    for (size_t i = 0; i < size; ++i) {
      ar >> (obj[i]);
    }
  }
}

/** Serialize a generic list */
template <typename Ar, typename ContainerT, typename T>
void save_list(Ar &ar, const ContainerT &obj) {
  ar << obj.size();
  for (auto iter = obj.cbegin(); iter != obj.cend(); ++iter) {
    ar << (*iter);
  }
}
/** Deserialize a generic list */
template <typename Ar, typename ContainerT, typename T>
void load_list(Ar &ar, ContainerT &obj) {
  size_t size;
  ar >> size;
  for (int i = 0; i < size; ++i) {
    obj.emplace_back();
    auto &last = obj.back();
    ar >> last;
  }
}

#endif  // HERMES_SHM_SERIALIZE_COMMON_H_
