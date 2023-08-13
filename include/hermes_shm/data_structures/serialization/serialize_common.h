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

/** Serialize a generic vector */
template <typename A, typename ContainerT, typename T>
void save_vec(A &ar, const ContainerT &obj) {
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
void save_list(A &ar, const ContainerT &obj) {
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
void save_string(A &ar, const StringT &text) {
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

#endif  // HERMES_SHM_SERIALIZE_COMMON_H_
