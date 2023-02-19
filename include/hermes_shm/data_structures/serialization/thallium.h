/*
 * Copyright (C) 2022  SCS Lab <scslab@iit.edu>,
 * Luke Logan <llogan@hawk.iit.edu>,
 * Jaime Cernuda Garcia <jcernudagarcia@hawk.iit.edu>
 * Jay Lofstead <gflofst@sandia.gov>,
 * Anthony Kougkas <akougkas@iit.edu>,
 * Xian-He Sun <sun@iit.edu>
 *
 * This file is part of hermes
 *
 * hermes is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public
 * License along with this program.  If not, see
 * <http://www.gnu.org/licenses/>.
 */


#ifndef HERMES_DATA_STRUCTURES_SERIALIZATION_THALLIUM_H_
#define HERMES_DATA_STRUCTURES_SERIALIZATION_THALLIUM_H_

#include <thallium.hpp>
#include <hermes_shm/data_structures/string.h>
#include <hermes_shm/data_structures/thread_unsafe/vector.h>
#include <hermes_shm/data_structures/thread_unsafe/list.h>
#include <hermes_shm/data_structures/thread_safe/unordered_map.h>

namespace thallium {

namespace hipc = hermes::ipc;

/**
 *  Lets Thallium know how to serialize an hipc::allocator_id.
 *
 * This function is called implicitly by Thallium.
 *
 * @param ar An archive provided by Thallium.
 * @param vec The vector to serialize.
 */
template <typename A>
void serialize(A &ar, hipc::allocator_id_t &alloc_id) {
  ar &alloc_id.int_;
}

/**
 *  Lets Thallium know how to serialize an hipc::vector.
 *
 * This function is called implicitly by Thallium.
 *
 * @param ar An archive provided by Thallium.
 * @param vec The vector to serialize.
 */
template <typename A, typename T>
void save(A &ar, hipc::vector<T> &vec) {
  ar << vec.GetAllocatorId();
  ar << vec.size();
  for (auto iter = vec.cbegin(); iter != vec.cend(); ++iter) {
    ar << *(*iter);
  }
}

/**
 *  Lets Thallium know how to deserialize an hipc::vector.
 *
 * This function is called implicitly by Thallium.
 *
 * @param ar An archive provided by Thallium.
 * @param target_id The vector to serialize.
 */
template <typename A, typename T>
void load(A &ar, hipc::vector<T> &vec) {
  size_t size;
  hipc::allocator_id_t alloc_id;
  ar >> alloc_id;
  ar >> size;
  auto alloc = HERMES_MEMORY_MANAGER->GetAllocator(alloc_id);
  vec.shm_init(alloc);
  vec.resize(size);
  for (auto iter = vec.begin(); iter != vec.end(); ++iter) {
    vec.emplace_back(std::move(**iter));
  }
}

/**
 *  Lets Thallium know how to serialize an hipc::string.
 *
 * This function is called implicitly by Thallium.
 *
 * @param ar An archive provided by Thallium.
 * @param text The string to serialize
 */
template <typename A>
void save(A &ar, hipc::string &text) {
  ar << text.GetAllocator()->GetId();
  ar << text.size();
  ar.write(text.data_mutable(), text.size());
}

/**
 *  Lets Thallium know how to deserialize an hipc::string.
 *
 * This function is called implicitly by Thallium.
 *
 * @param ar An archive provided by Thallium.
 * @param target_id The string to deserialize.
 */
template <typename A>
void load(A &ar, hipc::string &text) {
  hipc::allocator_id_t alloc_id;
  size_t size;
  ar >> alloc_id;
  ar >> size;
  auto alloc = HERMES_MEMORY_MANAGER->GetAllocator(alloc_id);
  text.shm_init(alloc, size);
  ar.read(text.data_mutable(), size);
}

}  // namespace thallium

#endif  // HERMES_DATA_STRUCTURES_SERIALIZATION_THALLIUM_H_
