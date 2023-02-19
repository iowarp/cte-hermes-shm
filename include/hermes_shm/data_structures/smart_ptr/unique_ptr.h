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


/**
 * Create a data structure in either shared or regular memory.
 * Destroy the data structure when the unique pointer is deconstructed
 * */

#ifndef HERMES_DATA_STRUCTURES_UNIQUE_PTR_H_
#define HERMES_DATA_STRUCTURES_UNIQUE_PTR_H_

#include "hermes_shm/memory/memory.h"
#include "hermes_shm/data_structures/data_structure.h"

namespace hermes::ipc {

/**
 * MACROS to simplify the unique_ptr namespace
 * */
#define CLASS_NAME unique_ptr
#define TYPED_CLASS unique_ptr<T>

/**
 * Create a unique instance of a shared-memory data structure.
 * The process which creates the data structure owns it and is responsible
 * for freeing. Other processes can deserialize the data structure, but
 * cannot destroy its data.
 * */
template<typename T>
class unique_ptr : public ShmSmartPtr<T> {
 public:
  SHM_SMART_PTR_TEMPLATE(T);

 public:
  /** Default constructor */
  unique_ptr() = default;

  /** Destroys all allocated memory */
  ~unique_ptr() = default;

  /** Allocates + constructs an object in shared memory */
  template<typename ...Args>
  void shm_init(Args&& ...args) {
    obj_.shm_init(std::forward<Args>(args)...);
  }

  /** Disable the copy constructor */
  unique_ptr(const unique_ptr &other) = delete;

  /** Move constructor */
  unique_ptr(unique_ptr&& other) noexcept {
    obj_ = std::move(other.obj_);
  }

  /** Move assignment operator */
  unique_ptr<T>& operator=(unique_ptr<T> &&other) {
    if (this != &other) {
      obj_ = std::move(other.obj_);
    }
    return *this;
  }

  /** Constructor. From a TypedPointer<uptr<T>> */
  explicit unique_ptr(TypedPointer<TYPED_CLASS> &ar) {
    obj_.shm_deserialize(ar);
  }

  /** Constructor. From a TypedAtomicPointer<uptr<T>> */
  explicit unique_ptr(TypedAtomicPointer<TYPED_CLASS> &ar) {
    obj_.shm_deserialize(ar);
  }

  /** Serialize into a TypedPointer<unique_ptr> */
  SHM_SERIALIZE_DESERIALIZE_WRAPPER(unique_ptr)
};

template<typename T>
using uptr = unique_ptr<T>;

template<typename T, typename ...Args>
static uptr<T> make_uptr(Args&& ...args) {
  uptr<T> ptr;
  ptr.shm_init(std::forward<Args>(args)...);
  return ptr;
}

}  // namespace hermes::ipc

#undef CLASS_NAME
#undef TYPED_CLASS

namespace std {

/** Hash function for unique_ptr */
template<typename T>
struct hash<hermes::ipc::unique_ptr<T>> {
  size_t operator()(const hermes::ipc::unique_ptr<T> &obj) const {
    return std::hash<T>{}(obj.get_ref_const());
  }
};

}  // namespace std

#endif  // HERMES_DATA_STRUCTURES_UNIQUE_PTR_H_
