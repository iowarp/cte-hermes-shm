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

#ifndef HERMES_SHM_INCLUDE_HERMES_SHM_DATA_STRUCTURES_IPC_POD_ARRAY_H_
#define HERMES_SHM_INCLUDE_HERMES_SHM_DATA_STRUCTURES_IPC_POD_ARRAY_H_

#include "vector.h"

namespace hshm::ipc {

/**
 * Small object (so) vector.
 * A vector which avoids memory allocation for a small number of objects
 * The array's size must remain fixed after the first modification to data.
 * */
template<typename T, int SO>
struct pod_array {
  int size_;
  union {
    ShmArchive<T> cache_[SO];
    ShmArchive<hipc::vector<T>> vec_;
  };

  /** Serialize */
  template<typename Ar>
  void serialize(Ar &ar) {
    ar &size_;
    if (size_ > SO) {
      ar &vec_;
    } else {
      for (int i = 0; i < size_; ++i) {
        ar &cache_[i];
      }
    }
  }

  /** Reserve */
  HSHM_ALWAYS_INLINE
  void resize(Allocator *alloc, int size) {
    if (size > SO) {
      HSHM_MAKE_AR0(vec_, alloc);
      vec_.get_ref().reserve(size);
    }
    size_ = size;
  }

  /** Destroy */
  HSHM_ALWAYS_INLINE
  void destroy() {
    if (size_ > SO) {
      HSHM_DESTROY_AR(vec_);
    }
  }

  /** Get */
  HSHM_ALWAYS_INLINE
  ShmArchive<T>* get() {
    if (size_ > SO) {
      return vec_.get_ref().data_ar();
    }
    return cache_;
  }

  /** Get (const) */
  HSHM_ALWAYS_INLINE
  const ShmArchive<T>* get() const {
    if (size_ > SO) {
      return vec_.get_ref().data_ar();
    }
    return cache_;
  }

  /** Index operator */
  HSHM_ALWAYS_INLINE
  ShmArchive<T>& operator[](int i) {
    return get()[i];
  }

  /** Index operator (const) */
  HSHM_ALWAYS_INLINE
  const ShmArchive<T>& operator[](int i) const {
    return get()[i];
  }
};

}  // namespace hshm::ipc

#endif  // HERMES_SHM_INCLUDE_HERMES_SHM_DATA_STRUCTURES_IPC_POD_ARRAY_H_
