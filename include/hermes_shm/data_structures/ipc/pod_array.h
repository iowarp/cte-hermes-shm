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
 * This vector is assumed to have its size set exactly once.
 * */
template<typename T, int SO>
struct pod_array {
  int size_;
  union {
    ShmArchive<T> cache_[SO];
    ShmArchive<hipc::vector<T>> vec_;
  };

  /** Reserve */
  HSHM_ALWAYS_INLINE
  void resize(Allocator *alloc, int size) {
    if (size > SO) {
      HSHM_MAKE_AR0(vec_, alloc);
      vec_.get_ref().reserve(size);
    }
    size_ = size;
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
};

}  // namespace hshm::ipc

#endif  // HERMES_SHM_INCLUDE_HERMES_SHM_DATA_STRUCTURES_IPC_POD_ARRAY_H_
