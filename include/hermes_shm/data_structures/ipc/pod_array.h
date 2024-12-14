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
template <typename T, int SO = sizeof(hipc::vector<T>) / sizeof(T) + 1>
struct pod_array {
  int size_;
  union {
    delay_ar<T> cache_[SO];
    delay_ar<hipc::vector<T>> vec_;
  };

  /** Default constructor */
  HSHM_INLINE_CROSS_FUN
  pod_array() : size_(0) {}

  /** Serialize */
  template <typename Ar>
  HSHM_CROSS_FUN void serialize(Ar& ar) {
    ar & size_;
    resize(size_);
    if (size_ > SO) {
      ar & vec_;
    } else {
      for (int i = 0; i < size_; ++i) {
        ar& cache_[i];
      }
    }
  }

  /** Construct */
  template <typename AllocT>
  HSHM_INLINE_CROSS_FUN void construct(const hipc::CtxAllocator<AllocT>& alloc,
                                       int size = 0) {
    HSHM_MAKE_AR0(vec_, alloc);
    if (size) {
      resize(size);
    }
  }

  /** Reserve */
  HSHM_INLINE_CROSS_FUN
  void resize(int size) {
    if (size > SO) {
      vec_.get_ref().resize(size);
    }
    size_ = size;
  }

  /** Destroy */
  HSHM_INLINE_CROSS_FUN
  void destroy() { HSHM_DESTROY_AR(vec_); }

  /** Get */
  HSHM_INLINE_CROSS_FUN
  delay_ar<T>* get() {
    if (size_ > SO) {
      return vec_.get_ref().data_ar();
    }
    return cache_;
  }

  /** Get (const) */
  HSHM_INLINE_CROSS_FUN
  const delay_ar<T>* get() const {
    if (size_ > SO) {
      return vec_.get_ref().data_ar();
    }
    return cache_;
  }

  /** Index operator */
  template <typename IdxType>
  HSHM_INLINE_CROSS_FUN T& operator[](IdxType i) {
    return *(get()[i]);
  }

  /** Index operator (const) */
  template <typename IdxType>
  HSHM_INLINE_CROSS_FUN const T& operator[](IdxType i) const {
    return *(get()[i]);
  }
};

}  // namespace hshm::ipc

namespace hshm {
using hshm::ipc::pod_array;
}  // namespace hshm

#endif  // HERMES_SHM_INCLUDE_HERMES_SHM_DATA_STRUCTURES_IPC_POD_ARRAY_H_
