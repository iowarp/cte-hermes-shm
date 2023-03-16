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


#ifndef HERMES_DATA_STRUCTURES_SHM_ARCHIVE_H_
#define HERMES_DATA_STRUCTURES_SHM_ARCHIVE_H_

#include "shm_macros.h"
#include "hermes_shm/memory/memory.h"
#include "shm_deserialize.h"

namespace hermes_shm::ipc {

/**
 * Constructs a TypedPointer in-place
 * */
template<typename T>
class _ShmArchive_Header {
 public:
  typedef typename T::header_t header_t;
  char obj_hdr_[sizeof(header_t)];

 public:
  /** Default constructor */
  _ShmArchive_Header() = default;

  /** Destructor */
  ~_ShmArchive_Header() = default;

  /** Returns a reference to the internal object */
  ShmDeserialize<T> internal_ref(Allocator *alloc) {
    return ShmDeserialize<T>(reinterpret_cast<header_t*>(obj_hdr_), alloc);
  }

  /** Returns a reference to the internal object */
  ShmDeserialize<T> internal_ref(Allocator *alloc) const {
    return ShmDeserialize<T>(reinterpret_cast<header_t*>(obj_hdr_), alloc);
  }

  /** Get reference to object */
  T& get_ref() {
    return reinterpret_cast<T&>(obj_hdr_);
  }
};

/**
 * Constructs an object in-place
 * */
template<typename T>
class _ShmArchive_T {
 public:
  typedef T header_t;
  char obj_hdr_[sizeof(T)]; /**< Store object without constructing */

 public:
  /** Default constructor. */
  _ShmArchive_T() = default;

  /** Destructor. Does nothing. */
  ~_ShmArchive_T() = default;

  /** Returns a reference to the internal object */
  T& internal_ref(Allocator *alloc) {
    (void) alloc;
    return reinterpret_cast<T&>(obj_hdr_);
  }

  /** Returns a reference to the internal object */
  T& internal_ref(Allocator *alloc) const {
    (void) alloc;
    return reinterpret_cast<T&>(obj_hdr_);
  }
};

/**
 * Whether or not to use _ShmArchive or _ShmArchive_T
 * */
#define SHM_MAKE_HEADER_OR_T(T) \
  SHM_X_OR_Y(T, _ShmArchive_Header<T>, _ShmArchive_T<T>)

/**
 * Represents the layout of a data structure in shared memory.
 * */
template<typename T>
class ShmArchive {
 public:
  typedef SHM_DESERIALIZE_OR_REF(T) T_Ar;
  typedef SHM_MAKE_HEADER_OR_T(T) T_Hdr;
  typedef typename T_Hdr::header_t header_t;
  T_Hdr obj_;

  /** Default constructor */
  ShmArchive() = default;

  /** Destructor */
  ~ShmArchive() = default;

  /** Returns a reference to the internal object */
  T_Ar internal_ref(Allocator *alloc) {
    return obj_.internal_ref(alloc);
  }

  /** Returns a reference to the internal object */
  T_Ar internal_ref(Allocator *alloc) const {
    return obj_.internal_ref(alloc);
  }

  /** Shm destructor */
  void shm_destroy(Allocator *alloc) {
    obj_.shm_destroy(alloc);
  }

  /** Copy constructor */
  ShmArchive(const ShmArchive &other) = delete;

  /** Copy assignment operator */
  ShmArchive& operator=(const ShmArchive &other) = delete;

  /** Move constructor */
  ShmArchive(ShmArchive &&other) = delete;

  /** Move assignment operator */
  ShmArchive& operator=(ShmArchive &&other) = delete;
};

}  // namespace hermes_shm::ipc

#endif  // HERMES_DATA_STRUCTURES_SHM_ARCHIVE_H_
