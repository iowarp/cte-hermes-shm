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

#ifndef HSHM_MEMORY_H
#define HSHM_MEMORY_H

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "hermes_shm/constants/macros.h"
#include "hermes_shm/data_structures/ipc/chararr.h"
#include "hermes_shm/memory/memory.h"

namespace hshm::ipc {

enum class MemoryBackendType {
  kPosixShmMmap,
  kMallocBackend,
  kArrayBackend,
  kPosixMmap,
  kGpuMalloc,
  kGpuShmMmap,
};

/** ID for memory backend */
class MemoryBackendId {
 public:
  u32 id_;

  HSHM_CROSS_FUN
  MemoryBackendId() = default;

  HSHM_CROSS_FUN
  MemoryBackendId(u32 id) : id_(id) {}

  HSHM_CROSS_FUN
  MemoryBackendId(const MemoryBackendId &other) : id_(other.id_) {}

  HSHM_CROSS_FUN
  MemoryBackendId(MemoryBackendId &&other) noexcept : id_(other.id_) {}

  HSHM_CROSS_FUN
  MemoryBackendId &operator=(const MemoryBackendId &other) {
    id_ = other.id_;
    return *this;
  }

  HSHM_CROSS_FUN
  MemoryBackendId &operator=(MemoryBackendId &&other) noexcept {
    id_ = other.id_;
    return *this;
  }

  HSHM_CROSS_FUN
  static MemoryBackendId GetRoot() { return {0}; }

  HSHM_CROSS_FUN
  static MemoryBackendId Get(u32 id) { return {id + 1}; }

  HSHM_CROSS_FUN
  bool operator==(const MemoryBackendId &other) const {
    return id_ == other.id_;
  }

  HSHM_CROSS_FUN
  bool operator!=(const MemoryBackendId &other) const {
    return id_ != other.id_;
  }
};
typedef MemoryBackendId memory_backend_id_t;

struct MemoryBackendHeader {
  union {
    size_t data_size_;  // For CPU-only backends
    size_t md_size_;    // For CPU+GPU backends
  };
  MemoryBackendType type_;
  MemoryBackendId id_;
  bitfield64_t flags_;
  size_t accel_data_size_;
  int accel_id_;

  HSHM_CROSS_FUN void Print() const {
    printf("(%s) MemoryBackendHeader: type: %d, id: %d, data_size: %lu\n",
           kCurrentDevice, static_cast<int>(type_), id_.id_,
           (long unsigned)data_size_);
  }
};

#define MEMORY_BACKEND_INITIALIZED BIT_OPT(u64, 0)
#define MEMORY_BACKEND_OWNED BIT_OPT(u64, 1)
#define MEMORY_BACKEND_COPY_GPU BIT_OPT(u64, 2)
#define MEMORY_BACKEND_MIRROR_GPU BIT_OPT(u64, 3)
#define MEMORY_BACKEND_HAS_ALLOC BIT_OPT(u64, 4)
#define MEMORY_BACKEND_HAS_GPU_ALLOC BIT_OPT(u64, 5)
#define MEMORY_BACKEND_IS_SCANNED BIT_OPT(u64, 6)

class UrlMemoryBackend {};

class MemoryBackend {
 public:
  MemoryBackendHeader *header_;
  union {
    char *data_; /** For CPU-only backends */
    char *md_;   /** For CPU+GPU backends */
  };
  union {
    size_t data_size_; /** For CPU-only backends */
    size_t md_size_;   /** For CPU+GPU backends */
  };
  bitfield64_t flags_;
  char *accel_data_;
  size_t accel_data_size_;
  int accel_id_;

 public:
  HSHM_CROSS_FUN
  MemoryBackend() : header_(nullptr), data_(nullptr) {}

  ~MemoryBackend() = default;

  HSHM_CROSS_FUN
  MemoryBackend Shift(size_t offset) {
    MemoryBackend backend;
    backend.header_ = header_;
    backend.md_ = md_ + offset;
    backend.md_size_ = md_size_ - offset;
    backend.accel_data_ = accel_data_;
    backend.accel_data_size_ = accel_data_size_;
    backend.Disown();
    backend.SetInitialized();
    return backend;
  }

  /** Mark data as valid */
  HSHM_CROSS_FUN
  void SetInitialized() { flags_.SetBits(MEMORY_BACKEND_INITIALIZED); }

  /** Check if data is valid */
  HSHM_CROSS_FUN
  bool IsInitialized() { return flags_.Any(MEMORY_BACKEND_INITIALIZED); }

  /** Mark data as invalid */
  HSHM_CROSS_FUN
  void UnsetInitialized() { flags_.UnsetBits(MEMORY_BACKEND_INITIALIZED); }

  /** Mark data for GPU copy */
  HSHM_CROSS_FUN
  void SetCopyGpu() { flags_.SetBits(MEMORY_BACKEND_COPY_GPU); }

  /** Check if data is marked for GPU copy */
  HSHM_CROSS_FUN
  bool IsCopyGpu() { return flags_.Any(MEMORY_BACKEND_COPY_GPU); }

  /** Unmark data for GPU copy */
  HSHM_CROSS_FUN
  void UnsetCopyGpu() { flags_.UnsetBits(MEMORY_BACKEND_COPY_GPU); }

  /** Mark data for GPU mirror */
  HSHM_CROSS_FUN
  void SetMirrorGpu() { flags_.SetBits(MEMORY_BACKEND_MIRROR_GPU); }

  /** Check if data is marked for GPU mirror */
  HSHM_CROSS_FUN
  bool IsMirrorGpu() { return flags_.Any(MEMORY_BACKEND_MIRROR_GPU); }

  /** Unmark data for GPU mirror */
  HSHM_CROSS_FUN
  void UnsetMirrorGpu() { flags_.UnsetBits(MEMORY_BACKEND_MIRROR_GPU); }

  /** Mark data as having an allocation */
  HSHM_CROSS_FUN
  void SetHasAlloc() { header_->flags_.SetBits(MEMORY_BACKEND_HAS_ALLOC); }

  /** Check if data has an allocation */
  HSHM_CROSS_FUN
  bool IsHasAlloc() { return header_->flags_.Any(MEMORY_BACKEND_HAS_ALLOC); }

  /** Unmark data as having an allocation */
  HSHM_CROSS_FUN
  void UnsetHasAlloc() { header_->flags_.UnsetBits(MEMORY_BACKEND_HAS_ALLOC); }

  /** Mark data as having a GPU allocation */
  HSHM_CROSS_FUN
  void SetHasGpuAlloc() {
    header_->flags_.SetBits(MEMORY_BACKEND_HAS_GPU_ALLOC);
  }

  /** Check if data has a GPU allocation */
  HSHM_CROSS_FUN
  bool IsHasGpuAlloc() {
    return header_->flags_.Any(MEMORY_BACKEND_HAS_GPU_ALLOC);
  }

  /** Unmark data as having a GPU allocation */
  HSHM_CROSS_FUN
  void UnsetHasGpuAlloc() {
    header_->flags_.UnsetBits(MEMORY_BACKEND_HAS_GPU_ALLOC);
  }

  /** Mark data as scanned */
  HSHM_CROSS_FUN
  void SetScanned() { flags_.SetBits(MEMORY_BACKEND_IS_SCANNED); }

  /** Check if data is scanned */
  HSHM_CROSS_FUN
  bool IsScanned() { return flags_.Any(MEMORY_BACKEND_IS_SCANNED); }

  /** Unmark data as scanned */
  HSHM_CROSS_FUN
  void UnsetScanned() { flags_.UnsetBits(MEMORY_BACKEND_IS_SCANNED); }

  /** This is the process which destroys the backend */
  HSHM_CROSS_FUN
  void Own() { flags_.SetBits(MEMORY_BACKEND_OWNED); }

  /** This is owned */
  HSHM_CROSS_FUN
  bool IsOwned() { return flags_.Any(MEMORY_BACKEND_OWNED); }

  /** This is not the process which destroys the backend */
  HSHM_CROSS_FUN
  void Disown() { flags_.UnsetBits(MEMORY_BACKEND_OWNED); }

  /** Get the ID of this backend */
  HSHM_CROSS_FUN
  MemoryBackendId &GetId() { return header_->id_; }

  /** Get the ID of this backend */
  HSHM_CROSS_FUN
  const MemoryBackendId &GetId() const { return header_->id_; }

  HSHM_CROSS_FUN
  void Print() const {
    header_->Print();
    printf("(%s) MemoryBackend: data: %p, data_size: %lu\n", kCurrentDevice,
           data_, (long unsigned)data_size_);
  }

  /// Each allocator must define its own shm_init.
  // virtual bool shm_init(size_t size, ...) = 0;
  // virtual bool shm_deserialize(const hshm::chararr &url) = 0;
  // virtual void shm_detach() = 0;
  // virtual void shm_destroy() = 0;
};

}  // namespace hshm::ipc

#endif  // HSHM_MEMORY_H
