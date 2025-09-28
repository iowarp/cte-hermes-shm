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

#ifndef HSHM_MEMORY_ALLOCATOR_ALLOCATOR_H_
#define HSHM_MEMORY_ALLOCATOR_ALLOCATOR_H_

#include <cstdint>

#include "hermes_shm/constants/macros.h"
#include "hermes_shm/memory/backend/memory_backend.h"
#include "hermes_shm/memory/memory.h"
#include "hermes_shm/thread/thread_model/thread_model.h"
#include "hermes_shm/types/numbers.h"
#include "hermes_shm/util/errors.h"

namespace hshm::ipc {

/**
 * The allocator type.
 * Used to reconstruct allocator from shared memory
 * */
enum class AllocatorType {
  kNullAllocator,
  kStackAllocator,
  kGpuStackAllocator,
  kSliceAllocator,
  kMallocAllocator,
  kFixedPageAllocator,
  kScalablePageAllocator,
  kThreadLocalAllocator,
  kTestAllocator
};

/**
 * The basic shared-memory allocator header.
 * Allocators inherit from this.
 * */
struct AllocatorHeader {
  AllocatorType allocator_type_;
  AllocatorId alloc_id_;
  size_t custom_header_size_;
  hipc::atomic<hshm::size_t> total_alloc_;

  HSHM_CROSS_FUN
  AllocatorHeader() = default;

  HSHM_CROSS_FUN
  void Configure(AllocatorId allocator_id, AllocatorType type,
                 size_t custom_header_size) {
    allocator_type_ = type;
    alloc_id_ = allocator_id;
    custom_header_size_ = custom_header_size;
    total_alloc_ = 0;
  }

  HSHM_INLINE_CROSS_FUN
  void AddSize(hshm::size_t size) {
#ifdef HSHM_ALLOC_TRACK_SIZE
    total_alloc_ += size;
#endif
  }

  HSHM_INLINE_CROSS_FUN
  void SubSize(hshm::size_t size) {
#ifdef HSHM_ALLOC_TRACK_SIZE
    total_alloc_ -= size;
#endif
  }

  HSHM_INLINE_CROSS_FUN
  hshm::size_t GetCurrentlyAllocatedSize() { return total_alloc_.load(); }
};

/** Memory context */
class MemContext {
 public:
  ThreadId tid_ = ThreadId::GetNull();

 public:
  /** Default constructor */
  HSHM_INLINE_CROSS_FUN
  MemContext() = default;

  /** Constructor */
  HSHM_INLINE_CROSS_FUN
  MemContext(const ThreadId &tid) : tid_(tid) {}
};

/** The allocator information struct */
class Allocator {
 public:
  AllocatorType type_;
  AllocatorId id_;
  MemoryBackend backend_;
  char *buffer_;
  size_t buffer_size_;
  char *custom_header_;

 public:
  /** Default constructor */
  HSHM_INLINE_CROSS_FUN
  Allocator() : custom_header_(nullptr) {}

  /** Get the allocator identifier */
  HSHM_INLINE_CROSS_FUN
  AllocatorId &GetId() { return id_; }

  /** Get the allocator identifier (const) */
  HSHM_INLINE_CROSS_FUN
  const AllocatorId &GetId() const { return id_; }

  /**
   * Construct custom header
   */
  template <typename HEADER_T>
  HSHM_INLINE_CROSS_FUN HEADER_T *ConstructHeader(void *buffer) {
    new ((HEADER_T *)buffer) HEADER_T();
    return reinterpret_cast<HEADER_T *>(buffer);
  }

  /**
   * Get the custom header of the shared-memory allocator
   *
   * @return Custom header pointer
   * */
  template <typename HEADER_T>
  HSHM_INLINE_CROSS_FUN HEADER_T *GetCustomHeader() {
    return reinterpret_cast<HEADER_T *>(custom_header_);
  }

  /**
   * Convert a process-independent pointer into a process-specific pointer
   *
   * @param p process-independent pointer
   * @return a process-specific pointer
   * */
  template <typename T, typename PointerT = Pointer>
  HSHM_INLINE_CROSS_FUN T *Convert(const PointerT &p) {
    if (p.IsNull()) {
      return nullptr;
    }
    return reinterpret_cast<T *>(buffer_ + p.off_.load());
  }

  /**
   * Convert a process-specific pointer into a process-independent pointer
   *
   * @param ptr process-specific pointer
   * @return a process-independent pointer
   * */
  template <typename T, typename PointerT = Pointer>
  HSHM_INLINE_CROSS_FUN PointerT Convert(const T *ptr) {
    if (ptr == nullptr) {
      return PointerT::GetNull();
    }
    return PointerT(GetId(), reinterpret_cast<size_t>(ptr) -
                                 reinterpret_cast<size_t>(buffer_));
  }

  /**
   * Determine whether or not this allocator contains a process-specific
   * pointer
   *
   * @param ptr process-specific pointer
   * @return True or false
   * */
  template <typename T = void>
  HSHM_INLINE_CROSS_FUN bool ContainsPtr(const T *ptr) {
    return reinterpret_cast<size_t>(buffer_) <= reinterpret_cast<size_t>(ptr) &&
           reinterpret_cast<size_t>(ptr) <
               reinterpret_cast<size_t>(buffer_) + buffer_size_;
  }

  /** Print */
  HSHM_CROSS_FUN
  void Print() {
    printf("(%s) Allocator: type: %d, id: %d.%d, custom_header: %p\n",
           kCurrentDevice, static_cast<int>(type_), GetId().bits_.major_,
           GetId().bits_.minor_, custom_header_);
  }

  /**====================================
   * Object Constructors
   * ===================================*/

  /**
   * Construct each object in an array of objects.
   *
   * @param ptr the array of objects (potentially archived)
   * @param old_count the original size of the ptr
   * @param new_count the new size of the ptr
   * @param args parameters to construct object of type T
   * @return None
   * */
  template <typename T, typename... Args>
  HSHM_INLINE_CROSS_FUN static void ConstructObjs(T *ptr, size_t old_count,
                                                  size_t new_count,
                                                  Args &&...args) {
    if (ptr == nullptr) {
      return;
    }
    for (size_t i = old_count; i < new_count; ++i) {
      ConstructObj<T>(*(ptr + i), std::forward<Args>(args)...);
    }
  }

  /**
   * Construct an object.
   *
   * @param ptr the object to construct (potentially archived)
   * @param args parameters to construct object of type T
   * @return None
   * */
  template <typename T, typename... Args>
  HSHM_INLINE_CROSS_FUN static void ConstructObj(T &obj, Args &&...args) {
    new (&obj) T(std::forward<Args>(args)...);
  }

  /**
   * Destruct an array of objects
   *
   * @param ptr the object to destruct (potentially archived)
   * @param count the length of the object array
   * @return None
   * */
  template <typename T>
  HSHM_INLINE_CROSS_FUN static void DestructObjs(T *ptr, size_t count) {
    if (ptr == nullptr) {
      return;
    }
    for (size_t i = 0; i < count; ++i) {
      DestructObj<T>(*(ptr + i));
    }
  }

  /**
   * Destruct an object
   *
   * @param ptr the object to destruct (potentially archived)
   * @param count the length of the object array
   * @return None
   * */
  template <typename T>
  HSHM_INLINE_CROSS_FUN static void DestructObj(T &obj) {
    obj.~T();
  }
};

/**
 * The allocator base class.
 * */
template <typename CoreAllocT>
class BaseAllocator : public CoreAllocT {
 public:
  /**====================================
   * Constructors
   * ===================================*/
  /**
   * Create the shared-memory allocator with \a id unique allocator id over
   * the particular slot of a memory backend.
   *
   * The shm_init function is required, but cannot be marked virtual as
   * each allocator has its own arguments to this method. Though each
   * allocator must have "id" as its first argument.
   * */
  template <typename... Args>
  HSHM_CROSS_FUN void shm_init(AllocatorId id, Args... args) {
    CoreAllocT::shm_init(id, std::forward<Args>(args)...);
  }

  /**
   * Deserialize allocator from a buffer.
   * */
  HSHM_CROSS_FUN
  void shm_deserialize(const MemoryBackend &backend) {
    CoreAllocT::shm_deserialize(backend);
  }

  /**====================================
   * Core Allocator API
   * ===================================*/
 public:
  /**
   * Allocate a region of memory of \a size size
   * */
  HSHM_CROSS_FUN
  OffsetPointer AllocateOffset(const MemContext &ctx, size_t size) {
    return CoreAllocT::AllocateOffset(ctx, size);
  }

  /**
   * Allocate a region of memory of \a size size
   * and \a alignment alignment. Assumes that
   * alignment is not 0.
   * */
  HSHM_CROSS_FUN
  OffsetPointer AlignedAllocateOffset(const MemContext &ctx, size_t size,
                                      size_t alignment) {
    return CoreAllocT::AlignedAllocateOffset(ctx, size, alignment);
  }

  /**
   * Reallocate \a pointer to \a new_size new size.
   * Assumes that p is not kNulFullPtr.
   *
   * @return true if p was modified.
   * */
  HSHM_CROSS_FUN
  OffsetPointer ReallocateOffsetNoNullCheck(const MemContext &ctx,
                                            OffsetPointer p, size_t new_size) {
    return CoreAllocT::ReallocateOffsetNoNullCheck(ctx, p, new_size);
  }

  /**
   * Free the memory pointed to by \a ptr Pointer
   * */
  HSHM_CROSS_FUN
  void FreeOffsetNoNullCheck(const MemContext &ctx, OffsetPointer p) {
    CoreAllocT::FreeOffsetNoNullCheck(ctx, p);
  }

  /**
   * Create a thread-local storage segment. This storage
   * is unique even across processes.
   * */
  HSHM_CROSS_FUN
  void CreateTls(MemContext &ctx) { CoreAllocT::CreateTls(ctx); }

  /**
   * Free a thread-local storage segment.
   * */
  HSHM_CROSS_FUN
  void FreeTls(const MemContext &ctx) { CoreAllocT::FreeTls(ctx); }

  /** Get the allocator identifier */
  HSHM_INLINE_CROSS_FUN
  AllocatorId &GetId() { return CoreAllocT::GetId(); }

  /** Get the allocator identifier (const) */
  HSHM_INLINE_CROSS_FUN
  const AllocatorId &GetId() const { return CoreAllocT::GetId(); }

  /**
   * Get the amount of memory that was allocated, but not yet freed.
   * Useful for memory leak checks.
   * */
  HSHM_CROSS_FUN
  size_t GetCurrentlyAllocatedSize() {
    return CoreAllocT::GetCurrentlyAllocatedSize();
  }

  /**====================================
   * SHM Pointer Allocator
   * ===================================*/
 public:
  /**
   * Allocate a region of memory to a specific pointer type
   * */
  template <typename T = void, typename PointerT = Pointer>
  HSHM_INLINE_CROSS_FUN FullPtr<T, PointerT> Allocate(const MemContext &ctx, size_t size) {
    FullPtr<T, PointerT> result;
    result.shm_ = PointerT(GetId(), AllocateOffset(ctx, size).load());
    result.ptr_ = reinterpret_cast<T*>(CoreAllocT::buffer_ + result.shm_.off_.load());
    return result;
  }

  /**
   * Allocate a region of memory to a specific pointer type
   * */
  template <typename T = void, typename PointerT = Pointer>
  HSHM_INLINE_CROSS_FUN FullPtr<T, PointerT> AlignedAllocate(const MemContext &ctx,
                                                 size_t size,
                                                 size_t alignment) {
    FullPtr<T, PointerT> result;
    result.shm_ = PointerT(GetId(),
                    AlignedAllocateOffset(ctx, size, alignment).load());
    result.ptr_ = reinterpret_cast<T*>(CoreAllocT::buffer_ + result.shm_.off_.load());
    return result;
  }

  /**
   * Allocate a region of \a size size and \a alignment
   * alignment. Will fall back to regular Allocate if
   * alignmnet is 0.
   * */
  template <typename T = void, typename PointerT = Pointer>
  HSHM_INLINE_CROSS_FUN FullPtr<T, PointerT> Allocate(const MemContext &ctx, size_t size,
                                          size_t alignment) {
    if (alignment == 0) {
      return Allocate<T, PointerT>(ctx, size);
    } else {
      return AlignedAllocate<T, PointerT>(ctx, size, alignment);
    }
  }

  /**
   * Reallocate \a pointer to \a new_size new size
   * If p is kNulFullPtr, will internally call Allocate.
   *
   * @return the reallocated FullPtr.
   * */
  template <typename T = void, typename PointerT = Pointer>
  HSHM_INLINE_CROSS_FUN FullPtr<T, PointerT> Reallocate(const MemContext &ctx, const FullPtr<T, PointerT> &p,
                                        size_t new_size) {
    if (p.IsNull()) {
      return Allocate<T, PointerT>(ctx, new_size);
    }
    auto new_off =
        ReallocateOffsetNoNullCheck(ctx, p.shm_.ToOffsetPointer(), new_size);
    FullPtr<T, PointerT> result;
    result.shm_ = PointerT(GetId(), new_off.load());
    result.ptr_ = reinterpret_cast<T*>(CoreAllocT::buffer_ + result.shm_.off_.load());
    return result;
  }

  /**
   * Free the memory pointed to by \a p Pointer
   * */
  template <typename T = void, typename PointerT = Pointer>
  HSHM_INLINE_CROSS_FUN void Free(const MemContext &ctx, const FullPtr<T, PointerT> &p) {
    if (p.IsNull()) {
      HSHM_THROW_ERROR(INVALID_FREE);
    }
    FreeOffsetNoNullCheck(ctx, OffsetPointer(p.shm_.off_.load()));
  }





  /**====================================
   * Private Object Allocators
   * ===================================*/

  /**
   * Allocate an array of objects (but don't construct).
   *
   * @return A FullPtr to the allocated memory
   * */
  template <typename T, typename PointerT = Pointer>
  HSHM_INLINE_CROSS_FUN FullPtr<T, PointerT> AllocateObjs(const MemContext &ctx, size_t count) {
    return Allocate<T, PointerT>(ctx, count * sizeof(T));
  }

  /** Allocate + construct an array of objects */
  template <typename T, typename... Args>
  HSHM_INLINE_CROSS_FUN FullPtr<T> NewObjs(const MemContext &ctx, size_t count,
                                   Args &&...args) {
    auto alloc_result = AllocateObjs<T, Pointer>(ctx, count);
    ConstructObjs<T>(alloc_result.ptr_, 0, count, std::forward<Args>(args)...);
    return alloc_result;
  }

  /** Allocate + construct a single object */
  template <typename T, typename... Args>
  HSHM_INLINE_CROSS_FUN FullPtr<T> NewObj(const MemContext &ctx, Args &&...args) {
    return NewObjs<T>(ctx, 1, std::forward<Args>(args)...);
  }

  /**
   * Reallocate a pointer of objects to a new size.
   *
   * @param p FullPtr to reallocate (input & output)
   * @param new_count the new number of objects
   *
   * @return A FullPtr to the reallocated objects
   * */
  template <typename T, typename PointerT = Pointer>
  HSHM_INLINE_CROSS_FUN FullPtr<T, PointerT> ReallocateObjs(const MemContext &ctx, 
                                                             FullPtr<T, PointerT> &p,
                                                             size_t new_count) {
    FullPtr<void, PointerT> old_full_ptr(reinterpret_cast<void*>(p.ptr_), p.shm_);
    auto new_full_ptr = Reallocate<void, PointerT>(ctx, old_full_ptr, new_count * sizeof(T));
    p.shm_ = new_full_ptr.shm_;
    p.ptr_ = reinterpret_cast<T*>(new_full_ptr.ptr_);
    return p;
  }

  /**
   * Free + destruct objects
   * */
  template <typename T, typename PointerT = Pointer>
  HSHM_INLINE_CROSS_FUN void DelObjs(const MemContext &ctx, 
                                     FullPtr<T, PointerT> &p,
                                     size_t count) {
    DestructObjs<T>(p.ptr_, count);
    FullPtr<void, PointerT> void_ptr(reinterpret_cast<void*>(p.ptr_), p.shm_);
    Free<void, PointerT>(ctx, void_ptr);
  }

  /**
   * Free + destruct an object
   * */
  template <typename T, typename PointerT = Pointer>
  HSHM_INLINE_CROSS_FUN void DelObj(const MemContext &ctx, 
                                    FullPtr<T, PointerT> &p) {
    DelObjs<T, PointerT>(ctx, p, 1);
  }


  /**====================================
   * Object Constructors
   * ===================================*/

  /**
   * Construct each object in an array of objects.
   *
   * @param ptr the array of objects (potentially archived)
   * @param old_count the original size of the ptr
   * @param new_count the new size of the ptr
   * @param args parameters to construct object of type T
   * @return None
   * */
  template <typename T, typename... Args>
  HSHM_INLINE_CROSS_FUN static void ConstructObjs(T *ptr, size_t old_count,
                                                  size_t new_count,
                                                  Args &&...args) {
    CoreAllocT::template ConstructObjs<T>(ptr, old_count, new_count,
                                          std::forward<Args>(args)...);
  }

  /**
   * Construct an object.
   *
   * @param ptr the object to construct (potentially archived)
   * @param args parameters to construct object of type T
   * @return None
   * */
  template <typename T, typename... Args>
  HSHM_INLINE_CROSS_FUN static void ConstructObj(T &obj, Args &&...args) {
    CoreAllocT::template ConstructObj<T>(obj, std::forward<Args>(args)...);
  }

  /**
   * Destruct an array of objects
   *
   * @param ptr the object to destruct (potentially archived)
   * @param count the length of the object array
   * @return None
   * */
  template <typename T>
  HSHM_INLINE_CROSS_FUN static void DestructObjs(T *ptr, size_t count) {
    CoreAllocT::template DestructObjs<T>(ptr, count);
  }

  /**
   * Destruct an object
   *
   * @param ptr the object to destruct (potentially archived)
   * @param count the length of the object array
   * @return None
   * */
  template <typename T>
  HSHM_INLINE_CROSS_FUN static void DestructObj(T &obj) {
    CoreAllocT::template DestructObj<T>(obj);
  }

  /**====================================
   * Helpers
   * ===================================*/

  /**
   * Get the custom header of the shared-memory allocator
   *
   * @return Custom header pointer
   * */
  template <typename HEADER_T>
  HSHM_INLINE_CROSS_FUN HEADER_T *GetCustomHeader() {
    return CoreAllocT::template GetCustomHeader<HEADER_T>();
  }

  /**
   * Convert a process-independent pointer into a process-specific pointer
   *
   * @param p process-independent pointer
   * @return a process-specific pointer
   * */
  template <typename T, typename PointerT = Pointer>
  HSHM_INLINE_CROSS_FUN T *Convert(const PointerT &p) {
    return CoreAllocT::template Convert<T, PointerT>(p);
  }

  /**
   * Convert a process-specific pointer into a process-independent pointer
   *
   * @param ptr process-specific pointer
   * @return a process-independent pointer
   * */
  template <typename T, typename PointerT = Pointer>
  HSHM_INLINE_CROSS_FUN PointerT Convert(const T *ptr) {
    return CoreAllocT::template Convert<T, PointerT>(ptr);
  }

  /**
   * Determine whether or not this allocator contains a process-specific
   * pointer
   *
   * @param ptr process-specific pointer
   * @return True or false
   * */
  template <typename T = void>
  HSHM_INLINE_CROSS_FUN bool ContainsPtr(const T *ptr) {
    return CoreAllocT::template ContainsPtr<T>(ptr);
  }

  /** Print */
  HSHM_CROSS_FUN
  void Print() { CoreAllocT::Print(); }
};

/** Get the full allocator within core allocator */
#define HSHM_ALLOCATOR(ALLOC_NAME)                    \
 public:                                              \
  typedef hipc::BaseAllocator<ALLOC_NAME> BaseAllocT; \
  HSHM_INLINE_CROSS_FUN                               \
  BaseAllocT *GetAllocator() { return (BaseAllocT *)(this); }

/** Demonstration allocator */
class _NullAllocator : public Allocator {
 public:
  /**====================================
   * Constructors
   * ===================================*/
  /**
   * Create the shared-memory allocator with \a id unique allocator id over
   * the particular slot of a memory backend.
   *
   * The shm_init function is required, but cannot be marked virtual as
   * each allocator has its own arguments to this method. Though each
   * allocator must have "id" as its first argument.
   * */
  HSHM_CROSS_FUN
  void shm_init(AllocatorId id, size_t custom_header_size,
                MemoryBackend backend) {
    type_ = AllocatorType::kNullAllocator;
    id_ = id;
    if (backend.IsCopyGpu()) {
      buffer_ = backend.accel_data_;
      buffer_size_ = backend.accel_data_size_;
    } else {
      buffer_ = backend.data_;
      buffer_size_ = backend.data_size_;
    }
  }

  /**
   * Deserialize allocator from a buffer.
   * */
  HSHM_CROSS_FUN
  void shm_deserialize(char *buffer, size_t buffer_size) {}

  /**====================================
   * Core Allocator API
   * ===================================*/
 public:
  /**
   * Allocate a region of memory of \a size size
   * */
  HSHM_CROSS_FUN
  OffsetPointer AllocateOffset(const MemContext &ctx, size_t size) {
    return OffsetPointer::GetNull();
  }

  /**
   * Allocate a region of memory of \a size size
   * and \a alignment alignment. Assumes that
   * alignment is not 0.
   * */
  HSHM_CROSS_FUN
  OffsetPointer AlignedAllocateOffset(const MemContext &ctx, size_t size,
                                      size_t alignment) {
    return OffsetPointer::GetNull();
  }

  /**
   * Reallocate \a pointer to \a new_size new size.
   * Assumes that p is not kNulFullPtr.
   *
   * @return true if p was modified.
   * */
  HSHM_CROSS_FUN
  OffsetPointer ReallocateOffsetNoNullCheck(const MemContext &ctx,
                                            OffsetPointer p, size_t new_size) {
    return p;
  }

  /**
   * Free the memory pointed to by \a ptr Pointer
   * */
  HSHM_CROSS_FUN
  void FreeOffsetNoNullCheck(const MemContext &ctx, OffsetPointer p) {}

  /**
   * Create a globally-unique thread ID
   * */
  HSHM_CROSS_FUN
  void CreateTls(MemContext &ctx) {}

  /**
   * Free the memory pointed to by \a ptr Pointer
   * */
  HSHM_CROSS_FUN
  void FreeTls(const MemContext &ctx) {}

  /**
   * Get the amount of memory that was allocated, but not yet freed.
   * Useful for memory leak checks.
   * */
  HSHM_CROSS_FUN
  size_t GetCurrentlyAllocatedSize() { return 0; }
};
typedef BaseAllocator<_NullAllocator> NullAllocator;

/**
 * Allocator with thread-local storage identifier
 * */
template <typename AllocT>
struct CtxAllocator {
  MemContext ctx_;
  AllocT *alloc_;

  /** Default constructor */
  HSHM_INLINE_CROSS_FUN
  CtxAllocator() = default;

  /** Allocator-only constructor */
  HSHM_INLINE_CROSS_FUN
  CtxAllocator(AllocT *alloc) : alloc_(alloc), ctx_() {}

  /** Allocator and thread identifier constructor */
  HSHM_INLINE_CROSS_FUN
  CtxAllocator(AllocT *alloc, const ThreadId &tid) : alloc_(alloc), ctx_(tid) {}

  /** Allocator and thread identifier constructor */
  HSHM_INLINE_CROSS_FUN
  CtxAllocator(const ThreadId &tid, AllocT *alloc) : alloc_(alloc), ctx_(tid) {}

  /** Allocator and ctx constructor */
  HSHM_INLINE_CROSS_FUN
  CtxAllocator(const MemContext &ctx, AllocT *alloc)
      : alloc_(alloc), ctx_(ctx) {}

  /** ctx and Allocator constructor */
  HSHM_INLINE_CROSS_FUN
  CtxAllocator(AllocT *alloc, const MemContext &ctx)
      : alloc_(alloc), ctx_(ctx) {}

  /** Arrow operator */
  HSHM_INLINE_CROSS_FUN
  AllocT *operator->() { return alloc_; }

  /** Arrow operator (const) */
  HSHM_INLINE_CROSS_FUN
  AllocT *operator->() const { return alloc_; }

  /** Star operator */
  HSHM_INLINE_CROSS_FUN
  AllocT *operator*() { return alloc_; }

  /** Star operator (const) */
  HSHM_INLINE_CROSS_FUN
  AllocT *operator*() const { return alloc_; }

  /** Equality operator */
  HSHM_INLINE_CROSS_FUN
  bool operator==(const CtxAllocator &rhs) const {
    return alloc_ == rhs.alloc_;
  }

  /** Inequality operator */
  HSHM_INLINE_CROSS_FUN
  bool operator!=(const CtxAllocator &rhs) const {
    return alloc_ != rhs.alloc_;
  }
};

/**
 * Scoped Allocator (thread-local)
 * */
template <typename AllocT>
class ScopedTlsAllocator {
 public:
  CtxAllocator<AllocT> alloc_;

 public:
  HSHM_INLINE_CROSS_FUN
  ScopedTlsAllocator(const MemContext &ctx, AllocT *alloc)
      : alloc_(ctx, alloc) {
    alloc_->CreateTls(alloc_.ctx_);
  }

  HSHM_INLINE_CROSS_FUN
  ScopedTlsAllocator(const CtxAllocator<AllocT> &alloc) : alloc_(alloc) {
    alloc_->CreateTls(alloc_.ctx_);
  }

  HSHM_INLINE_CROSS_FUN
  ~ScopedTlsAllocator() { alloc_->FreeTls(alloc_.ctx_); }

  /** Arrow operator */
  HSHM_INLINE_CROSS_FUN
  CtxAllocator<AllocT> &operator->() { return alloc_; }

  /** Arrow operator (const) */
  HSHM_INLINE_CROSS_FUN
  const CtxAllocator<AllocT> &operator->() const { return alloc_; }

  /** Star operator */
  HSHM_INLINE_CROSS_FUN
  CtxAllocator<AllocT> &operator*() { return alloc_; }

  /** Star operator (const) */
  HSHM_INLINE_CROSS_FUN
  const CtxAllocator<AllocT> &operator*() const { return alloc_; }
};

/** Thread-local storage manager */
template <typename AllocT>
class TlsAllocatorInfo : public thread::ThreadLocalData {
 public:
  AllocT *alloc_;
  ThreadId tid_;

 public:
  HSHM_CROSS_FUN
  TlsAllocatorInfo() : alloc_(nullptr), tid_(ThreadId::GetNull()) {}

  HSHM_CROSS_FUN
  void destroy() { alloc_->FreeTls(tid_); }
};

}  // namespace hshm::ipc

#endif  // HSHM_MEMORY_ALLOCATOR_ALLOCATOR_H_
