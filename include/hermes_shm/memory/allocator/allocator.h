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

#ifndef HERMES_MEMORY_ALLOCATOR_ALLOCATOR_H_
#define HERMES_MEMORY_ALLOCATOR_ALLOCATOR_H_

#include <cstdint>
#include <hermes_shm/memory/memory.h>
#include <hermes_shm/util/errors.h>

namespace hshm::ipc {

/**
 * The allocator type.
 * Used to reconstruct allocator from shared memory
 * */
enum class AllocatorType {
  kStackAllocator,
  kMallocAllocator,
  kFixedPageAllocator,
  kScalablePageAllocator,
};

/**
 * The basic shared-memory allocator header.
 * Allocators inherit from this.
 * */
struct AllocatorHeader {
  AllocatorType allocator_type_;
  AllocatorId allocator_id_;
  size_t custom_header_size_;

  HSHM_CROSS_FUN
  AllocatorHeader() = default;

  HSHM_CROSS_FUN
  void Configure(AllocatorId allocator_id,
                 AllocatorType type,
                 size_t custom_header_size) {
    allocator_type_ = type;
    allocator_id_ = allocator_id;
    custom_header_size_ = custom_header_size;
  }
};

/**
 * The allocator base class.
 * */
class Allocator {
 public:
  AllocatorType type_;
  AllocatorId id_;
  char *buffer_;
  size_t buffer_size_;
  char *custom_header_;

 public:
  /**====================================
  * Constructors
  * ===================================*/

  /**
   * Constructor
   * */
  HSHM_CROSS_FUN Allocator() : custom_header_(nullptr) {}

  /**
   * Destructor
   * */
  HSHM_CROSS_FUN virtual ~Allocator() = default;

  /**
   * Create the shared-memory allocator with \a id unique allocator id over
   * the particular slot of a memory backend.
   *
   * The shm_init function is required, but cannot be marked virtual as
   * each allocator has its own arguments to this method. Though each
   * allocator must have "id" as its first argument.
   * */
  // virtual void shm_init(AllocatorId id, Args ...args) = 0;

  /**
   * Deserialize allocator from a buffer.
   * */
  HSHM_CROSS_FUN
  virtual void shm_deserialize(char *buffer,
                               size_t buffer_size) = 0;

  /**====================================
  * Core Allocator API
  * ===================================*/
 public:
  /**
   * Allocate a region of memory of \a size size
   * */
  HSHM_CROSS_FUN
  virtual OffsetPointer AllocateOffset(
      const ThreadId &tid, size_t size) = 0;

  /**
   * Allocate a region of memory of \a size size
   * and \a alignment alignment. Assumes that
   * alignment is not 0.
   * */
  HSHM_CROSS_FUN
  virtual OffsetPointer AlignedAllocateOffset(
      const ThreadId &tid,
      size_t size,
      size_t alignment) = 0;

  /**
   * Reallocate \a pointer to \a new_size new size.
   * Assumes that p is not kNullPointer.
   *
   * @return true if p was modified.
   * */
  HSHM_CROSS_FUN
  virtual OffsetPointer ReallocateOffsetNoNullCheck(
      const ThreadId &tid,
      OffsetPointer p,
      size_t new_size) = 0;


  /**
   * Free the memory pointed to by \a ptr Pointer
   * */
  HSHM_CROSS_FUN
  virtual void FreeOffsetNoNullCheck(
      const ThreadId &tid,
      OffsetPointer p) = 0;

  /** Get the allocator identifier */
  HSHM_CROSS_FUN
  AllocatorId& GetId()  {
    return id_;
  }

  /** Get the allocator identifier (const) */
  HSHM_CROSS_FUN
  const AllocatorId& GetId() const {
    return id_;
  }

  /**
   * Get the amount of memory that was allocated, but not yet freed.
   * Useful for memory leak checks.
   * */
  HSHM_CROSS_FUN
  virtual size_t GetCurrentlyAllocatedSize() = 0;

  /**====================================
  * SHM Pointer Allocator
  * ===================================*/
 public:
  /**
   * Allocate a region of memory to a specific pointer type
   * */
  template<typename PointerT = Pointer>
  HSHM_INLINE_CROSS_FUN
  PointerT Allocate(const ThreadId &tid, size_t size) {
    return PointerT(GetId(), AllocateOffset(tid, size).load());
  }

  /**
   * Allocate a region of memory to a specific pointer type
   * */
  template<typename PointerT = Pointer>
  HSHM_INLINE_CROSS_FUN
  PointerT AlignedAllocate(const ThreadId &tid,
                           size_t size, size_t alignment) {
    return PointerT(
        GetId(), AlignedAllocateOffset(tid, size, alignment).load());
  }

  /**
   * Allocate a region of \a size size and \a alignment
   * alignment. Will fall back to regular Allocate if
   * alignmnet is 0.
   * */
  template<typename PointerT = Pointer>
  HSHM_INLINE_CROSS_FUN
  PointerT Allocate(const ThreadId &tid,
                    size_t size, size_t alignment) {
    if (alignment == 0) {
      return Allocate<PointerT>(tid, size);
    } else {
      return AlignedAllocate<PointerT>(tid, size, alignment);
    }
  }

  /**
   * Reallocate \a pointer to \a new_size new size
   * If p is kNullPointer, will internally call Allocate.
   *
   * @return true if p was modified.
   * */
  template<typename PointerT = Pointer>
  HSHM_INLINE_CROSS_FUN
  bool Reallocate(const ThreadId &tid,
                  PointerT &p, size_t new_size) {
    if (p.IsNull()) {
      p = Allocate<PointerT>(tid, new_size);
      return true;
    }
    auto new_p = ReallocateOffsetNoNullCheck(
        tid, p.ToOffsetPointer(), new_size);
    bool ret = new_p == p.ToOffsetPointer();
    p.off_ = new_p.load();
    return ret;
  }

  /**
   * Free the memory pointed to by \a p Pointer
   * */
  template<typename PointerT = Pointer>
  HSHM_INLINE_CROSS_FUN
  void Free(const ThreadId &tid, PointerT &p) {
    if (p.IsNull()) {
      HERMES_THROW_ERROR(INVALID_FREE);
    }
    FreeOffsetNoNullCheck(tid, OffsetPointer(p.off_.load()));
  }

  /**====================================
  * Private Pointer Allocators
  * ===================================*/

  /**
 * Allocate a pointer of \a size size and return \a p process-independent
 * pointer and a process-specific pointer.
 * */
  template<typename T, typename PointerT = Pointer>
  HSHM_INLINE_CROSS_FUN
  T* AllocatePtr(
      const ThreadId &tid,
      size_t size,
      PointerT &p,
      size_t alignment = 0) {
    p = Allocate<PointerT>(tid, size, alignment);
    if (p.IsNull()) { return nullptr; }
    return reinterpret_cast<T*>(buffer_ + p.off_.load());
  }

  /**
   * Allocate a pointer of \a size size
   * */
  template<typename T, typename PointerT = Pointer>
  HSHM_INLINE_CROSS_FUN T* AllocatePtr(
      const ThreadId &tid,
      size_t size, size_t alignment = 0) {
    PointerT p;
    return AllocatePtr<T, PointerT>(tid, size, p, alignment);
  }

  /**
 * Allocate a pointer of \a size size and return \a p process-independent
 * pointer and a process-specific pointer.
 * */
  template<typename T, typename PointerT = Pointer>
  HSHM_INLINE_CROSS_FUN
  T* ClearAllocatePtr(const ThreadId &tid,
                      size_t size,
                      PointerT &p,
                      size_t alignment = 0) {
    p = Allocate<PointerT>(tid, size, alignment);
    if (p.IsNull()) { return nullptr; }
    auto ptr = reinterpret_cast<T*>(buffer_ + p.off_.load());
    if (ptr) {
      memset(ptr, 0, size);
    }
    return ptr;
  }

  /**
   * Allocate a pointer of \a size size
   * */
  template<typename T, typename PointerT = Pointer>
  HSHM_INLINE_CROSS_FUN T* ClearAllocatePtr(
      const ThreadId &tid,
      size_t size,
      size_t alignment = 0) {
    PointerT p;
    return ClearAllocatePtr<T, PointerT>(tid, size, p, alignment);
  }

  /**
 * Reallocate a pointer to a new size
 *
 * @param p process-independent pointer (input & output)
 * @param new_size the new size to allocate
 * @param modified whether or not p was modified (output)
 * @return A process-specific pointer
 * */
  template<typename T, typename PointerT = Pointer>
  HSHM_INLINE_CROSS_FUN
  T* ReallocatePtr(const ThreadId &tid,
                   PointerT &p,
                   size_t new_size,
                   bool &modified) {
    modified = Reallocate<PointerT>(tid, p, new_size);
    return Convert<T>(p);
  }

  /**
   * Reallocate a pointer to a new size
   *
   * @param p process-independent pointer (input & output)
   * @param new_size the new size to allocate
   * @return A process-specific pointer
   * */
  template<typename T, typename PointerT = Pointer>
  HSHM_INLINE_CROSS_FUN T* ReallocatePtr(
      const ThreadId &tid,
      PointerT &p, size_t new_size) {
    Reallocate<PointerT>(tid, p, new_size);
    return Convert<T>(p);
  }

  /**
   * Reallocate a pointer to a new size
   *
   * @param old_ptr process-specific pointer to reallocate
   * @param new_size the new size to allocate
   * @return A process-specific pointer
   * */
  template<typename T>
  HSHM_INLINE_CROSS_FUN
  T* ReallocatePtr(
      const ThreadId &tid,
      T *old_ptr, size_t new_size) {
    OffsetPointer p = Convert<T, OffsetPointer>(old_ptr);
    return ReallocatePtr<T, OffsetPointer>(tid, p, new_size);
  }

  /**
   * Free the memory pointed to by \a ptr Pointer
   * */
  template<typename T = void>
  HSHM_INLINE_CROSS_FUN
  void FreePtr(const ThreadId &tid, T *ptr) {
    if (ptr == nullptr) {
      HERMES_THROW_ERROR(INVALID_FREE);
    }
    FreeOffsetNoNullCheck(tid, Convert<T, OffsetPointer>(ptr));
  }

  /**====================================
  * Local Pointer Allocators
  * ===================================*/

  /**
   * Allocate a pointer of \a size size and return \a p process-independent
   * pointer and a process-specific pointer.
   * */
  template<typename T, typename PointerT = Pointer>
  HSHM_INLINE_CROSS_FUN
  LPointer<T, PointerT>
  AllocateLocalPtr(const ThreadId &tid,
                   size_t size, size_t alignment = 0) {
    LPointer<T, PointerT> p;
    p.ptr_ = AllocatePtr<T, PointerT>(tid, size, p.shm_, alignment);
    return p;
  }

  /**
   * Allocate a pointer of \a size size
   * */
  template<typename T, typename PointerT = Pointer>
  HSHM_INLINE_CROSS_FUN
  LPointer<T, PointerT>
  ClearAllocateLocalPtr(const ThreadId &tid,
                        size_t size, size_t alignment = 0) {
    LPointer<T, PointerT> p;
    p.ptr_ = ClearAllocatePtr<T, PointerT>(tid, size, p.shm_, alignment);
    return p;
  }

  /**
   * Reallocate a pointer to a new size
   *
   * @param p process-independent pointer (input & output)
   * @param new_size the new size to allocate
   * @return A process-specific pointer
   * */
  template<typename T, typename PointerT = Pointer>
  HSHM_INLINE_CROSS_FUN
  LPointer<T, PointerT>
  ReallocateLocalPtr(const ThreadId &tid,
                     LPointer<T, PointerT> &p, size_t new_size) {
    Reallocate<PointerT>(tid, p.shm_, new_size);
    p.ptr_ = Convert<T>(p.shm_);
    return p;
  }

  /**
   * Free the memory pointed to by \a ptr Pointer
   * */
  template<typename T = void, typename PointerT = Pointer>
  HSHM_INLINE_CROSS_FUN
  void FreeLocalPtr(const ThreadId &tid,
                    LPointer<T, PointerT> &ptr) {
    if (ptr.ptr_ == nullptr) {
      HERMES_THROW_ERROR(INVALID_FREE);
    }
    FreeOffsetNoNullCheck(tid, ptr.shm_.ToOffsetPointer());
  }

  /**====================================
  * SHM Array Allocators
  * ===================================*/

  /**
   * Allocate a pointer of \a size size and return \a p process-independent
   * pointer and its size.
   * */
  template<typename T, typename PointerT = Pointer>
  HSHM_INLINE_CROSS_FUN
  Array<PointerT>
  AllocateArray(const ThreadId &tid,
                size_t size, size_t alignment = 0) {
    Array<PointerT> p;
    p.shm_ = Allocate<PointerT>(tid, size, alignment);
    p.size_ = size;
    return p;
  }

  /**
   * Allocate a pointer of \a size size
   * */
  template<typename T, typename PointerT = Pointer>
  HSHM_INLINE_CROSS_FUN
  Array<PointerT> ClearAllocateArray(
      const ThreadId &tid,
      size_t size, size_t alignment = 0) {
    Array<PointerT> p;
    ClearAllocatePtr<T, PointerT>(tid, size, p.shm_, alignment);
    p.size_ = size;
    return p;
  }

  /**
   * Reallocate a pointer to a new size
   *
   * @param p process-independent pointer (input & output)
   * @param new_size the new size to allocate
   * @return A process-specific pointer
   * */
  template<typename T, typename PointerT = Pointer>
  HSHM_INLINE_CROSS_FUN
  Array<PointerT>
  ReallocateArray(const ThreadId &tid,
                  Array<PointerT> &p, size_t new_size) {
    Reallocate<PointerT>(tid, p.shm_, new_size);
    p.size_ = new_size;
    return p;
  }

  /**
   * Free the memory pointed to by \a ptr Pointer
   * */
  template<typename T = void, typename PointerT = Pointer>
  HSHM_INLINE_CROSS_FUN
  void FreeArray(const ThreadId &tid,
                 Array<PointerT> &ptr) {
    if (ptr.shm_.IsNull()) {
      HERMES_THROW_ERROR(INVALID_FREE);
    }
    FreeOffsetNoNullCheck(tid, ptr.shm_.ToOffsetPointer());
  }

  /**====================================
  * Local Array Allocators
  * ===================================*/

  /**
   * Allocate a pointer of \a size size and return \a p process-independent
   * pointer, a process-specific pointer, and its size.
   * */
  template<typename T, typename PointerT = Pointer>
  HSHM_INLINE_CROSS_FUN
  LArray<T, PointerT>
  AllocateLocalArray(const ThreadId &tid,
                     size_t size, size_t alignment = 0) {
    LArray<T, PointerT> p;
    p.ptr_ = AllocatePtr<T, PointerT>(tid, size, p.shm_, alignment);
    p.size_ = size;
    return p;
  }

  /**
  * Allocate a pointer of \a size size
  * */
  template<typename T, typename PointerT = Pointer>
  HSHM_INLINE_CROSS_FUN
  LArray<T, PointerT>
  ClearAllocateLocalArray(const ThreadId &tid,
                          size_t size, size_t alignment = 0) {
    LArray<T, PointerT> p;
    p.ptr_ = ClearAllocatePtr<T, PointerT>(tid, size, p.shm_, alignment);
    p.size_ = size;
    return p;
  }

  /**
   * Reallocate a pointer to a new size
   *
   * @param p process-independent pointer (input & output)
   * @param new_size the new size to allocate
   * @return A process-specific pointer
   * */
  template<typename T, typename PointerT = Pointer>
  HSHM_INLINE_CROSS_FUN
  LArray<T, PointerT> ReallocateLocalArray(
      const ThreadId &tid,
      LArray<T, PointerT> &p,
      size_t new_size) {
    Reallocate<PointerT>(tid, p.shm_, new_size);
    p.ptr_ = Convert<T>(p.shm_);
    p.size_ = new_size;
    return p;
  }

  /**
   * Free the memory pointed to by \a ptr Pointer
   * */
  template<typename T = void, typename PointerT = Pointer>
  HSHM_INLINE_CROSS_FUN
  void FreeLocalArray(const ThreadId &tid,
                      LArray<T, PointerT> &ptr) {
    if (ptr.ptr_ == nullptr) {
      HERMES_THROW_ERROR(INVALID_FREE);
    }
    FreeOffsetNoNullCheck(tid, ptr.shm_.ToOffsetPointer());
  }

  /**====================================
  * Private Object Allocators
  * ===================================*/

  /**
   * Allocate an array of objects (but don't construct).
   *
   * @param count the number of objects to allocate
   * @param p process-independent pointer (output)
   * @return A process-specific pointer
   * */
  template<typename T, typename PointerT = Pointer>
  HSHM_INLINE_CROSS_FUN
  T* AllocateObjs(const ThreadId &tid,
                  size_t count, PointerT &p) {
    return AllocatePtr<T>(tid, count * sizeof(T), p);
  }

  /**
   * Allocate an array of objects (but don't construct).
   *
   * @return A process-specific pointer
   * */
  template<typename T>
  HSHM_INLINE_CROSS_FUN
  T* AllocateObjs(const ThreadId &tid, size_t count) {
    OffsetPointer p;
    return AllocateObjs<T>(tid, count, p);
  }

  /**
   * Allocate and construct an array of objects
   *
   * @param count the number of objects to allocate
   * @param p process-independent pointer (output)
   * @param args parameters to construct object of type T
   * @return A process-specific pointer
   * */
  template<
      typename T,
      typename PointerT = Pointer,
      typename ...Args>
  HSHM_INLINE_CROSS_FUN
  T* AllocateConstructObjs(
      const ThreadId &tid,
      size_t count,
      PointerT &p, Args&& ...args) {
    T *ptr = AllocateObjs<T>(tid, count, p);
    ConstructObjs<T>(ptr, 0, count, std::forward<Args>(args)...);
    return ptr;
  }

  /**
   * Allocate and construct an array of objects
   *
   * @param count the number of objects to allocate
   * @param p process-independent pointer (output)
   * @param args parameters to construct object of type T
   * @return A process-specific pointer
   * */
  template<
      typename T,
      typename ...Args>
  HSHM_INLINE_CROSS_FUN
  T* AllocateConstructObjs(const ThreadId &tid,
                           size_t count, Args&& ...args) {
    OffsetPointer p;
    return AllocateConstructObjs<T, OffsetPointer>(
        tid, count, p, std::forward<Args>(args)...);
  }

  /** Allocate + construct an array of objects */
  template<
      typename T,
      typename PointerT = Pointer,
      typename ...Args>
  HSHM_INLINE_CROSS_FUN
  T* NewObjs(const ThreadId &tid, size_t count,
             PointerT &p, Args&& ...args) {
    return AllocateConstructObjs<T>(tid, count, p,
                                    std::forward<Args>(args)...);
  }

  /** Allocate + construct an array of objects */
  template<
      typename T,
      typename ...Args>
  HSHM_INLINE_CROSS_FUN
  T* NewObjs(const ThreadId &tid, size_t count,
             Args&& ...args) {
    OffsetPointer p;
    return NewObjs<T>(tid, count, p, std::forward<Args>(args)...);
  }

  /** Allocate + construct a single object */
  template<
      typename T,
      typename PointerT = Pointer,
      typename ...Args>
  HSHM_INLINE_CROSS_FUN
  T* NewObj(const ThreadId &tid, PointerT &p, Args&& ...args) {
    return NewObjs<T>(tid, 1, p, std::forward<Args>(args)...);
  }

  /** Allocate + construct a single object */
  template<
      typename T,
      typename ...Args>
  HSHM_INLINE_CROSS_FUN
  T* NewObj(const ThreadId &tid, Args&& ...args) {
    OffsetPointer p;
    return NewObj<T>(tid, p, std::forward<Args>(args)...);
  }

  /**
   * Reallocate a pointer of objects to a new size.
   *
   * @param p process-independent pointer (input & output)
   * @param old_count the original number of objects (avoids reconstruction)
   * @param new_count the new number of objects
   *
   * @return A process-specific pointer
   * */
  template<typename T, typename PointerT = Pointer>
  HSHM_INLINE_CROSS_FUN
  T* ReallocateObjs(const ThreadId &tid,
                    PointerT &p, size_t new_count) {
    T *ptr = ReallocatePtr<T>(tid, p, new_count * sizeof(T));
    return ptr;
  }

  /**
   * Reallocate a pointer of objects to a new size and construct the
   * new elements in-place.
   *
   * @param p process-independent pointer (input & output)
   * @param old_count the original number of objects (avoids reconstruction)
   * @param new_count the new number of objects
   * @param args parameters to construct object of type T
   *
   * @return A process-specific pointer
   * */
  template<
      typename T,
      typename PointerT = Pointer,
      typename ...Args>
  HSHM_INLINE_CROSS_FUN
  T* ReallocateConstructObjs(const ThreadId &tid,
                             PointerT &p,
                             size_t old_count,
                             size_t new_count,
                             Args&& ...args) {
    T *ptr = ReallocatePtr<T>(tid, p, new_count * sizeof(T));
    ConstructObjs<T>(ptr, old_count, new_count, std::forward<Args>(args)...);
    return ptr;
  }

  /**
 * Free + destruct objects
 * */
  template <typename T>
  HSHM_INLINE_CROSS_FUN
  void FreeDestructObjs(const ThreadId &tid, T *ptr, size_t count) {
    DestructObjs<T>(ptr, count);
    auto p = Convert<T, OffsetPointer>(ptr);
    Free(tid, p);
  }


  /**
   * Free + destruct objects
   * */
  template <typename T>
  HSHM_INLINE_CROSS_FUN
  void DelObjs(const ThreadId &tid, T *ptr, size_t count) {
    FreeDestructObjs<T>(tid, ptr, count);
  }


  /**
   * Free + destruct an object
   * */
  template <typename T>
  HSHM_INLINE_CROSS_FUN void DelObj(const ThreadId &tid, T *ptr) {
    FreeDestructObjs<T>(tid, ptr, 1);
  }

  /**====================================
   * Local Object Allocators
   * ===================================*/

  /**
   * Allocate an array of objects (but don't construct).
   *
   * @return A LocalPointer
   * */
  template<typename T, typename PointerT = Pointer>
  HSHM_INLINE_CROSS_FUN
  LPointer<T, PointerT>
  AllocateObjsLocal(const ThreadId &tid, size_t count) {
    LPointer<T, PointerT> p;
    p.ptr_ = AllocateObjs<T>(tid, count, p.shm_);
    return p;
  }

  /**
   * Allocate and construct an array of objects
   *
   * @param count the number of objects to allocate
   * @param p process-independent pointer (output)
   * @param args parameters to construct object of type T
   * @return A process-specific pointer
   * */
  template<
      typename T,
      typename PointerT = Pointer,
      typename ...Args>
  HSHM_INLINE_CROSS_FUN
  LPointer<T, PointerT>
  AllocateConstructObjsLocal(const ThreadId &tid,
                             size_t count,
                             Args&& ...args) {
    LPointer<T, PointerT> p;
    p.ptr_ = AllocateConstructObjs<T, OffsetPointer>(
        tid, count, p.shm_, std::forward<Args>(args)...);
    return p;
  }

  /** Allocate + construct an array of objects */
  template<
      typename T,
      typename PointerT = Pointer,
      typename ...Args>
  HSHM_INLINE_CROSS_FUN
  LPointer<T, PointerT>
  NewObjsLocal(const ThreadId &tid, size_t count, Args&& ...args) {
    LPointer<T, PointerT> p;
    p.ptr_ = NewObjs<T>(tid, count, p.shm_, std::forward<Args>(args)...);
    return p;
  }

  /** Allocate + construct a single object */
  template<
      typename T,
      typename PointerT = Pointer,
      typename ...Args>
  HSHM_INLINE_CROSS_FUN
  LPointer<T, PointerT> NewObjLocal(const ThreadId &tid, Args&& ...args) {
    LPointer<T, PointerT> p;
    p.ptr_ = NewObj<T>(tid, p.shm_, std::forward<Args>(args)...);
    return p;
  }

  /**
   * Reallocate a pointer of objects to a new size.
   *
   * @param p process-independent pointer (input & output)
   * @param old_count the original number of objects (avoids reconstruction)
   * @param new_count the new number of objects
   *
   * @return A process-specific pointer
   * */
  template<typename T, typename PointerT = Pointer>
  HSHM_INLINE_CROSS_FUN
  void ReallocateObjsLocal(const ThreadId &tid,
                           LPointer<T, PointerT> &p,
                           size_t new_count) {
    p.ptr_ = ReallocatePtr<T>(tid, p.shm_, new_count * sizeof(T));
  }

  /**
  * Reallocate a pointer of objects to a new size and construct the
  * new elements in-place.
  *
  * @param p process-independent pointer (input & output)
  * @param old_count the original number of objects (avoids reconstruction)
  * @param new_count the new number of objects
  * @param args parameters to construct object of type T
  *
  * @return A process-specific pointer
  * */
  template<
      typename T,
      typename PointerT = Pointer,
      typename ...Args>
  HSHM_INLINE_CROSS_FUN
  void ReallocateConstructObjsLocal(const ThreadId &tid,
                                    LPointer<T, PointerT> &p,
                                    size_t old_count,
                                    size_t new_count,
                                    Args&& ...args) {
    p.ptr_ = ReallocateConstructObjs<T>(tid, p.shm_, old_count, new_count,
                                        std::forward<Args>(args)...);
  }

  /**
   * Free + destruct objects
   * */
  template <typename T, typename PointerT>
  HSHM_INLINE_CROSS_FUN
  void FreeDestructObjsLocal(const ThreadId &tid,
                             LPointer<T, PointerT> &p, size_t count) {
    DestructObjs<T>(p.ptr_, count);
    Free(tid, p.shm_);
  }

  /**
   * Free + destruct objects
   * */
  template <typename T, typename PointerT>
  HSHM_INLINE_CROSS_FUN
  void DelObjsLocal(const ThreadId &tid,
                    LPointer<T, PointerT> &p, size_t count) {
    FreeDestructObjsLocal<T>(tid, p, count);
  }

  /**
   * Free + destruct an object
   * */
  template <typename T, typename PointerT>
  HSHM_INLINE_CROSS_FUN
  void DelObjLocal(const ThreadId &tid,
                   LPointer<T, PointerT> &p) {
    FreeDestructObjsLocal<T>(tid, p, 1);
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
  template<
      typename T,
      typename ...Args>
  HSHM_INLINE_CROSS_FUN
  static void ConstructObjs(T *ptr,
                            size_t old_count,
                            size_t new_count, Args&& ...args) {
    if (ptr == nullptr) { return; }
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
  template<
      typename T,
      typename ...Args>
  HSHM_INLINE_CROSS_FUN
  static void ConstructObj(T &obj, Args&& ...args) {
    new (&obj) T(std::forward<Args>(args)...);
  }

  /**
   * Destruct an array of objects
   *
   * @param ptr the object to destruct (potentially archived)
   * @param count the length of the object array
   * @return None
   * */
  template<typename T>
  HSHM_INLINE_CROSS_FUN
  static void DestructObjs(T *ptr, size_t count) {
    if (ptr == nullptr) { return; }
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
  template<typename T>
  HSHM_INLINE_CROSS_FUN static void DestructObj(T &obj) {
    obj.~T();
  }

  /**====================================
  * Helpers
  * ===================================*/

  /**
   * Get the custom header of the shared-memory allocator
   *
   * @return Custom header pointer
   * */
  template<typename HEADER_T>
  HSHM_INLINE_CROSS_FUN HEADER_T* GetCustomHeader() {
    return reinterpret_cast<HEADER_T*>(custom_header_);
  }

  /**
   * Convert a process-independent pointer into a process-specific pointer
   *
   * @param p process-independent pointer
   * @return a process-specific pointer
   * */
  template<typename T, typename PointerT = Pointer>
  HSHM_INLINE_CROSS_FUN T* Convert(const PointerT &p) {
    if (p.IsNull()) { return nullptr; }
    return reinterpret_cast<T*>(buffer_ + p.off_.load());
  }

  /**
   * Convert a process-specific pointer into a process-independent pointer
   *
   * @param ptr process-specific pointer
   * @return a process-independent pointer
   * */
  template<typename T, typename PointerT = Pointer>
  HSHM_INLINE_CROSS_FUN PointerT Convert(const T *ptr) {
    if (ptr == nullptr) { return PointerT::GetNull(); }
    return PointerT(GetId(),
                    reinterpret_cast<size_t>(ptr) -
                        reinterpret_cast<size_t>(buffer_));
  }

  /**
   * Determine whether or not this allocator contains a process-specific
   * pointer
   *
   * @param ptr process-specific pointer
   * @return True or false
   * */
  template<typename T = void>
  HSHM_INLINE_CROSS_FUN bool ContainsPtr(T *ptr) {
    return  reinterpret_cast<size_t>(ptr) >=
        reinterpret_cast<size_t>(buffer_);
  }

  /** Print */
  HSHM_CROSS_FUN
  void Print() {
    printf("(%s) Allocator: type: %d, id: %d.%d, custom_header: %p\n",
           kCurrentDevice,
           static_cast<int>(type_),
           GetId().bits_.major_,
           GetId().bits_.minor_,
           custom_header_);
  }
};

typedef u64 ThreadLocalId;

template<typename AllocT>
struct ThreadLocalAllocator {
  ThreadLocalId tls_;
  AllocT *alloc_;
};

}  // namespace hshm::ipc

#endif  // HERMES_MEMORY_ALLOCATOR_ALLOCATOR_H_
