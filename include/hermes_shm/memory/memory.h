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

#ifndef HSHM_MEMORY_MEMORY_H_
#define HSHM_MEMORY_MEMORY_H_

#include <cstdio>

#include "hermes_shm/constants/macros.h"
#include "hermes_shm/data_structures/ipc/hash.h"
#include "hermes_shm/introspect/system_info.h"
#include "hermes_shm/types/atomic.h"
#include "hermes_shm/types/bitfield.h"
#include "hermes_shm/types/real_number.h"

namespace hshm::ipc {

/**
 * The identifier for an allocator
 * */
union AllocatorId {
  struct {
    i32 major_;  // Typically some sort of process id
    i32 minor_;  // Typically a process-local id
  } bits_;
  u64 int_;

  HSHM_INLINE_CROSS_FUN AllocatorId() = default;

  /**
   * Constructor which sets major & minor
   * */
  HSHM_INLINE_CROSS_FUN explicit AllocatorId(i32 major, i32 minor) {
    bits_.major_ = major;
    bits_.minor_ = minor;
  }

  /**
   * Set this allocator to null
   * */
  HSHM_INLINE_CROSS_FUN void SetNull() { (*this) = GetNull(); }

  /**
   * Check if this is the null allocator
   * */
  HSHM_INLINE_CROSS_FUN bool IsNull() const { return (*this) == GetNull(); }

  /** Equality check */
  HSHM_INLINE_CROSS_FUN bool operator==(const AllocatorId &other) const {
    return other.int_ == int_;
  }

  /** Inequality check */
  HSHM_INLINE_CROSS_FUN bool operator!=(const AllocatorId &other) const {
    return other.int_ != int_;
  }

  /** Get the null allocator */
  HSHM_INLINE_CROSS_FUN static AllocatorId GetNull() {
    return AllocatorId(-1, -1);
  }

  /** To index */
  HSHM_INLINE_CROSS_FUN uint32_t ToIndex() const {
    return bits_.major_ * 2 + bits_.minor_;
  }

  /** Serialize an hipc::allocator_id */
  template <typename Ar>
  HSHM_INLINE_CROSS_FUN void serialize(Ar &ar) {
    ar & int_;
  }

  /** Print */
  HSHM_CROSS_FUN
  void Print() const {
    printf("(%s) Allocator ID: %u.%u\n", kCurrentDevice, bits_.major_,
           bits_.minor_);
  }
};

class Allocator;

/** Pointer type base */
class ShmPointer {};

/**
 * Stores an offset into a memory region. Assumes the developer knows
 * which allocator the pointer comes from.
 * */
template <bool ATOMIC = false>
struct OffsetPointerBase : public ShmPointer {
  hipc::opt_atomic<hshm::size_t, ATOMIC>
      off_; /**< Offset within the allocator's slot */

  /** Default constructor */
  HSHM_INLINE_CROSS_FUN OffsetPointerBase() = default;

  /** Full constructor */
  HSHM_INLINE_CROSS_FUN explicit OffsetPointerBase(size_t off) : off_(off) {}

  /** Full constructor */
  HSHM_INLINE_CROSS_FUN explicit OffsetPointerBase(
      hipc::opt_atomic<hshm::size_t, ATOMIC> off)
      : off_(off.load()) {}

  /** Pointer constructor */
  HSHM_INLINE_CROSS_FUN explicit OffsetPointerBase(AllocatorId alloc_id,
                                                   size_t off)
      : off_(off) {
    (void)alloc_id;
  }

  /** Pointer constructor (alloc + atomic offset)*/
  HSHM_INLINE_CROSS_FUN explicit OffsetPointerBase(AllocatorId id,
                                                   OffsetPointerBase<true> off)
      : off_(off.load()) {
    (void)id;
  }

  /** Pointer constructor (alloc + non-offeset) */
  HSHM_INLINE_CROSS_FUN explicit OffsetPointerBase(AllocatorId id,
                                                   OffsetPointerBase<false> off)
      : off_(off.load()) {
    (void)id;
  }

  /** Copy constructor */
  HSHM_INLINE_CROSS_FUN OffsetPointerBase(const OffsetPointerBase &other)
      : off_(other.off_.load()) {}

  /** Other copy constructor */
  HSHM_INLINE_CROSS_FUN OffsetPointerBase(
      const OffsetPointerBase<!ATOMIC> &other)
      : off_(other.off_.load()) {}

  /** Move constructor */
  HSHM_INLINE_CROSS_FUN OffsetPointerBase(OffsetPointerBase &&other) noexcept
      : off_(other.off_.load()) {
    other.SetNull();
  }

  /** Get the offset pointer */
  HSHM_INLINE_CROSS_FUN OffsetPointerBase<false> ToOffsetPointer() {
    return OffsetPointerBase<false>(off_.load());
  }

  /** Set to null (offsets can be 0, so not 0) */
  HSHM_INLINE_CROSS_FUN void SetNull() { off_ = (size_t)-1; }

  /** Check if null */
  HSHM_INLINE_CROSS_FUN bool IsNull() const {
    return off_.load() == (size_t)-1;
  }

  /** Get the null pointer */
  HSHM_INLINE_CROSS_FUN static OffsetPointerBase GetNull() {
    return OffsetPointerBase((size_t)-1);
  }

  /** Atomic load wrapper */
  HSHM_INLINE_CROSS_FUN size_t
  load(std::memory_order order = std::memory_order_seq_cst) const {
    return off_.load(order);
  }

  /** Atomic exchange wrapper */
  HSHM_INLINE_CROSS_FUN void exchange(
      size_t count, std::memory_order order = std::memory_order_seq_cst) {
    off_.exchange(count, order);
  }

  /** Atomic compare exchange weak wrapper */
  HSHM_INLINE_CROSS_FUN bool compare_exchange_weak(
      size_t &expected, size_t desired,
      std::memory_order order = std::memory_order_seq_cst) {
    return off_.compare_exchange_weak(expected, desired, order);
  }

  /** Atomic compare exchange strong wrapper */
  HSHM_INLINE_CROSS_FUN bool compare_exchange_strong(
      size_t &expected, size_t desired,
      std::memory_order order = std::memory_order_seq_cst) {
    return off_.compare_exchange_weak(expected, desired, order);
  }

  /** Atomic add operator */
  HSHM_INLINE_CROSS_FUN OffsetPointerBase operator+(size_t count) const {
    return OffsetPointerBase(off_ + count);
  }

  /** Atomic subtract operator */
  HSHM_INLINE_CROSS_FUN OffsetPointerBase operator-(size_t count) const {
    return OffsetPointerBase(off_ - count);
  }

  /** Atomic add assign operator */
  HSHM_INLINE_CROSS_FUN OffsetPointerBase &operator+=(size_t count) {
    off_ += count;
    return *this;
  }

  /** Atomic subtract assign operator */
  HSHM_INLINE_CROSS_FUN OffsetPointerBase &operator-=(size_t count) {
    off_ -= count;
    return *this;
  }

  /** Atomic increment (post) */
  HSHM_INLINE_CROSS_FUN OffsetPointerBase operator++(int) {
    return OffsetPointerBase(off_++);
  }

  /** Atomic increment (pre) */
  HSHM_INLINE_CROSS_FUN OffsetPointerBase &operator++() {
    ++off_;
    return *this;
  }

  /** Atomic decrement (post) */
  HSHM_INLINE_CROSS_FUN OffsetPointerBase operator--(int) {
    return OffsetPointerBase(off_--);
  }

  /** Atomic decrement (pre) */
  HSHM_INLINE_CROSS_FUN OffsetPointerBase &operator--() {
    --off_;
    return *this;
  }

  /** Atomic assign operator */
  HSHM_INLINE_CROSS_FUN OffsetPointerBase &operator=(size_t count) {
    off_ = count;
    return *this;
  }

  /** Atomic copy assign operator */
  HSHM_INLINE_CROSS_FUN OffsetPointerBase &operator=(
      const OffsetPointerBase &count) {
    off_ = count.load();
    return *this;
  }

  /** Equality check */
  HSHM_INLINE_CROSS_FUN bool operator==(const OffsetPointerBase &other) const {
    return off_ == other.off_;
  }

  /** Inequality check */
  HSHM_INLINE_CROSS_FUN bool operator!=(const OffsetPointerBase &other) const {
    return off_ != other.off_;
  }

  /** Mark first bit */
  HSHM_INLINE_CROSS_FUN OffsetPointerBase Mark() const {
    return OffsetPointerBase(MARK_FIRST_BIT(size_t, off_.load()));
  }

  /** Check if first bit is marked */
  HSHM_INLINE_CROSS_FUN bool IsMarked() const {
    return IS_FIRST_BIT_MARKED(size_t, off_.load());
  }

  /** Unmark first bit */
  HSHM_INLINE_CROSS_FUN OffsetPointerBase Unmark() const {
    return OffsetPointerBase(UNMARK_FIRST_BIT(size_t, off_.load()));
  }

  /** Set to 0 */
  HSHM_INLINE_CROSS_FUN void SetZero() { off_ = 0; }
};

/** Non-atomic offset */
typedef OffsetPointerBase<false> OffsetPointer;

/** Atomic offset */
typedef OffsetPointerBase<true> AtomicOffsetPointer;

/** Typed offset pointer */
template <typename T>
using TypedOffsetPointer = OffsetPointer;

/** Typed atomic pointer */
template <typename T>
using TypedAtomicOffsetPointer = AtomicOffsetPointer;

/**
 * A process-independent pointer, which stores both the allocator's
 * information and the offset within the allocator's region
 * */
template <bool ATOMIC = false>
struct PointerBase : public ShmPointer {
  AllocatorId alloc_id_;           /// Allocator the pointer comes from
  OffsetPointerBase<ATOMIC> off_;  /// Offset within the allocator's slot

  /** Default constructor */
  HSHM_INLINE_CROSS_FUN PointerBase() = default;

  /** Full constructor */
  HSHM_INLINE_CROSS_FUN explicit PointerBase(AllocatorId id, size_t off)
      : alloc_id_(id), off_(off) {}

  /** Full constructor using offset pointer */
  HSHM_INLINE_CROSS_FUN explicit PointerBase(AllocatorId id, OffsetPointer off)
      : alloc_id_(id), off_(off) {}

  /** Copy constructor */
  HSHM_INLINE_CROSS_FUN PointerBase(const PointerBase &other)
      : alloc_id_(other.alloc_id_), off_(other.off_) {}

  /** Other copy constructor */
  HSHM_INLINE_CROSS_FUN PointerBase(const PointerBase<!ATOMIC> &other)
      : alloc_id_(other.alloc_id_), off_(other.off_.load()) {}

  /** Move constructor */
  HSHM_INLINE_CROSS_FUN PointerBase(PointerBase &&other) noexcept
      : alloc_id_(other.alloc_id_), off_(other.off_) {
    other.SetNull();
  }

  /** Get the offset pointer */
  HSHM_INLINE_CROSS_FUN OffsetPointerBase<false> ToOffsetPointer() const {
    return OffsetPointerBase<false>(off_.load());
  }

  /** Set to null */
  HSHM_INLINE_CROSS_FUN void SetNull() { alloc_id_.SetNull(); }

  /** Check if null */
  HSHM_INLINE_CROSS_FUN bool IsNull() const { return alloc_id_.IsNull(); }

  /** Get the null pointer */
  HSHM_INLINE_CROSS_FUN static PointerBase GetNull() {
    return PointerBase{AllocatorId::GetNull(), OffsetPointer::GetNull()};
  }

  /** Copy assignment operator */
  HSHM_INLINE_CROSS_FUN PointerBase &operator=(const PointerBase &other) {
    if (this != &other) {
      alloc_id_ = other.alloc_id_;
      off_ = other.off_;
    }
    return *this;
  }

  /** Move assignment operator */
  HSHM_INLINE_CROSS_FUN PointerBase &operator=(PointerBase &&other) {
    if (this != &other) {
      alloc_id_ = other.alloc_id_;
      off_.exchange(other.off_.load());
      other.SetNull();
    }
    return *this;
  }

  /** Addition operator */
  HSHM_INLINE_CROSS_FUN PointerBase operator+(size_t size) const {
    PointerBase p;
    p.alloc_id_ = alloc_id_;
    p.off_ = off_ + size;
    return p;
  }

  /** Subtraction operator */
  HSHM_INLINE_CROSS_FUN PointerBase operator-(size_t size) const {
    PointerBase p;
    p.alloc_id_ = alloc_id_;
    p.off_ = off_ - size;
    return p;
  }

  /** Addition assignment operator */
  HSHM_INLINE_CROSS_FUN PointerBase &operator+=(size_t size) {
    off_ += size;
    return *this;
  }

  /** Subtraction assignment operator */
  HSHM_INLINE_CROSS_FUN PointerBase &operator-=(size_t size) {
    off_ -= size;
    return *this;
  }

  /** Increment operator (pre) */
  HSHM_INLINE_CROSS_FUN PointerBase &operator++() {
    off_++;
    return *this;
  }

  /** Decrement operator (pre) */
  HSHM_INLINE_CROSS_FUN PointerBase &operator--() {
    off_--;
    return *this;
  }

  /** Increment operator (post) */
  HSHM_INLINE_CROSS_FUN PointerBase operator++(int) {
    PointerBase tmp(*this);
    operator++();
    return tmp;
  }

  /** Decrement operator (post) */
  HSHM_INLINE_CROSS_FUN PointerBase operator--(int) {
    PointerBase tmp(*this);
    operator--();
    return tmp;
  }

  /** Equality check */
  HSHM_INLINE_CROSS_FUN bool operator==(const PointerBase &other) const {
    return (other.alloc_id_ == alloc_id_ && other.off_ == off_);
  }

  /** Inequality check */
  HSHM_INLINE_CROSS_FUN bool operator!=(const PointerBase &other) const {
    return (other.alloc_id_ != alloc_id_ || other.off_ != off_);
  }

  /** Mark first bit */
  HSHM_INLINE_CROSS_FUN PointerBase Mark() const {
    return PointerBase(alloc_id_, off_.Mark());
  }

  /** Check if first bit is marked */
  HSHM_INLINE_CROSS_FUN bool IsMarked() const { return off_.IsMarked(); }

  /** Unmark first bit */
  HSHM_INLINE_CROSS_FUN PointerBase Unmark() const {
    return PointerBase(alloc_id_, off_.Unmark());
  }

  /** Set to 0 */
  HSHM_INLINE_CROSS_FUN void SetZero() { off_.SetZero(); }
};

/** Non-atomic pointer */
typedef PointerBase<false> Pointer;

/** Atomic pointer */
typedef PointerBase<true> AtomicPointer;

/** Typed pointer */
template <typename T>
using TypedPointer = Pointer;

/** Typed atomic pointer */
template <typename T>
using TypedAtomicPointer = AtomicPointer;

/** Struct containing both private and shared pointer */
template <typename T = char, typename PointerT = Pointer>
struct LPointer : public ShmPointer {
  T *ptr_;
  PointerT shm_;

  /** Default constructor */
  HSHM_INLINE_CROSS_FUN LPointer() = default;

  /** Full constructor */
  HSHM_INLINE_CROSS_FUN LPointer(T *ptr, const PointerT &shm)
      : ptr_(ptr), shm_(shm) {}

  /** SHM constructor (in memory_manager.h) */
  HSHM_INLINE_CROSS_FUN explicit LPointer(const PointerT &shm);

  /** Private half constructor (in memory_manager.h) */
  HSHM_INLINE_CROSS_FUN explicit LPointer(T *ptr);

  /** Private half + alloc constructor (in memory_manager.h) */
  HSHM_INLINE_CROSS_FUN explicit LPointer(hipc::Allocator *alloc, T *ptr);

  /** Shared half + alloc constructor (in memory_manager.h) */
  HSHM_INLINE_CROSS_FUN explicit LPointer(hipc::Allocator *alloc,
                                          const OffsetPointer &shm);

  /** Copy constructor */
  HSHM_INLINE_CROSS_FUN LPointer(const LPointer &other)
      : ptr_(other.ptr_), shm_(other.shm_) {}

  /** Move constructor */
  HSHM_INLINE_CROSS_FUN LPointer(LPointer &&other) noexcept
      : ptr_(other.ptr_), shm_(other.shm_) {
    other.SetNull();
  }

  /** Copy assignment operator */
  HSHM_INLINE_CROSS_FUN LPointer &operator=(const LPointer &other) {
    if (this != &other) {
      ptr_ = other.ptr_;
      shm_ = other.shm_;
    }
    return *this;
  }

  /** Move assignment operator */
  HSHM_INLINE_CROSS_FUN LPointer &operator=(LPointer &&other) {
    if (this != &other) {
      ptr_ = other.ptr_;
      shm_ = other.shm_;
      other.SetNull();
    }
    return *this;
  }

  /** Overload arrow */
  HSHM_INLINE_CROSS_FUN T *operator->() const { return ptr_; }

  /** Overload dereference */
  HSHM_INLINE_CROSS_FUN T &operator*() const { return *ptr_; }

  /** Equality operator */
  HSHM_INLINE_CROSS_FUN bool operator==(const LPointer &other) const {
    return ptr_ == other.ptr_ && shm_ == other.shm_;
  }

  /** Inequality operator */
  HSHM_INLINE_CROSS_FUN bool operator!=(const LPointer &other) const {
    return ptr_ != other.ptr_ || shm_ != other.shm_;
  }

  /** Addition operator */
  HSHM_INLINE_CROSS_FUN LPointer operator+(size_t size) const {
    return LPointer(ptr_ + size, shm_ + size);
  }

  /** Subtraction operator */
  HSHM_INLINE_CROSS_FUN LPointer operator-(size_t size) const {
    return LPointer(ptr_ - size, shm_ - size);
  }

  /** Addition assignment operator */
  HSHM_INLINE_CROSS_FUN LPointer &operator+=(size_t size) {
    ptr_ += size;
    shm_ += size;
    return *this;
  }

  /** Subtraction assignment operator */
  HSHM_INLINE_CROSS_FUN LPointer &operator-=(size_t size) {
    ptr_ -= size;
    shm_ -= size;
    return *this;
  }

  /** Increment operator (pre) */
  HSHM_INLINE_CROSS_FUN LPointer &operator++() {
    ptr_++;
    shm_++;
    return *this;
  }

  /** Decrement operator (pre) */
  HSHM_INLINE_CROSS_FUN LPointer &operator--() {
    ptr_--;
    shm_--;
    return *this;
  }

  /** Increment operator (post) */
  HSHM_INLINE_CROSS_FUN LPointer operator++(int) {
    LPointer tmp(*this);
    operator++();
    return tmp;
  }

  /** Decrement operator (post) */
  HSHM_INLINE_CROSS_FUN LPointer operator--(int) {
    LPointer tmp(*this);
    operator--();
    return tmp;
  }

  /** Check if null */
  HSHM_INLINE_CROSS_FUN bool IsNull() const { return ptr_ == nullptr; }

  /** Get null */
  HSHM_INLINE_CROSS_FUN static LPointer GetNull() {
    return LPointer(nullptr, Pointer::GetNull());
  }

  /** Set to null */
  HSHM_INLINE_CROSS_FUN void SetNull() { ptr_ = nullptr; }

  /** Reintrepret cast to other internal type */
  template <typename U>
  HSHM_INLINE_CROSS_FUN LPointer<U, PointerT> &Cast() {
    return DeepCast<LPointer<U, PointerT>>();
  }

  /** Reintrepret cast to other internal type (const) */
  template <typename U>
  HSHM_INLINE_CROSS_FUN const LPointer<U, PointerT> &Cast() const {
    return DeepCast<LPointer<U, PointerT>>();
  }

  /** Reintrepret cast to another LPointer */
  template <typename LPointerT>
  HSHM_INLINE_CROSS_FUN LPointerT &DeepCast() {
    return *((LPointerT *)this);
  }

  /** Reintrepret cast to another LPointer (const) */
  template <typename LPointerT>
  HSHM_INLINE_CROSS_FUN const LPointerT &DeepCast() const {
    return *((LPointerT *)this);
  }

  /** Mark first bit */
  HSHM_INLINE_CROSS_FUN LPointer Mark() const {
    return LPointer(ptr_, shm_.Mark());
  }

  /** Check if first bit is marked */
  HSHM_INLINE_CROSS_FUN bool IsMarked() const { return shm_.IsMarked(); }

  /** Unmark first bit */
  HSHM_INLINE_CROSS_FUN LPointer Unmark() const {
    return LPointer(ptr_, shm_.Unmark());
  }

  /** Set to 0 */
  HSHM_INLINE_CROSS_FUN void SetZero() { shm_.SetZero(); }
};

/** Alias to local pointer */
template <typename T = char, typename PointerT = Pointer>
using FullPtr = LPointer<T, PointerT>;

/** Struct containing both a pointer and its size */
template <typename PointerT = Pointer>
struct Array {
  PointerT shm_;
  size_t size_;
};

/** Struct containing a shared pointer, private pointer, and the data size */
template <typename T = char, typename PointerT = Pointer>
struct LArray {
  PointerT shm_;
  size_t size_;
  T *ptr_;

  /** Overload arrow */
  HSHM_INLINE_CROSS_FUN T *operator->() const { return ptr_; }

  /** Overload dereference */
  HSHM_INLINE_CROSS_FUN T &operator*() const { return *ptr_; }
};

class MemoryAlignment {
 public:
  /**
   * Round up to the nearest multiple of the alignment
   * @param alignment the alignment value (e.g., 4096)
   * @param size the size to make a multiple of alignment (e.g., 4097)
   * @return the new size  (e.g., 8192)
   * */
  static size_t AlignTo(size_t alignment, size_t size) {
    auto page_size = HSHM_SYSTEM_INFO->page_size_;
    size_t new_size = size;
    size_t page_off = size % alignment;
    if (page_off) {
      new_size = size + page_size - page_off;
    }
    return new_size;
  }

  /**
   * Round up to the nearest multiple of page size
   * @param size the size to align to the PAGE_SIZE
   * */
  static size_t AlignToPageSize(size_t size) {
    auto page_size = HSHM_SYSTEM_INFO->page_size_;
    size_t new_size = AlignTo(page_size, size);
    return new_size;
  }
};

}  // namespace hshm::ipc

namespace std {

/** Allocator ID hash */
template <>
struct hash<hshm::ipc::AllocatorId> {
  std::size_t operator()(const hshm::ipc::AllocatorId &key) const {
    return hshm::hash<uint64_t>{}(key.int_);
  }
};

}  // namespace std

namespace hshm {

/** Allocator ID hash */
template <>
struct hash<hshm::ipc::AllocatorId> {
  HSHM_INLINE_CROSS_FUN std::size_t operator()(
      const hshm::ipc::AllocatorId &key) const {
    return hshm::hash<uint64_t>{}(key.int_);
  }
};

}  // namespace hshm

#define IS_SHM_POINTER(T) std::is_base_of_v<hipc::ShmPointer, T>

#endif  // HSHM_MEMORY_MEMORY_H_
