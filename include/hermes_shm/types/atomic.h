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

#ifndef HSHM_INCLUDE_HSHM_TYPES_ATOMIC_H_
#define HSHM_INCLUDE_HSHM_TYPES_ATOMIC_H_

#include <atomic>
#include <type_traits>

#include "hermes_shm/constants/macros.h"
#include "numbers.h"
#ifdef HSHM_ENABLE_CUDA
#include <cuda/atomic>
#endif
#ifdef HSHM_ENABLE_ROCM
#include <hip/hip_runtime.h>
#endif

namespace hshm::ipc {

/** Provides the API of an atomic, without being atomic */
template <typename T>
struct nonatomic {
  T x;

  /** Serialization */
  template <typename Ar>
  void serialize(Ar &ar) {
    ar(x);
  }

  /** Integer convertion */
  HSHM_INLINE_CROSS_FUN operator T() const { return x; }

  /** Constructor */
  HSHM_INLINE_CROSS_FUN nonatomic() = default;

  /** Full constructor */
  template <typename U>
  HSHM_INLINE_CROSS_FUN nonatomic(U def) : x(def) {}

  /** Copy constructor */
  HSHM_INLINE_CROSS_FUN nonatomic(const nonatomic &other) : x(other.x) {}

  /* Move constructor */
  HSHM_INLINE_CROSS_FUN nonatomic(nonatomic &&other) : x(std::move(other.x)) {}

  /** Copy assign operator */
  HSHM_INLINE_CROSS_FUN nonatomic &operator=(const nonatomic &other) {
    x = other.x;
    return *this;
  }

  /** Move assign operator */
  HSHM_INLINE_CROSS_FUN nonatomic &operator=(nonatomic &&other) {
    x = std::move(other.x);
    return *this;
  }

  /** Atomic fetch_add wrapper*/
  template <typename U>
  HSHM_INLINE_CROSS_FUN T
  fetch_add(U count, std::memory_order order = std::memory_order_seq_cst) {
    (void)order;
    T orig_x = x;
    x += (T)count;
    return orig_x;
  }

  /** Atomic fetch_sub wrapper*/
  template <typename U>
  HSHM_INLINE_CROSS_FUN T
  fetch_sub(U count, std::memory_order order = std::memory_order_seq_cst) {
    (void)order;
    T orig_x = x;
    x -= (T)count;
    return orig_x;
  }

  /** Atomic load wrapper */
  HSHM_INLINE_CROSS_FUN T
  load(std::memory_order order = std::memory_order_seq_cst) const {
    (void)order;
    return x;
  }

  /** Get reference to x */
  HSHM_INLINE_CROSS_FUN T &ref() { return x; }

  /** Get const reference to x */
  HSHM_INLINE_CROSS_FUN const T &ref() const { return x; }

  /** Atomic exchange wrapper */
  template <typename U>
  HSHM_INLINE_CROSS_FUN void exchange(
      U count, std::memory_order order = std::memory_order_seq_cst) {
    (void)order;
    x = count;
  }

  /** Atomic compare exchange weak wrapper */
  template <typename U>
  HSHM_INLINE_CROSS_FUN bool compare_exchange_weak(
      T &expected, U desired,
      std::memory_order order = std::memory_order_seq_cst) {
    (void)expected;
    (void)order;
    x = desired;
    return true;
  }

  /** Atomic compare exchange strong wrapper */
  template <typename U>
  HSHM_INLINE_CROSS_FUN bool compare_exchange_strong(
      T &expected, U desired,
      std::memory_order order = std::memory_order_seq_cst) {
    (void)expected;
    (void)order;
    x = desired;
    return true;
  }

  /** Atomic pre-increment operator */
  HSHM_INLINE_CROSS_FUN nonatomic &operator++() {
    ++x;
    return *this;
  }

  /** Atomic post-increment operator */
  HSHM_INLINE_CROSS_FUN nonatomic operator++(int) { return atomic(x + 1); }

  /** Atomic pre-decrement operator */
  HSHM_INLINE_CROSS_FUN nonatomic &operator--() {
    --x;
    return *this;
  }

  /** Atomic post-decrement operator */
  HSHM_INLINE_CROSS_FUN nonatomic operator--(int) {
    nonatomic orig_x(x);
    --x;
    return orig_x;
  }

  /** Atomic add operator */
  template <typename U>
  HSHM_INLINE_CROSS_FUN nonatomic operator+(U count) const {
    return nonatomic(x + count);
  }

  /** Atomic subtract operator */
  template <typename U>
  HSHM_INLINE_CROSS_FUN nonatomic operator-(U count) const {
    return nonatomic(x - count);
  }

  /** Atomic add assign operator */
  template <typename U>
  HSHM_INLINE_CROSS_FUN nonatomic &operator+=(U count) {
    x += count;
    return *this;
  }

  /** Atomic subtract assign operator */
  template <typename U>
  HSHM_INLINE_CROSS_FUN nonatomic &operator-=(U count) {
    x -= count;
    return *this;
  }

  /** Atomic assign operator */
  template <typename U>
  HSHM_INLINE_CROSS_FUN nonatomic &operator=(U count) {
    x = count;
    return *this;
  }

  /** Equality check (number) */
  template <typename U>
  HSHM_INLINE_CROSS_FUN bool operator==(U other) const {
    return (other == x);
  }

  /** Inequality check (number) */
  template <typename U>
  HSHM_INLINE_CROSS_FUN bool operator!=(U other) const {
    return (other != x);
  }

  /** Equality check */
  HSHM_INLINE_CROSS_FUN bool operator==(const nonatomic &other) const {
    return (other.x == x);
  }

  /** Inequality check */
  HSHM_INLINE_CROSS_FUN bool operator!=(const nonatomic &other) const {
    return (other.x != x);
  }

  /** Bitwise and */
  template <typename U>
  HSHM_INLINE_CROSS_FUN nonatomic operator&(U other) const {
    return nonatomic(x & other);
  }

  /** Bitwise or */
  template <typename U>
  HSHM_INLINE_CROSS_FUN nonatomic operator|(U other) const {
    return nonatomic(x | other);
  }

  /** Bitwise xor */
  template <typename U>
  HSHM_INLINE_CROSS_FUN nonatomic operator^(U other) const {
    return nonatomic(x ^ other);
  }

  /** Bitwise and assign */
  template <typename U>
  HSHM_INLINE_CROSS_FUN nonatomic &operator&=(U other) {
    x &= other;
    return *this;
  }

  /** Bitwise or assign */
  template <typename U>
  HSHM_INLINE_CROSS_FUN nonatomic &operator|=(U other) {
    x |= other;
    return *this;
  }

  /** Bitwise xor assign */
  template <typename U>
  HSHM_INLINE_CROSS_FUN nonatomic &operator^=(U other) {
    x ^= other;
    return *this;
  }
};

/** A wrapper for CUDA atomic operations */
#if defined(HSHM_ENABLE_CUDA) || defined(HSHM_ENABLE_ROCM)
template <typename T>
struct rocm_atomic {
  T x;

  /** Integer convertion */
  HSHM_INLINE_CROSS_FUN operator T() const { return x; }

  /** Constructor */
  HSHM_INLINE_CROSS_FUN rocm_atomic() = default;

  /** Full constructor */
  template <typename U>
  HSHM_INLINE_CROSS_FUN rocm_atomic(U def) : x(def) {}

  /** Copy constructor */
  HSHM_INLINE_CROSS_FUN rocm_atomic(const rocm_atomic &other) : x(other.x) {}

  /* Move constructor */
  HSHM_INLINE_CROSS_FUN rocm_atomic(rocm_atomic &&other)
      : x(std::move(other.x)) {}

  /** Copy assign operator */
  HSHM_INLINE_CROSS_FUN rocm_atomic &operator=(const rocm_atomic &other) {
    x = other.x;
    return *this;
  }

  /** Move assign operator */
  HSHM_INLINE_CROSS_FUN rocm_atomic &operator=(rocm_atomic &&other) {
    x = std::move(other.x);
    return *this;
  }

  /** Atomic fetch_add wrapper*/
  template <typename U>
  HSHM_INLINE_CROSS_FUN T
  fetch_add(U count, std::memory_order order = std::memory_order_seq_cst) {
    return atomicAdd(&x, count);
  }

  /** Atomic fetch_sub wrapper*/
  template <typename U>
  HSHM_INLINE_CROSS_FUN T
  fetch_sub(U count, std::memory_order order = std::memory_order_seq_cst) {
    return atomicAdd(&x, -count);
  }

  /** Atomic load wrapper */
  HSHM_INLINE_CROSS_FUN T
  load(std::memory_order order = std::memory_order_seq_cst) const {
    return x;
  }

  /** Atomic store wrapper */
  template <typename U>
  HSHM_INLINE_CROSS_FUN void store(
      U count, std::memory_order order = std::memory_order_seq_cst) {
    exchange(count);
  }

  /** Atomic exchange wrapper */
  template <typename U>
  HSHM_INLINE_CROSS_FUN T
  exchange(U count, std::memory_order order = std::memory_order_seq_cst) {
    return atomicExch(&x, count);
  }

  /** Atomic compare exchange weak wrapper */
  template <typename U>
  HSHM_INLINE_CROSS_FUN bool compare_exchange_weak(
      T &expected, U desired,
      std::memory_order order = std::memory_order_seq_cst) {
    return atomicCAS(const_cast<T *>(&x), expected, desired);
  }

  /** Atomic compare exchange strong wrapper */
  template <typename U>
  HSHM_INLINE_CROSS_FUN bool compare_exchange_strong(
      T &expected, U desired,
      std::memory_order order = std::memory_order_seq_cst) {
    return atomicCAS(const_cast<T *>(&x), expected, desired);
  }

  /** Atomic pre-increment operator */
  HSHM_INLINE_CROSS_FUN rocm_atomic &operator++() {
    atomicAdd(&x, 1);
    return *this;
  }

  /** Atomic post-increment operator */
  HSHM_INLINE_CROSS_FUN rocm_atomic operator++(int) { return atomic(x + 1); }

  /** Atomic pre-decrement operator */
  HSHM_INLINE_CROSS_FUN rocm_atomic &operator--() {
    atomicAdd(&x, (T)(-1));
    return (*this);
  }

  /** Atomic post-decrement operator */
  HSHM_INLINE_CROSS_FUN rocm_atomic operator--(int) { return atomic(x - 1); }

  /** Atomic add operator */
  template <typename U>
  HSHM_INLINE_CROSS_FUN rocm_atomic operator+(U count) const {
    return atomicAdd(&x, count);
  }

  /** Atomic subtract operator */
  template <typename U>
  HSHM_INLINE_CROSS_FUN rocm_atomic operator-(U count) const {
    return atomicAdd(&x, (T)(-count));
  }

  /** Atomic add assign operator */
  template <typename U>
  HSHM_INLINE_CROSS_FUN rocm_atomic &operator+=(U count) {
    atomicAdd(&x, count);
    return *this;
  }

  /** Atomic subtract assign operator */
  template <typename U>
  HSHM_INLINE_CROSS_FUN rocm_atomic &operator-=(U count) {
    atomicAdd(&x, -count);
    return *this;
  }

  /** Atomic assign operator */
  template <typename U>
  HSHM_INLINE_CROSS_FUN rocm_atomic &operator=(U count) {
    store(count);
    return *this;
  }

  /** Equality check (number) */
  template <typename U>
  HSHM_INLINE_CROSS_FUN bool operator==(U other) const {
    return atomicCAS(const_cast<T *>(&x), other, other);
  }

  /** Inequality check (number) */
  template <typename U>
  HSHM_INLINE_CROSS_FUN bool operator!=(U other) const {
    return !atomicCAS(const_cast<T *>(&x), other, other);
  }

  /** Equality check */
  HSHM_INLINE_CROSS_FUN bool operator==(const rocm_atomic &other) const {
    return atomicCAS(const_cast<T *>(&x), other.x, other.x);
  }

  /** Inequality check */
  HSHM_INLINE_CROSS_FUN bool operator!=(const rocm_atomic &other) const {
    return !atomicCAS(const_cast<T *>(&x), other.x, other.x);
  }

  /** Bitwise and */
  template <typename U>
  HSHM_INLINE_CROSS_FUN rocm_atomic operator&(U other) const {
    T *addr = const_cast<T *>(&x);
    return atomicAnd(addr, other);
  }

  /** Bitwise or */
  template <typename U>
  HSHM_INLINE_CROSS_FUN rocm_atomic operator|(U other) const {
    T *addr = const_cast<T *>(&x);
    return atomicOr(addr, other);
  }

  /** Bitwise xor */
  template <typename U>
  HSHM_INLINE_CROSS_FUN rocm_atomic operator^(U other) const {
    T *addr = const_cast<T *>(&x);
    return atomicXor(addr, other);
  }

  /** Bitwise and assign */
  template <typename U>
  HSHM_INLINE_CROSS_FUN rocm_atomic &operator&=(U other) {
    atomicAnd(&x, other);
    return *this;
  }

  /** Bitwise or assign */
  template <typename U>
  HSHM_INLINE_CROSS_FUN rocm_atomic &operator|=(U other) {
    atomicOr(&x, other);
    return *this;
  }

  /** Bitwise xor assign */
  template <typename U>
  HSHM_INLINE_CROSS_FUN rocm_atomic &operator^=(U other) {
    atomicXor(&x, other);
    return *this;
  }

  /** Serialization */
  template <typename Ar>
  void serialize(Ar &ar) {
    ar(x);
  }
};
#endif

/** A wrapper around std::atomic */
template <typename T>
struct std_atomic {
  std::atomic<T> x;

  /** Serialization */
  template <typename Ar>
  void serialize(Ar &ar) {
    ar(x);
  }

  /** Integer convertion */
  HSHM_INLINE_CROSS_FUN operator T() const { return x; }

  /** Constructor */
  HSHM_INLINE std_atomic() = default;

  /** Full constructor */
  template <typename U>
  HSHM_INLINE std_atomic(U def) : x(def) {}

  /** Copy constructor */
  HSHM_INLINE std_atomic(const std_atomic &other) : x(other.x.load()) {}

  /* Move constructor */
  HSHM_INLINE std_atomic(std_atomic &&other) : x(other.x.load()) {}

  /** Copy assign operator */
  HSHM_INLINE std_atomic &operator=(const std_atomic &other) {
    x = other.x.load();
    return *this;
  }

  /** Move assign operator */
  HSHM_INLINE std_atomic &operator=(std_atomic &&other) {
    x = other.x.load();
    return *this;
  }

  /** Atomic fetch_add wrapper*/
  template <typename U>
  HSHM_INLINE T fetch_add(U count,
                          std::memory_order order = std::memory_order_seq_cst) {
    return x.fetch_add(count, order);
  }

  /** Atomic fetch_sub wrapper*/
  template <typename U>
  HSHM_INLINE T fetch_sub(U count,
                          std::memory_order order = std::memory_order_seq_cst) {
    return x.fetch_sub(count, order);
  }

  /** Atomic load wrapper */
  HSHM_INLINE T
  load(std::memory_order order = std::memory_order_seq_cst) const {
    return x.load(order);
  }

  /** Atomic store wrapper */
  template <typename U>
  HSHM_INLINE void store(U count,
                         std::memory_order order = std::memory_order_seq_cst) {
    x.store(count, order);
  }

  /** Atomic exchange wrapper */
  template <typename U>
  HSHM_INLINE void exchange(
      U count, std::memory_order order = std::memory_order_seq_cst) {
    x.exchange(count, order);
  }

  /** Atomic compare exchange weak wrapper */
  template <typename U>
  HSHM_INLINE bool compare_exchange_weak(
      T &expected, U desired,
      std::memory_order order = std::memory_order_seq_cst) {
    return x.compare_exchange_weak(expected, desired, order);
  }

  /** Atomic compare exchange strong wrapper */
  template <typename U>
  HSHM_INLINE bool compare_exchange_strong(
      T &expected, U desired,
      std::memory_order order = std::memory_order_seq_cst) {
    return x.compare_exchange_strong(expected, desired, order);
  }

  /** Atomic pre-increment operator */
  HSHM_INLINE std_atomic &operator++() {
    ++x;
    return *this;
  }

  /** Atomic post-increment operator */
  HSHM_INLINE std_atomic operator++(int) { return atomic(x + 1); }

  /** Atomic pre-decrement operator */
  HSHM_INLINE std_atomic &operator--() {
    --x;
    return *this;
  }

  /** Atomic post-decrement operator */
  HSHM_INLINE std_atomic operator--(int) { return atomic(x - 1); }

  /** Atomic add operator */
  template <typename U>
  HSHM_INLINE std_atomic operator+(U count) const {
    return x + count;
  }

  /** Atomic subtract operator */
  template <typename U>
  HSHM_INLINE std_atomic operator-(U count) const {
    return x - count;
  }

  /** Atomic add assign operator */
  template <typename U>
  HSHM_INLINE std_atomic &operator+=(U count) {
    x += count;
    return *this;
  }

  /** Atomic subtract assign operator */
  template <typename U>
  HSHM_INLINE std_atomic &operator-=(U count) {
    x -= count;
    return *this;
  }

  /** Atomic assign operator */
  template <typename U>
  HSHM_INLINE std_atomic &operator=(U count) {
    x.exchange(count);
    return *this;
  }

  /** Equality check (number) */
  template <typename U>
  HSHM_INLINE bool operator==(U other) const {
    return (other == x);
  }

  /** Inequality check (number) */
  template <typename U>
  HSHM_INLINE bool operator!=(U other) const {
    return (other != x);
  }

  /** Equality check */
  HSHM_INLINE bool operator==(const std_atomic &other) const {
    return (other.x == x);
  }

  /** Inequality check */
  HSHM_INLINE bool operator!=(const std_atomic &other) const {
    return (other.x != x);
  }

  /** Bitwise and */
  template <typename U>
  HSHM_INLINE std_atomic operator&(U other) const {
    return x & other;
  }

  /** Bitwise or */
  template <typename U>
  HSHM_INLINE std_atomic operator|(U other) const {
    return x | other;
  }

  /** Bitwise xor */
  template <typename U>
  HSHM_INLINE std_atomic operator^(U other) const {
    return x ^ other;
  }

  /** Bitwise and assign */
  template <typename U>
  HSHM_INLINE std_atomic &operator&=(U other) {
    x &= other;
    return *this;
  }

  /** Bitwise or assign */
  template <typename U>
  HSHM_INLINE std_atomic &operator|=(U other) {
    x |= other;
    return *this;
  }

  /** Bitwise xor assign */
  template <typename U>
  HSHM_INLINE std_atomic &operator^=(U other) {
    x ^= other;
    return *this;
  }
};

#ifdef HSHM_IS_HOST
template <typename T>
using atomic = std_atomic<T>;
#else
template <typename T>
using atomic = rocm_atomic<T>;
#endif

template <typename T, bool is_atomic>
using opt_atomic =
    typename std::conditional<is_atomic, atomic<T>, nonatomic<T>>::type;

}  // namespace hshm::ipc

#endif  // HSHM_INCLUDE_HSHM_TYPES_ATOMIC_H_
