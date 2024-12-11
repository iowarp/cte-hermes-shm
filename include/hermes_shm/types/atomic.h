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

#ifndef HERMES_INCLUDE_HERMES_TYPES_ATOMIC_H_
#define HERMES_INCLUDE_HERMES_TYPES_ATOMIC_H_

#include <atomic>
#include <hermes_shm/constants/macros.h>
#include <type_traits>
#include "numbers.h"
#ifdef HERMES_ENABLE_CUDA
#include <cuda/atomic>
#endif

namespace hshm::ipc {

/** Provides the API of an atomic, without being atomic */
template<typename T>
struct nonatomic {
  T x;

  /** Serialization */
  template <typename Ar>
  void serialize(Ar &ar) {
    ar(x);
  }

  /** Constructor */
  HSHM_INLINE_CROSS nonatomic() = default;

  /** Full constructor */
  HSHM_INLINE_CROSS explicit nonatomic(T def) : x(def) {}

  /** Copy constructor */
  HSHM_INLINE_CROSS nonatomic(const nonatomic &other) : x(other.x) {}

  /* Move constructor */
  HSHM_INLINE_CROSS nonatomic(nonatomic &&other) : x(std::move(other.x)) {}

  /** Copy assign operator */
  HSHM_INLINE_CROSS nonatomic& operator=(const nonatomic &other) {
    x = other.x;
    return *this;
  }

  /** Move assign operator */
  HSHM_INLINE_CROSS nonatomic& operator=(nonatomic &&other) {
    x = std::move(other.x);
    return *this;
  }

  /** Atomic fetch_add wrapper*/
  HSHM_INLINE_CROSS T fetch_add(
      T count, std::memory_order order = std::memory_order_seq_cst) {
    (void) order;
    T orig_x = x;
    x += count;
    return orig_x;
  }

  /** Atomic fetch_sub wrapper*/
  HSHM_INLINE_CROSS T fetch_sub(
      T count, std::memory_order order = std::memory_order_seq_cst) {
    (void) order;
    T orig_x = x;
    x -= count;
    return orig_x;
  }

  /** Atomic load wrapper */
  HSHM_INLINE_CROSS T load(
      std::memory_order order = std::memory_order_seq_cst) const {
    (void) order;
    return x;
  }

  /** Atomic exchange wrapper */
  HSHM_INLINE_CROSS void exchange(
      T count, std::memory_order order = std::memory_order_seq_cst) {
    (void) order;
    x = count;
  }

  /** Atomic compare exchange weak wrapper */
  HSHM_INLINE_CROSS bool compare_exchange_weak(T& expected, T desired,
                                                   std::memory_order order =
                                                   std::memory_order_seq_cst) {
    (void) expected; (void) order;
    x = desired;
    return true;
  }

  /** Atomic compare exchange strong wrapper */
  HSHM_INLINE_CROSS bool compare_exchange_strong(T& expected, T desired,
                                                     std::memory_order order =
                                                     std::memory_order_seq_cst) {
    (void) expected; (void) order;
    x = desired;
    return true;
  }

  /** Atomic pre-increment operator */
  HSHM_INLINE_CROSS nonatomic& operator++() {
    ++x;
    return *this;
  }

  /** Atomic post-increment operator */
  HSHM_INLINE_CROSS nonatomic operator++(int) {
    return atomic(x+1);
  }

  /** Atomic pre-decrement operator */
  HSHM_INLINE_CROSS nonatomic& operator--() {
    --x;
    return *this;
  }

  /** Atomic post-decrement operator */
  HSHM_INLINE_CROSS nonatomic operator--(int) {
    nonatomic orig_x(x);
    --x;
    return orig_x;
  }

  /** Atomic add operator */
  HSHM_INLINE_CROSS nonatomic operator+(T count) const {
    return nonatomic(x + count);
  }

  /** Atomic subtract operator */
  HSHM_INLINE_CROSS nonatomic operator-(T count) const {
    return nonatomic(x - count);
  }

  /** Atomic add assign operator */
  HSHM_INLINE_CROSS nonatomic& operator+=(T count) {
    x += count;
    return *this;
  }

  /** Atomic subtract assign operator */
  HSHM_INLINE_CROSS nonatomic& operator-=(T count) {
    x -= count;
    return *this;
  }

  /** Atomic assign operator */
  HSHM_INLINE_CROSS nonatomic& operator=(T count) {
    x = count;
    return *this;
  }

  /** Equality check */
  HSHM_INLINE_CROSS bool operator==(const nonatomic &other) const {
    return (other.x == x);
  }

  /** Inequality check */
  HSHM_INLINE_CROSS bool operator!=(const nonatomic &other) const {
    return (other.x != x);
  }
};

/** A wrapper for CUDA atomic operations */
#ifdef HSHM_IS_GPU
template<typename T>
struct cuda_atomic {
  T x;

  /** Constructor */
  HSHM_INLINE_GPU cuda_atomic() = default;

  /** Full constructor */
  HSHM_INLINE_GPU explicit cuda_atomic(T def) : x(def) {}

  /** Copy constructor */
  HSHM_INLINE_GPU cuda_atomic(const cuda_atomic &other) : x(other.x) {}

  /* Move constructor */
  HSHM_INLINE_GPU cuda_atomic(cuda_atomic &&other) : x(std::move(other.x)) {}

  /** Copy assign operator */
  HSHM_INLINE_GPU cuda_atomic& operator=(const cuda_atomic &other) {
    x = other.x;
    return *this;
  }

  /** Move assign operator */
  HSHM_INLINE_GPU cuda_atomic& operator=(cuda_atomic &&other) {
    x = std::move(other.x);
    return *this;
  }

  /** Atomic fetch_add wrapper*/
  HSHM_INLINE_GPU T fetch_add(T count) {
    return atomicAdd(&x, count);
  }

  /** Atomic fetch_sub wrapper*/
  HSHM_INLINE_GPU T fetch_sub(T count) {
    return atomicAdd(&x, -count);
  }

  /** Atomic load wrapper */
  HSHM_INLINE_GPU T load() const {
    return x;
  }

  /** Atomic store wrapper */
  HSHM_INLINE_GPU void store(T count) {
    exchange(count);
  }

  /** Atomic exchange wrapper */
  HSHM_INLINE_GPU T exchange(T count) {
    return atomicExch(&x, count);
  }

  /** Atomic compare exchange weak wrapper */
  HSHM_INLINE_GPU bool compare_exchange_weak(T& expected, T desired) {
    return atomicCAS(&x, expected, desired);
  }

  /** Atomic compare exchange strong wrapper */
  HSHM_INLINE_GPU bool compare_exchange_strong(T& expected, T desired) {
    return atomicCAS(&x, expected, desired);
  }

  /** Atomic pre-increment operator */
  HSHM_INLINE_GPU cuda_atomic& operator++() {
    atomicAdd(&x, 1);
    return *this;
  }

  /** Atomic post-increment operator */
  HSHM_INLINE_GPU cuda_atomic operator++(int) {
    return atomic(x + 1);
  }

  /** Atomic pre-decrement operator */
  HSHM_INLINE_GPU cuda_atomic& operator--() {
    atomicAdd(&x, (T)(-1));
    return (*this);
  }

  /** Atomic post-decrement operator */
  HSHM_INLINE_GPU cuda_atomic operator--(int) {
    return atomic(x - 1);
  }

  /** Atomic add operator */
  HSHM_INLINE_GPU cuda_atomic operator+(T count) const {
    return atomicAdd(&x, count);
  }

  /** Atomic subtract operator */
  HSHM_INLINE_GPU cuda_atomic operator-(T count) const {
    return atomicAdd(&x, (T)(-count));
  }

  /** Atomic add assign operator */
  HSHM_INLINE_GPU cuda_atomic& operator+=(T count) {
    atomicAdd(&x, count);
    return *this;
  }

  /** Atomic subtract assign operator */
  HSHM_INLINE_GPU cuda_atomic& operator-=(T count) {
    atomicAdd(&x, -count);
    return *this;
  }

  /** Atomic assign operator */
  HSHM_INLINE_GPU cuda_atomic& operator=(T count) {
    store(count);
    return *this;
  }

  /** Equality check */
  HSHM_INLINE_GPU bool operator==(const cuda_atomic &other) const {
    return atomicCAS(&x, other.x, other.x);
  }

  /** Inequality check */
  HSHM_INLINE_GPU bool operator!=(const cuda_atomic &other) const {
    return !atomicCAS(&x, other.x, other.x);
  }
};
#endif

/** A wrapper around std::atomic */
template<typename T>
struct std_atomic {
  std::atomic<T> x;

  /** Serialization */
  template <typename Ar>
  void serialize(Ar &ar) {
    ar(x);
  }

  /** Constructor */
  HSHM_INLINE std_atomic() = default;

  /** Full constructor */
  HSHM_INLINE explicit std_atomic(T def) : x(def) {}

  /** Copy constructor */
  HSHM_INLINE std_atomic(const std_atomic &other) : x(other.x) {}

  /* Move constructor */
  HSHM_INLINE std_atomic(std_atomic &&other) : x(std::move(other.x)) {}

  /** Copy assign operator */
  HSHM_INLINE std_atomic& operator=(const std_atomic &other) {
    x = other.x;
    return *this;
  }

  /** Move assign operator */
  HSHM_INLINE std_atomic& operator=(std_atomic &&other) {
    x = std::move(other.x);
    return *this;
  }

  /** Atomic fetch_add wrapper*/
  HSHM_INLINE T fetch_add(
      T count, std::memory_order order = std::memory_order_seq_cst) {
    return x.fetch_add(count, order);
  }

  /** Atomic fetch_sub wrapper*/
  HSHM_INLINE T fetch_sub(
      T count, std::memory_order order = std::memory_order_seq_cst) {
    return x.fetch_sub(count, order);
  }

  /** Atomic load wrapper */
  HSHM_INLINE T load(
      std::memory_order order = std::memory_order_seq_cst) const {
    return x.load(order);
  }

  /** Atomic store wrapper */
  HSHM_INLINE void store(T count,
                                std::memory_order order = std::memory_order_seq_cst) {
    x.store(count, order);
  }

  /** Atomic exchange wrapper */
  HSHM_INLINE void exchange(
      T count, std::memory_order order = std::memory_order_seq_cst) {
    x.exchange(count, order);
  }

  /** Atomic compare exchange weak wrapper */
  HSHM_INLINE bool compare_exchange_weak(T& expected, T desired,
                                                std::memory_order order =
                                                std::memory_order_seq_cst) {
    return x.compare_exchange_weak(expected, desired, order);
  }

  /** Atomic compare exchange strong wrapper */
  HSHM_INLINE bool compare_exchange_strong(T& expected, T desired,
                                                  std::memory_order order =
                                                  std::memory_order_seq_cst) {
    return x.compare_exchange_strong(expected, desired, order);
  }

  /** Atomic pre-increment operator */
  HSHM_INLINE std_atomic& operator++() {
    ++x;
    return *this;
  }

  /** Atomic post-increment operator */
  HSHM_INLINE std_atomic operator++(int) {
    return atomic(x + 1);
  }

  /** Atomic pre-decrement operator */
  HSHM_INLINE std_atomic& operator--() {
    --x;
    return *this;
  }

  /** Atomic post-decrement operator */
  HSHM_INLINE std_atomic operator--(int) {
    return atomic(x - 1);
  }

  /** Atomic add operator */
  HSHM_INLINE std_atomic operator+(T count) const {
    return x + count;
  }

  /** Atomic subtract operator */
  HSHM_INLINE std_atomic operator-(T count) const {
    return x - count;
  }

  /** Atomic add assign operator */
  HSHM_INLINE std_atomic& operator+=(T count) {
    x += count;
    return *this;
  }

  /** Atomic subtract assign operator */
  HSHM_INLINE std_atomic& operator-=(T count) {
    x -= count;
    return *this;
  }

  /** Atomic assign operator */
  HSHM_INLINE std_atomic& operator=(T count) {
    x.exchange(count);
    return *this;
  }

  /** Equality check */
  HSHM_INLINE bool operator==(const std_atomic &other) const {
    return (other.x == x);
  }

  /** Inequality check */
  HSHM_INLINE bool operator!=(const std_atomic &other) const {
    return (other.x != x);
  }
};

#ifdef HSHM_IS_HOST
template<typename T>
using atomic = std_atomic<T>;
#else
template<typename T>
using atomic = cuda_atomic<T>;
#endif

template<typename T, bool is_atomic>
using opt_atomic = typename std::conditional<is_atomic, atomic<T>, nonatomic<T>>::type;

}  // namespace hshm::ipc

#endif  // HERMES_INCLUDE_HERMES_TYPES_ATOMIC_H_
