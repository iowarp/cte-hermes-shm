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

#ifndef HERMES_MACROS_H
#define HERMES_MACROS_H

#include "settings.h"

/** Bytes -> Bytes */
#ifndef BYTES
#define BYTES(n) (size_t)((n) * (((size_t)1) << 0))
#endif

/** KILOBYTES -> Bytes */
#ifndef KILOBYTES
#define KILOBYTES(n) (size_t)((n) * (((size_t)1) << 10))
#endif

/** MEGABYTES -> Bytes */
#ifndef MEGABYTES
#define MEGABYTES(n) (size_t)((n) * (((size_t)1) << 20))
#endif

/** GIGABYTES -> Bytes */
#ifndef GIGABYTES
#define GIGABYTES(n) (size_t)((n) * (((size_t)1) << 30))
#endif

/** TERABYTES -> Bytes */
#ifndef TERABYTES
#define TERABYTES(n) (size_t)((n) * (((size_t)1) << 40))
#endif

/** PETABYTES -> Bytes */
#ifndef PETABYTES
#define PETABYTES(n) (size_t)((n) * (((size_t)1) << 50))
#endif

/** Seconds to nanoseconds */
#ifndef SECONDS
#define SECONDS(n) (size_t)((n) * 1000000000)
#endif

/** Milliseconds to nanoseconds */
#ifndef MILLISECONDS
#define MILLISECONDS(n) (size_t)((n) * 1000000)
#endif

/** Microseconds to nanoseconds */
#ifndef MICROSECONDS
#define MICROSECONDS(n) (size_t)((n) * 1000)
#endif

/** Nanoseconds to nanoseconds */
#ifndef NANOSECONDS
#define NANOSECONDS(n) (size_t)(n)
#endif

/**
 * Remove parenthesis surrounding "X" if it has parenthesis
 * Used for helper macros which take templated types as parameters
 * E.g., let's say we have:
 *
 * #define HELPER_MACRO(T) TYPE_UNWRAP(T)
 * HELPER_MACRO( (std::vector<std::pair<int, int>>) )
 * will return std::vector<std::pair<int, int>> without the parenthesis
 * */
#define TYPE_UNWRAP(X) ESC(ISH X)
#define ISH(...) ISH __VA_ARGS__
#define ESC(...) ESC_(__VA_ARGS__)
#define ESC_(...) VAN##__VA_ARGS__
#define VANISH
#define __TU(X) TYPE_UNWRAP(X)

/** Includes for CUDA and ROCm */
#ifdef HERMES_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

#ifdef HERMES_ENABLE_ROCM
#include <hip/hip_runtime.h>
#endif

/** Macros for CUDA functions */
#if defined(HERMES_ENABLE_CUDA) || defined(HERMES_ENABLE_ROCM)
#define ROCM_HOST __host__
#define ROCM_DEVICE __device__
#define ROCM_HOST_DEVICE __device__ __host__
#else
#define ROCM_HOST_DEVICE
#define ROCM_HOST
#define ROCM_DEVICE
#endif

/** Error checking for ROCM */
#define HIP_ERROR_CHECK(X)                                                  \
  do {                                                                      \
    if (X != hipSuccess) {                                                  \
      hipError_t hipErr = hipGetLastError();                                \
      HELOG(kFatal, "HIP Error {}: {}", hipErr, hipGetErrorString(hipErr)); \
    }                                                                       \
  } while (false)

/**
 * Ensure that the compiler ALWAYS inlines a particular function.
 * */
#ifndef HSHM_DEBUG
#define HSHM_INLINE inline __attribute__((always_inline))
#else
#define HSHM_INLINE __attribute__((noinline))
#endif

/** Function decorators */
#define HSHM_REG_FUN ROCM_HOST
#define HSHM_HOST_FUN ROCM_HOST
#define HSHM_GPU_FUN ROCM_DEVICE
#define HSHM_CROSS_FUN ROCM_HOST_DEVICE

/** Function content selector for CUDA */
#ifdef __CUDA_ARCH__
#define HSHM_IS_CUDA_GPU
#endif

/** Function content selector for ROCm */
#if __HIP_DEVICE_COMPILE__
#define HSHM_IS_ROCM_GPU
#endif

/** Function internals */
#if defined(HSHM_IS_CUDA_GPU) || defined(HSHM_IS_ROCM_GPU)
#define HSHM_IS_GPU
#else
#define HSHM_IS_HOST
#endif

/** Macro for inline function */
#define HSHM_INLINE_CROSS_FUN HSHM_INLINE HSHM_CROSS_FUN
#define HSHM_INLINE_GPU ROCM_DEVICE HSHM_INLINE
#define HSHM_INLINE_HOST ROCM_HOST HSHM_INLINE

/** Bitfield macros */
#define MARK_FIRST_BIT_MASK(T) ((T)1 << (sizeof(T) * 8 - 1))
#define MARK_FIRST_BIT(T, X) ((X) | MARK_FIRST_BIT_MASK(T))
#define IS_FIRST_BIT_MARKED(T, X) ((X) & MARK_FIRST_BIT_MASK(T))
#define UNMARK_FIRST_BIT(T, X) ((X) & ~MARK_FIRST_BIT_MASK(T))

/** Class constant macro */
#define CLS_CONST static inline constexpr const

/** Class constant macro */
#define GLOBAL_CONST inline const

/** Namespace definitions */
namespace hshm {}
namespace hshm::ipc {}
namespace hipc = hshm::ipc;

/** The name of the current device */
#ifdef HSHM_IS_HOST
#define kCurrentDevice "cpu"
#else
#define kCurrentDevice "gpu"
#endif

/***************************************************
 * CUSTOM SETTINGS FOR ALLOCATORS + THREAD MODELS
 * ************************************************* */
/** Define the root allocator class */
#ifndef HSHM_ROOT_ALLOC_T
#define HSHM_ROOT_ALLOC_T hipc::StackAllocator
#endif
#define HSHM_ROOT_ALLOC \
  HERMES_MEMORY_MANAGER->template GetRootAllocator<HSHM_ROOT_ALLOC_T>()

/** Define the default allocator class */
#ifndef HSHM_DEFAULT_ALLOC_T
#define HSHM_DEFAULT_ALLOC_T hipc::MallocAllocator
#endif
#define HSHM_DEFAULT_ALLOC \
  HERMES_MEMORY_MANAGER->template GetDefaultAllocator<HSHM_DEFAULT_ALLOC_T>()

/** Define the default thread model class */
// CUDA
#if defined(HSHM_IS_CUDA_GPU)
#define HSHM_DEFAULT_THREAD_MODEL hshm::thread::Cuda
#endif
// ROCM
#if defined(HSHM_IS_ROCM_GPU)
#define HSHM_DEFAULT_THREAD_MODEL hshm::thread::Rocm
#endif
// CPU
#ifndef HSHM_DEFAULT_THREAD_MODEL
#define HSHM_DEFAULT_THREAD_MODEL hshm::thread::Pthread
#endif

/** Default memory context object */
#define HSHM_DEFAULT_MEM_CTX \
  {}

/** Compatability hack for static_assert */
template <bool TRUTH, typename T = int>
class assert_hack {
 public:
  CLS_CONST bool value = TRUTH;
};

/** A hack for static asserts */
#define STATIC_ASSERT(TRUTH, MSG, T) \
  static_assert(assert_hack<TRUTH, __TU(T)>::value, MSG)

#endif  // HERMES_MACROS_H
