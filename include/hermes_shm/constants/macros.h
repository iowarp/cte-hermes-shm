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
#define ESC_(...) VAN ## __VA_ARGS__
#define VANISH

/** Macros for CUDA functions */
#ifdef HERMES_ENABLE_CUDA
#include <cuda_runtime.h>
#define CUDA_HOST __host__
#define CUDA_DEVICE __device__
#define CUDA_HOST_DEVICE __device__ __host__
#else
#define CUDA_HOST_DEVICE
#define CUDA_HOST
#define CUDA_DEVICE
#endif

/**
 * Ensure that the compiler ALWAYS inlines a particular function.
 * */
#define HSHM_ALWAYS_INLINE \
  inline __attribute__((always_inline))

/** Macro for functions */
#define HSHM_REG_FUN CUDA_HOST
#define HSHM_HOST_FUN CUDA_HOST
#define HSHM_GPU_FUN CUDA_DEVICE
#define HSHM_CROSS_FUN CUDA_HOST_DEVICE
#ifdef __CUDA_ARCH__
#define HSHM_CROSS_VAR CUDA_DEVICE
#else
#define HSHM_CROSS_VAR
#endif

/** Macro for inline function */
#define HSHM_INLINE_CROSS_FUN HSHM_ALWAYS_INLINE HSHM_CROSS_FUN
#define HSHM_INLINE_GPU_FUN CUDA_DEVICE HSHM_ALWAYS_INLINE
#define HSHM_INLINE_HOST_FUN CUDA_HOST HSHM_ALWAYS_INLINE

/** Bitfield macros */
#define MARK_FIRST_BIT_MASK(T) ((T)1 << (sizeof(T) * 8 - 1))
#define MARK_FIRST_BIT(T, X) ((X) | MARK_FIRST_BIT_MASK(T))
#define IS_FIRST_BIT_MARKED(T, X) ((X) & MARK_FIRST_BIT_MASK(T))
#define UNMARK_FIRST_BIT(T, X) ((X) & ~MARK_FIRST_BIT_MASK(T))

/** Class constant macro */
#define CLS_CONST static inline constexpr const

/** Class constant macro */
#define GLOBAL_CONST static inline const HSHM_CROSS_VAR

namespace hshm {}
namespace hshm::ipc {}
namespace hipc = hshm::ipc;

#ifndef __CUDA_ARCH__
GLOBAL_CONST char *kCurrentDevice = "cpu";
#else
GLOBAL_CONST char *kCurrentDevice = "gpu";
#endif

#ifndef HSHM_DEFAULT_ALLOC
#define HSHM_DEFAULT_ALLOC hipc::Allocator
#endif

#endif  // HERMES_MACROS_H
