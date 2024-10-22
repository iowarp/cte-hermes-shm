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

#ifndef HERMES_SHM_INCLUDE_HERMES_SHM_TYPES_NUMBERS_H_
#define HERMES_SHM_INCLUDE_HERMES_SHM_TYPES_NUMBERS_H_

#include <cstdint>
#include "hermes_shm/constants/macros.h"

namespace hshm {

#ifndef __CUDA_ARCH__
typedef uint8_t u8;   /**< 8-bit unsigned integer */
typedef uint16_t u16; /**< 16-bit unsigned integer */
typedef uint32_t u32; /**< 32-bit unsigned integer */
typedef uint64_t u64; /**< 64-bit unsigned integer */
typedef int8_t i8;    /**< 8-bit signed integer */
typedef int16_t i16;  /**< 16-bit signed integer */
typedef int32_t i32;  /**< 32-bit signed integer */
typedef int64_t i64;  /**< 64-bit signed integer */
typedef float f32;    /**< 32-bit float */
typedef double f64;   /**< 64-bit float */

typedef uint8_t s_u8;   /**< 8-bit unsigned integer */
typedef uint16_t s_u16; /**< 16-bit unsigned integer */
typedef uint32_t s_u32; /**< 32-bit unsigned integer */
typedef uint64_t s_u64; /**< 64-bit unsigned integer */
typedef int8_t s_i8;    /**< 8-bit signed integer */
typedef int16_t s_i16;  /**< 16-bit signed integer */
typedef int32_t s_i32;  /**< 32-bit signed integer */
typedef int64_t s_i64;  /**< 64-bit signed integer */
typedef float s_f32;    /**< 32-bit float */
typedef double s_f64;   /**< 64-bit float */
#else
typedef unsigned char u8;   /**< 8-bit unsigned integer */
typedef unsigned short u16; /**< 16-bit unsigned integer */
typedef unsigned u32; /**< 32-bit unsigned integer */
typedef long long unsigned u64; /**< 64-bit unsigned integer */
typedef char i8;    /**< 8-bit signed integer */
typedef short i16;  /**< 16-bit signed integer */
typedef int i32;  /**< 32-bit signed integer */
typedef long long int i64;  /**< 64-bit signed integer */
typedef float f32;    /**< 32-bit float */
typedef double f64;   /**< 64-bit float */

typedef int s_u8;   /**< 8-bit unsigned integer */
typedef int s_u16; /**< 16-bit unsigned integer */
typedef int s_u32; /**< 32-bit unsigned integer */
typedef int s_u64; /**< 64-bit unsigned integer */
typedef int s_i8;    /**< 8-bit signed integer */
typedef int s_i16;  /**< 16-bit signed integer */
typedef int s_i32;  /**< 32-bit signed integer */
typedef int s_i64;  /**< 64-bit signed integer */
typedef float s_f32;    /**< 32-bit float */
typedef double s_f64;   /**< 64-bit float */
#endif

}  // namespace hshm

#endif  // HERMES_SHM_INCLUDE_HERMES_SHM_TYPES_NUMBERS_H_
