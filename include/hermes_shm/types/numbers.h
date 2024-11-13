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

typedef char byte; /**< Signed char */
typedef unsigned char ubyte; /**< Unsigned char */
typedef short short_int; /**< Signed int */
typedef unsigned short short_uint; /**< Unsigned int */
typedef int reg_int; /**< Signed int */
typedef unsigned reg_uint; /**< Unsigned int */
typedef long long big_int; /**< Long long */
typedef unsigned long long big_uint; /**< Unsigned long long */

struct ThreadId {
  hshm::u64 tid_;

  HSHM_INLINE_CROSS_FUN
  ThreadId() = default;

  HSHM_INLINE_CROSS_FUN
  explicit ThreadId(hshm::u64 tid) : tid_(tid) {}

  HSHM_INLINE_CROSS_FUN
  static ThreadId GetNull() {
    return ThreadId{(hshm::u64)-1};
  }

  HSHM_INLINE_CROSS_FUN
  bool IsNull() const {
    return tid_ == (hshm::u64)-1;
  }

  HSHM_INLINE_CROSS_FUN
  void SetNull() {
    tid_ = (hshm::u64)-1;
  }

  HSHM_INLINE_CROSS_FUN
  bool operator==(const ThreadId &other) const {
    return tid_ == other.tid_;
  }

  HSHM_INLINE_CROSS_FUN
  bool operator!=(const ThreadId &other) const {
    return tid_ != other.tid_;
  }

  HSHM_INLINE_CROSS_FUN
  bool operator<(const ThreadId &other) const {
    return tid_ < other.tid_;
  }

  HSHM_INLINE_CROSS_FUN
  bool operator>(const ThreadId &other) const {
    return tid_ > other.tid_;
  }

  HSHM_INLINE_CROSS_FUN
  bool operator<=(const ThreadId &other) const {
    return tid_ <= other.tid_;
  }

  HSHM_INLINE_CROSS_FUN
  bool operator>=(const ThreadId &other) const {
    return tid_ >= other.tid_;
  }
};

#ifndef HERMES_ENABLE_CUDA
typedef i16 min_i16;
typedef i32 min_i32;
typedef i64 min_i64;

typedef u16 min_u16;
typedef u32 min_u32;
typedef u64 min_u64;
#else
typedef reg_int min_i16;
typedef reg_int min_i32;
typedef big_uint min_i64;

typedef reg_uint min_u16;
typedef reg_uint min_u32;
typedef big_uint min_u64;
#endif

}  // namespace hshm

#endif  // HERMES_SHM_INCLUDE_HERMES_SHM_TYPES_NUMBERS_H_
