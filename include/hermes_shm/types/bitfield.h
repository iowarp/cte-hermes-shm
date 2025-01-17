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

#ifndef HSHM_INCLUDE_HSHM_TYPES_BITFIELD_H_
#define HSHM_INCLUDE_HSHM_TYPES_BITFIELD_H_

#include <cstdint>

#include "hermes_shm/constants/macros.h"
#include "hermes_shm/types/atomic.h"

namespace hshm {

#define BIT_OPT(T, n) (((T)1) << n)
#define ALL_BITS(T) (~((T)0))

/**
 * A generic bitfield template
 * */
template <typename T = u32, bool ATOMIC = false>
struct bitfield {
  hipc::opt_atomic<T, ATOMIC> bits_;

  HSHM_INLINE_CROSS_FUN bitfield() : bits_(0) {}

  HSHM_INLINE_CROSS_FUN explicit bitfield(T mask) : bits_(mask) {}

  HSHM_INLINE_CROSS_FUN void SetBits(T mask) { bits_ |= mask; }

  HSHM_INLINE_CROSS_FUN void UnsetBits(T mask) { bits_ &= ~mask; }

  HSHM_INLINE_CROSS_FUN T Any(T mask) const { return bits_ & mask; }

  HSHM_INLINE_CROSS_FUN T All(T mask) const { return Any(mask) == mask; }

  HSHM_INLINE_CROSS_FUN void CopyBits(bitfield field, T mask) {
    bits_ &= (field.bits_ & mask);
  }

  HSHM_INLINE_CROSS_FUN void Clear() { bits_ = 0; }

  HSHM_INLINE_CROSS_FUN static T MakeMask(int start, int length) {
    return ((((T)1) << length) - 1) << start;
  }

  template <typename Ar>
  void serialize(Ar &ar) {
    ar & bits_;
  }
};
typedef bitfield<u8> bitfield8_t;
typedef bitfield<u16> bitfield16_t;
typedef bitfield<u32> bitfield32_t;

typedef bitfield<u8, true> abitfield8_t;
typedef bitfield<u16, true> abitfield16_t;
typedef bitfield<u32, true> abitfield32_t;

/**
 * A helper type needed for std::conditional
 * */
template <size_t LEN>
struct len_bits {
  static constexpr size_t value = LEN;
};

/**
 * A generic bitfield template
 * */
template <size_t NUM_BITS,
          typename LEN = typename std::conditional<
              ((NUM_BITS % 32 == 0) && (NUM_BITS > 0)),
              len_bits<(NUM_BITS / 32)>, len_bits<(NUM_BITS / 32) + 1>>::type>
struct big_bitfield {
  bitfield32_t bits_[LEN::value];

  HSHM_INLINE_CROSS_FUN big_bitfield() : bits_() {}

  HSHM_INLINE_CROSS_FUN size_t size() const { return LEN::value; }

  HSHM_INLINE_CROSS_FUN void SetBits(int start, int length) {
    int bf_idx = start / 32;
    int bf_idx_count = 32 - bf_idx;
    int rem = length;
    while (rem) {
      bits_[bf_idx].SetBits(bitfield32_t::MakeMask(start, bf_idx_count));
      rem -= bf_idx_count;
      bf_idx += 1;
      if (rem >= 32) {
        bf_idx_count = 32;
      } else {
        bf_idx_count = rem;
      }
    }
  }

  HSHM_INLINE_CROSS_FUN void UnsetBits(int start, int length) {
    int bf_idx = start / 32;
    int bf_idx_count = 32 - bf_idx;
    int rem = length;
    while (rem) {
      bits_[bf_idx].SetBits(bitfield32_t::MakeMask(start, bf_idx_count));
      rem -= bf_idx_count;
      bf_idx += 1;
      if (rem >= 32) {
        bf_idx_count = 32;
      } else {
        bf_idx_count = rem;
      }
    }
  }

  HSHM_INLINE_CROSS_FUN bool Any(int start, int length) const {
    int bf_idx = start / 32;
    int bf_idx_count = 32 - bf_idx;
    int rem = length;
    while (rem) {
      if (bits_[bf_idx].Any(bitfield32_t::MakeMask(start, bf_idx_count))) {
        return true;
      }
      rem -= bf_idx_count;
      bf_idx += 1;
      if (rem >= 32) {
        bf_idx_count = 32;
      } else {
        bf_idx_count = rem;
      }
    }
    return false;
  }

  HSHM_INLINE_CROSS_FUN bool All(int start, int length) const {
    int bf_idx = start / 32;
    int bf_idx_count = 32 - bf_idx;
    int rem = length;
    while (rem) {
      if (!bits_[bf_idx].All(bitfield32_t::MakeMask(start, bf_idx_count))) {
        return false;
      }
      rem -= bf_idx_count;
      bf_idx += 1;
      if (rem >= 32) {
        bf_idx_count = 32;
      } else {
        bf_idx_count = rem;
      }
    }
    return true;
  }

  HSHM_INLINE_CROSS_FUN void Clear() {
    memset((void *)bits_, 0, sizeof(bitfield32_t) * LEN::value);
  }
};

}  // namespace hshm

#endif  // HSHM_INCLUDE_HSHM_TYPES_BITFIELD_H_
