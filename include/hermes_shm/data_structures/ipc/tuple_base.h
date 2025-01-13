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

#ifndef HSHM_INCLUDE_HSHM_DATA_STRUCTURES_TupleBase_H_
#define HSHM_INCLUDE_HSHM_DATA_STRUCTURES_TupleBase_H_

#include <utility>

#include "hermes_shm/types/argpack.h"
#include "hermes_shm/types/real_number.h"

namespace hshm {

/** The null container wrapper */
template <typename T>
using NullWrap = T;

/** Recurrence used to create argument pack */
template <template <typename> typename Wrap, size_t idx,
          typename T = EndTemplateRecurrence, typename... Args>
struct TupleBaseRecur {
  Wrap<T> arg_;                                  /**< The element stored */
  TupleBaseRecur<Wrap, idx + 1, Args...> recur_; /**< Remaining args */

  /** Default constructor */
  HSHM_CROSS_FUN TupleBaseRecur() = default;

  /** Constructor. Const reference. */
  HSHM_CROSS_FUN
  explicit TupleBaseRecur(const T &arg, Args &&...args)
      : arg_(std::forward<T>(arg)), recur_(std::forward<Args>(args)...) {}

  /** Constructor. Lvalue reference. */
  HSHM_CROSS_FUN
  explicit TupleBaseRecur(T &arg, Args &&...args)
      : arg_(std::forward<T>(arg)), recur_(std::forward<Args>(args)...) {}

  /** Constructor. Rvalue reference. */
  HSHM_CROSS_FUN
  explicit TupleBaseRecur(T &&arg, Args &&...args)
      : arg_(std::forward<T>(arg)), recur_(std::forward<Args>(args)...) {}

  /** Move constructor */
  HSHM_CROSS_FUN
  TupleBaseRecur(TupleBaseRecur &&other) noexcept
      : arg_(std::move(other.arg_)), recur_(std::move(other.recur_)) {}

  /** Move assignment operator */
  HSHM_CROSS_FUN
  TupleBaseRecur &operator=(TupleBaseRecur &&other) {
    if (this != &other) {
      arg_ = std::move(other.arg_);
      recur_ = std::move(other.recur_);
    }
    return *this;
  }

  /** Copy constructor */
  HSHM_CROSS_FUN
  TupleBaseRecur(const TupleBaseRecur &other)
      : arg_(other.arg_), recur_(other.recur_) {}

  /** Copy assignment operator */
  HSHM_CROSS_FUN
  TupleBaseRecur &operator=(const TupleBaseRecur &other) {
    if (this != &other) {
      arg_ = other.arg_;
      recur_ = other.recur_;
    }
    return *this;
  }

  /** Solidification constructor */
  template <typename... CArgs>
  HSHM_CROSS_FUN explicit TupleBaseRecur(ArgPack<CArgs...> &&other)
      : arg_(other.template Forward<idx>()),
        recur_(std::forward<ArgPack<CArgs...>>(other)) {}

  /** Get reference to internal variable (only if tuple) */
  template <size_t i>
  HSHM_CROSS_FUN constexpr auto &Get() {
    if constexpr (i == idx) {
      return arg_;
    } else {
      return recur_.template Get<i>();
    }
  }

  /** Get reference to internal variable (only if tuple, const) */
  template <size_t i>
  HSHM_CROSS_FUN constexpr auto &Get() const {
    if constexpr (i == idx) {
      return arg_;
    } else {
      return recur_.template Get<i>();
    }
  }
};

/** Terminator of the TupleBase recurrence */
template <template <typename> typename Wrap, size_t idx>
struct TupleBaseRecur<Wrap, idx, EndTemplateRecurrence> {
  /** Default constructor */
  TupleBaseRecur() = default;

  /** Solidification constructor */
  template <typename... CArgs>
  HSHM_CROSS_FUN explicit TupleBaseRecur(ArgPack<CArgs...> &&other) {}

  /** Getter */
  template <size_t i>
  HSHM_CROSS_FUN void Get() {
    // TODO(llogan): fix assert
    STATIC_ASSERT(true, "(Get) TupleBase index outside of range", void);
  }

  /** Getter */
  template <size_t i>
  HSHM_CROSS_FUN void Get() const {
    // TODO(llogan): fix assert
    STATIC_ASSERT(true, "(Get) TupleBase index outside of range", void);
  }
};

/** Used to semantically pack arguments */
template <bool is_argpack, template <typename> typename Wrap, typename... Args>
struct TupleBase {
  /** Variable argument pack */
  TupleBaseRecur<Wrap, 0, Args...> recur_;

  /** Default constructor */
  HSHM_CROSS_FUN TupleBase() = default;

  /** General Constructor. */
  template <typename... CArgs>
  HSHM_CROSS_FUN explicit TupleBase(Args &&...args)
      : recur_(std::forward<Args>(args)...) {}

  /** Move constructor */
  HSHM_CROSS_FUN TupleBase(TupleBase &&other) noexcept
      : recur_(std::move(other.recur_)) {}

  /** Move assignment operator */
  HSHM_CROSS_FUN TupleBase &operator=(TupleBase &&other) noexcept {
    if (this != &other) {
      recur_ = std::move(other.recur_);
    }
    return *this;
  }

  /** Copy constructor */
  HSHM_CROSS_FUN TupleBase(const TupleBase &other) : recur_(other.recur_) {}

  /** Copy assignment operator */
  HSHM_CROSS_FUN TupleBase &operator=(const TupleBase &other) {
    if (this != &other) {
      recur_ = other.recur_;
    }
    return *this;
  }

  /** Solidification constructor */
  template <typename... CArgs>
  HSHM_CROSS_FUN explicit TupleBase(ArgPack<CArgs...> &&other)
      : recur_(std::forward<ArgPack<CArgs...>>(other)) {}

  /** Getter */
  template <size_t idx>
  HSHM_CROSS_FUN constexpr auto &Get() {
    return recur_.template Get<idx>();
  }

  /** Getter (const) */
  template <size_t idx>
  HSHM_CROSS_FUN constexpr auto &Get() const {
    return recur_.template Get<idx>();
  }

  /** Size */
  HSHM_CROSS_FUN constexpr static size_t Size() { return sizeof...(Args); }
};

/** Tuple definition */
template <typename... Containers>
using tuple = TupleBase<false, NullWrap, Containers...>;

/** Tuple Wrapper Definition */
template <template <typename> typename Wrap, typename... Containers>
using tuple_wrap = TupleBase<false, Wrap, Containers...>;

/** Apply a function over an entire TupleBase / tuple */
template <bool reverse>
class IterateTuple {
 public:
  /** Apply a function to every element of a tuple */
  template <typename TupleT, typename F>
  HSHM_CROSS_FUN constexpr static void Apply(TupleT &pack, F &&f) {
    _Apply<0, TupleT, F>(pack, std::forward<F>(f));
  }

 private:
  /** Apply the function recursively */
  template <size_t i, typename TupleT, typename F>
  HSHM_CROSS_FUN constexpr static void _Apply(TupleT &pack, F &&f) {
    if constexpr (i < TupleT::Size()) {
      if constexpr (reverse) {
        _Apply<i + 1, TupleT, F>(pack, std::forward<F>(f));
        f(MakeConstexpr<size_t, i>(), pack.template Get<i>());
      } else {
        f(MakeConstexpr<size_t, i>(), pack.template Get<i>());
        _Apply<i + 1, TupleT, F>(pack, std::forward<F>(f));
      }
    }
  }
};

/** Forward iterate over tuple and apply function  */
using ForwardIterateTuple = IterateTuple<false>;

/** Reverse iterate over tuple and apply function */
using ReverseIterateTuple = IterateTuple<true>;

}  // namespace hshm

#endif  // HSHM_INCLUDE_HSHM_DATA_STRUCTURES_TupleBase_H_
