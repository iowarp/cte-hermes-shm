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

#ifndef HSHM_INCLUDE_HSHM_DATA_STRUCTURES_PAIR_H_
#define HSHM_INCLUDE_HSHM_DATA_STRUCTURES_PAIR_H_

#include "hermes_shm/constants/macros.h"
#include "hermes_shm/data_structures/internal/shm_internal.h"

namespace hshm::ipc {

/** forward declaration for string */
template <typename FirstT, typename SecondT, HSHM_CLASS_TEMPL_WITH_DEFAULTS>
class pair;

/**
 * MACROS used to simplify the string namespace
 * Used as inputs to the HIPC_CONTAINER_TEMPLATE
 * */
#define CLASS_NAME pair
#define CLASS_NEW_ARGS FirstT, SecondT

/**
 * A pair of two objects.
 * */
template <typename FirstT, typename SecondT, HSHM_CLASS_TEMPL>
class pair : public ShmContainer {
 public:
  /**====================================
   * Variables
   * ===================================*/
  HIPC_CONTAINER_TEMPLATE((CLASS_NAME), (CLASS_NEW_ARGS))
  delay_ar<FirstT> first_;
  delay_ar<SecondT> second_;

 public:
  /**====================================
   * Default Constructor
   * ===================================*/

  /** Constructor. Default. */
  HSHM_CROSS_FUN
  explicit pair() {
    shm_init(HSHM_MEMORY_MANAGER->GetDefaultAllocator<AllocT>());
  }

  /** SHM constructor. Default. */
  HSHM_CROSS_FUN
  explicit pair(const hipc::CtxAllocator<AllocT> &alloc) { shm_init(alloc); }

  /** SHM constructor */
  HSHM_CROSS_FUN
  void shm_init(const hipc::CtxAllocator<AllocT> &alloc) {
    init_shm_container(alloc);
    first_.shm_init(GetCtxAllocator());
    second_.shm_init(GetCtxAllocator());
  }

  /**====================================
   * Emplace Constructors
   * ===================================*/

  /** Constructor. Move parameters. */
  HSHM_CROSS_FUN
  explicit pair(FirstT &&first, SecondT &&second) {
    init_shm_container(HSHM_MEMORY_MANAGER->GetDefaultAllocator<AllocT>());
    first_.shm_init(GetCtxAllocator(), std::forward<FirstT>(first));
    second_.shm_init(GetCtxAllocator(), std::forward<SecondT>(second));
  }

  /** SHM constructor. Move parameters. */
  HSHM_CROSS_FUN
  explicit pair(const hipc::CtxAllocator<AllocT> &alloc, FirstT &&first,
                SecondT &&second) {
    init_shm_container(alloc);
    first_.shm_init(GetCtxAllocator(), std::forward<FirstT>(first));
    second_.shm_init(GetCtxAllocator(), std::forward<SecondT>(second));
  }

  /** Constructor. Copy parameters. */
  HSHM_CROSS_FUN
  explicit pair(const FirstT &first, const SecondT &second) {
    init_shm_container(HSHM_MEMORY_MANAGER->GetDefaultAllocator<AllocT>());
    first_.shm_init(GetCtxAllocator(), first);
    second_.shm_init(GetCtxAllocator(), second);
  }

  /** SHM constructor. Copy parameters. */
  HSHM_CROSS_FUN
  explicit pair(const hipc::CtxAllocator<AllocT> &alloc, const FirstT &first,
                const SecondT &second) {
    init_shm_container(alloc);
    first_.shm_init(GetCtxAllocator(), first);
    second_.shm_init(GetCtxAllocator(), second);
  }

  /** SHM constructor. Piecewise emplace. */
  template <typename FirstArgPackT, typename SecondArgPackT>
  HSHM_CROSS_FUN explicit pair(PiecewiseConstruct &&hint, FirstArgPackT &&first,
                               SecondArgPackT &&second) {
    init_shm_container(HSHM_MEMORY_MANAGER->GetDefaultAllocator<AllocT>());
    HSHM_MAKE_AR_PW(first_, GetCtxAllocator(),
                    std::forward<FirstArgPackT>(first))
    HSHM_MAKE_AR_PW(second_, GetCtxAllocator(),
                    std::forward<SecondArgPackT>(second))
  }

  /** SHM constructor. Piecewise emplace. */
  template <typename FirstArgPackT, typename SecondArgPackT>
  HSHM_CROSS_FUN explicit pair(const hipc::CtxAllocator<AllocT> &alloc,
                               PiecewiseConstruct &&hint, FirstArgPackT &&first,
                               SecondArgPackT &&second) {
    init_shm_container(alloc);
    HSHM_MAKE_AR_PW(first_, GetCtxAllocator(),
                    std::forward<FirstArgPackT>(first))
    HSHM_MAKE_AR_PW(second_, GetCtxAllocator(),
                    std::forward<SecondArgPackT>(second))
  }

  /**====================================
   * Copy Constructors
   * ===================================*/

  /** Copy constructor. From pair. */
  HSHM_CROSS_FUN
  explicit pair(const pair &other) {
    init_shm_container(other.GetCtxAllocator());
    shm_strong_copy_construct(other);
  }

  /** SHM copy constructor. From pair. */
  HSHM_CROSS_FUN
  explicit pair(const hipc::CtxAllocator<AllocT> &alloc, const pair &other) {
    init_shm_container(alloc);
    shm_strong_copy_construct(other);
  }

  /** SHM copy constructor main */
  HSHM_CROSS_FUN
  void shm_strong_copy_construct(const pair &other){
      first_.shm_init(GetCtxAllocator(), *other.first_);
          second_.shm_init(GetCtxAllocator(), *other.second_);}

  /** SHM copy assignment operator. From pair. */
  HSHM_CROSS_FUN pair &operator=(const pair &other) {
    if (this != &other) {
      shm_destroy();
      shm_strong_copy_assign_op(other);
    }
    return *this;
  }

  /** SHM copy assignment operator main */
  HSHM_CROSS_FUN
  void shm_strong_copy_assign_op(const pair &other) {
    (first_.get_ref()) = (*other.first_);
    (second_.get_ref()) = (*other.second_);
  }

  /**====================================
   * Move Constructors
   * ===================================*/

  /** SHM move constructor. From pair. */
  HSHM_CROSS_FUN
  explicit pair(pair &&other) {
    init_shm_container(other.GetCtxAllocator());
    if (GetAllocator() == other.GetAllocator()) {
      first_.shm_init(GetCtxAllocator(),
                   std::forward<FirstT>(*other.first_));
      second_.shm_init(GetCtxAllocator(),
                   std::forward<SecondT>(*other.second_));
    } else {
      shm_strong_copy_construct(other);
      other.shm_destroy();
    }
  }

  /** SHM move constructor. From pair. */
  HSHM_CROSS_FUN
  explicit pair(const hipc::CtxAllocator<AllocT> &alloc, pair &&other) {
    init_shm_container(alloc);
    if (GetAllocator() == other.GetAllocator()) {
      first_.shm_init(GetCtxAllocator(),
                   std::forward<FirstT>(*other.first_));
      second_.shm_init(GetCtxAllocator(),
                   std::forward<SecondT>(*other.second_));
    } else {
      shm_strong_copy_construct(other);
      other.shm_destroy();
    }
  }

  /** SHM move assignment operator. From pair. */
  HSHM_CROSS_FUN
  pair &operator=(pair &&other) noexcept {
    if (this != &other) {
      if (GetAllocator() == other.GetAllocator()) {
        (*first_) = std::move(*other.first_);
        (*second_) = std::move(*other.second_);
        other.SetNull();
      } else {
        shm_strong_copy_assign_op(other);
        other.shm_destroy();
      }
    }
    return *this;
  }

  /**====================================
   * Destructor
   * ===================================*/

  /** Check if the pair is empty */
  HSHM_INLINE_CROSS_FUN bool IsNull() const { return false; }

  /** Sets this pair as empty */
  HSHM_INLINE_CROSS_FUN void SetNull() {}

  /** Destroy the shared-memory data */
  HSHM_INLINE_CROSS_FUN void shm_destroy_main(){HSHM_DESTROY_AR(first_)
                                                    HSHM_DESTROY_AR(second_)}

  /**====================================
   * pair Methods
   * ===================================*/

  /** Get the first object */
  HSHM_INLINE_CROSS_FUN FirstT &GetFirst() {
    return first_.get_ref();
  }

  /** Get the first object (const) */
  HSHM_INLINE_CROSS_FUN FirstT &GetFirst() const { return first_.get_ref(); }

  /** Get the second object */
  HSHM_INLINE_CROSS_FUN SecondT &GetSecond() { return second_.get_ref(); }

  /** Get the second object (const) */
  HSHM_INLINE_CROSS_FUN SecondT &GetSecond() const { return second_.get_ref(); }

  /** Get the first object (treated as key) */
  HSHM_INLINE_CROSS_FUN FirstT &GetKey() { return first_.get_ref(); }

  /** Get the first object (treated as key) (const) */
  HSHM_INLINE_CROSS_FUN FirstT &GetKey() const { return first_.get_ref(); }

  /** Get the second object (treated as value) */
  HSHM_INLINE_CROSS_FUN SecondT &GetVal() { return second_.get_ref(); }

  /** Get the second object (treated as value) (const) */
  HSHM_INLINE_CROSS_FUN SecondT &GetVal() const { return second_.get_ref(); }
};

}  // namespace hshm::ipc

namespace hshm {

template <typename FirstT, typename SecondT,
          HSHM_CLASS_TEMPL_WITH_PRIV_DEFAULTS>
using pair = hipc::pair<FirstT, SecondT, HSHM_CLASS_TEMPL_ARGS>;

}  // namespace hshm

#undef CLASS_NAME
#undef CLASS_NEW_ARGS

#endif  // HSHM_INCLUDE_HSHM_DATA_STRUCTURES_PAIR_H_
