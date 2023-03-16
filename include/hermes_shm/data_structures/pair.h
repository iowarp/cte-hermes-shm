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

#ifndef HERMES_INCLUDE_HERMES_DATA_STRUCTURES_PAIR_H_
#define HERMES_INCLUDE_HERMES_DATA_STRUCTURES_PAIR_H_

#include "internal/shm_internal.h"
#include "hermes_shm/data_structures/smart_ptr/smart_ptr_base.h"

namespace hermes_shm::ipc {

/** forward declaration for string */
template<typename FirstT, typename SecondT>
class pair;

/**
* MACROS used to simplify the string namespace
* Used as inputs to the SHM_CONTAINER_TEMPLATE
* */
#define CLASS_NAME pair
#define TYPED_CLASS pair<FirstT, SecondT>
#define TYPED_HEADER ShmHeader<TYPED_CLASS>

/** pair shared-memory header */
template<typename FirstT, typename SecondT>
struct ShmHeader<TYPED_CLASS> {
  SHM_CONTAINER_HEADER_TEMPLATE(ShmHeader)
  ShmArchive<FirstT> first_;
  ShmArchive<SecondT> second_;
  void strong_copy() {}
};

/**
* A string of characters.
* */
template<typename FirstT, typename SecondT>
class pair : public ShmContainer {
 public:
  SHM_CONTAINER_TEMPLATE((CLASS_NAME), (TYPED_CLASS), (TYPED_HEADER))

 public:
  hipc::Ref<FirstT> first_;
  hipc::Ref<SecondT> second_;

 public:
  /**====================================
   * SHM Overrides
   * ===================================*/

  /** SHM constructor. Default. */
  void shm_init() {
    first_ = make_ref<FirstT>(header_->first_, alloc_);
    second_ = make_ref<SecondT>(header_->second_, alloc_);
  }

  /** SHM constructor. Move parameters. */
  void shm_init(FirstT &&first, SecondT &&second) {
    first_ = make_ref<FirstT>(header_->first_,
                              alloc_, std::forward<FirstT>(first));
    second_ = make_ref<SecondT>(header_->second_,
                                alloc_, std::forward<SecondT>(second));
  }

  /** SHM constructor. Copy parameters. */
  void shm_init(const FirstT &first, const SecondT &second) {
    first_ = make_ref<FirstT>(header_->first_, alloc_, first);
    second_ = make_ref<SecondT>(header_->second_, alloc_, second);
  }

  /** SHM constructor. Piecewise emplace. */
  template<typename FirstArgPackT, typename SecondArgPackT>
  void shm_init(PiecewiseConstruct &&hint,
                FirstArgPackT &&first,
                SecondArgPackT &&second) {
    first_ = make_ref_piecewise<FirstT>(make_argpack(header_->first_, alloc_),
                                        std::forward<FirstArgPackT>(first));
    second_ = make_ref_piecewise<SecondT>(make_argpack(header_->second_, alloc_),
                                          std::forward<SecondArgPackT>(second));
  }

  /** Internal move operation */
  void shm_strong_move_main(CLASS_NAME &&other) {
    (*first_) = std::move(*other.first_);
    (*second_) = std::move(*other.second_);
  }

  /** Internal copy operation */
  void shm_strong_copy_main(const pair &other) {
    (*first_) = (*other.first_);
    (*second_) = (*other.second_);
  }

  /** Destroy the shared-memory data */
  void shm_destroy_main() {
    first_.shm_destroy();
    second_.shm_destroy();
  }

  /** Load from shared memory */
  void shm_deserialize_main() {
    first_ = hipc::Ref<FirstT>(header_->first_, alloc_);
    second_ = hipc::Ref<SecondT>(header_->second_, alloc_);
  }

  /** Check if the pair is empty */
  bool IsNull() const {
    return header_ == nullptr;
  }

  /** Sets this pair as empty */
  void SetNull() {}

  /**====================================
   * pair Methods
   * ===================================*/

  /** Get the first object */
  FirstT& GetFirst() { return *first_; }

  /** Get the first object (const) */
  FirstT& GetFirst() const { return *first_; }

  /** Get the second object */
  SecondT& GetSecond() { return *second_; }

  /** Get the second object (const) */
  SecondT& GetSecond() const { return *second_; }

  /** Get the first object (treated as key) */
  FirstT& GetKey() { return *first_; }

  /** Get the first object (treated as key) (const) */
  FirstT& GetKey() const { return *first_; }

  /** Get the second object (treated as value) */
  SecondT& GetVal() { return *second_; }

  /** Get the second object (treated as value) (const) */
  SecondT& GetVal() const { return *second_; }
};

#undef CLASS_NAME
#undef TYPED_CLASS
#undef TYPED_HEADER

}  // namespace hermes_shm::ipc

#endif  // HERMES_INCLUDE_HERMES_DATA_STRUCTURES_PAIR_H_
