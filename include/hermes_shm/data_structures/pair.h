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
struct ShmHeader<TYPED_CLASS> : public ShmBaseHeader {
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
  hipc::ShmRef<FirstT> first_;
  hipc::ShmRef<SecondT> second_;

 public:
  /**====================================
   * SHM Overrides
   * ===================================*/

  /** Default shm constructor */
  explicit pair(TYPED_HEADER *header, Allocator *alloc) {
    shm_init_header(header, alloc);
    header_->first_.shm_init();
    header_->second_.shm_init();
  }

  /** Construct pair by moving parameters */
  explicit pair(TYPED_HEADER *header, Allocator *alloc,
                FirstT &&first, SecondT &&second) {
    shm_init_header(header, alloc);
    header_->first_.shm_init(std::forward<FirstT>(first));
    header_->second_.shm_init(std::forward<SecondT>(second));
    shm_deserialize_main();

  }

  /** Construct pair by copying parameters */
  explicit pair(TYPED_HEADER *header, Allocator *alloc,
                const FirstT &first, const SecondT &second) {
    shm_init_header(header, alloc);
    header_->first_.shm_init(first);
    header_->second_.shm_init(second);
    shm_deserialize_main();
  }

  /** Construct pair piecewise */
  template<typename FirstArgPackT, typename SecondArgPackT>
  explicit pair(TYPED_HEADER *header, Allocator *alloc,
                PiecewiseConstruct &&hint,
                FirstArgPackT &&first,
                SecondArgPackT &&second) {
    shm_init_header(header, alloc);
    header_->first_.shm_init_piecewise(std::forward<FirstArgPackT>(first));
    header_->second_.shm_init_piecewise(std::forward<SecondArgPackT>(second));
    shm_deserialize_main();
  }

  /** Move constructor */
  explicit CLASS_NAME(CLASS_NAME &&other) {
    shm_init_header(other.header_, other.alloc_);
    shm_deserialize_main();
    other.RemoveHeader();
  }

  /** Move assignment operator */
  CLASS_NAME& operator=(CLASS_NAME &&other) {
    if (this == &other) {
      return *this;
    }
    shm_destroy();
    if (!other.IsNull()) {
      if (alloc_ == other.alloc_) {
        (*first_) = std::move(*other.first_);
        (*second_) = std::move(*other.second_);
        other.SetNull();
      } else {
        shm_strong_copy_main(other);
        other.shm_destroy();
      }
    }
    return *this;
  }

  /** Copy assignment operator */
  CLASS_NAME& operator=(const pair &&other) {
    if (this == &other) {
      return *this;
    }
    shm_destroy();
    if (!other.IsNull()) {
      shm_strong_copy_main(other);
    }
    return *this;
  }

  /** Copy constructor */
  void shm_strong_copy_main(const pair &other) {
    (*first_) = (*other.first_);
    (*second_) = (*other.second_);
  }

  /** Destroy the shared-memory data */
  void shm_destroy() {
    if (IsNull()) { return; }
    first_->shm_destroy();
    second_->shm_destroy();
    SetNull();
  }

  /** Store into shared memory */
  void shm_serialize_main() const {}

  /** Load from shared memory */
  void shm_deserialize_main() {
    first_ = hipc::ShmRef<FirstT>(header_->first_.internal_ref(alloc_));
    second_ = hipc::ShmRef<SecondT>(header_->second_.internal_ref(alloc_));
  }

  /** Check if the pair is empty */
  bool IsNull() {
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
