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

#ifndef HERMES_INCLUDE_HERMES_DATA_STRUCTURES_INTERNAL_SHM_CONTAINER_EXAMPLE_H_
#define HERMES_INCLUDE_HERMES_DATA_STRUCTURES_INTERNAL_SHM_CONTAINER_EXAMPLE_H_

#include "hermes_shm/data_structures/internal/shm_container.h"
#include "hermes_shm/data_structures/internal/shm_deserialize.h"

namespace honey {

class ShmContainerExample;

#define CLASS_NAME ShmContainerExample
#define TYPED_CLASS ShmContainerExample
#define TYPED_HEADER ShmHeader<ShmContainerExample>

template<typename T>
class ShmHeader;

template<>
 class ShmHeader<ShmContainerExample> : public hipc::ShmBaseHeader {
};

class ShmContainerExample : public hipc::ShmContainer {
 public:
  /**====================================
   * Variables & Types
   * ===================================*/

  typedef TYPED_HEADER header_t; /** Header type query */
  header_t *header_; /**< Header of the shared-memory data structure */
  hipc::Allocator *alloc_; /**< hipc::Allocator used for this data structure */
  hermes_shm::bitfield32_t flags_; /**< Flags used data structure status */

 public:
  /**====================================
   * Shm Overrides
   * ===================================*/

  /** Constructor. Empty. */
  explicit CLASS_NAME(TYPED_HEADER *header,
                      hipc::Allocator *alloc) {
    shm_make_header(header, alloc);
  }

  /** Default initialization */
  void shm_init() {
    SetNull();
  }

  /** Load from shared memory */
  void shm_deserialize_main() {}

  /** Destroy object */
  void shm_destroy_main() {}

  /** Internal copy operation */
  void shm_strong_copy_main(const CLASS_NAME &other) {
  }

  /** Internal move operation */
  void shm_weak_move_main(CLASS_NAME &&other) {
    memcpy(header_, other.header_, sizeof(*header_));
  }

  /** Check if header is NULL */
  bool IsNull() {
    return header_ == nullptr;
  }

  /** Nullify object header */
  void SetNull() {
  }

  /**====================================
   * Move Operations
   * ===================================*/

  /** Move constructor */
  explicit CLASS_NAME(CLASS_NAME &&other) {
    shm_make_header(other.header_, other.alloc_);
    shm_deserialize_main();
    other.RemoveHeader();
  }

   /** SHM Constructor. Move operator. */
   void shm_init(CLASS_NAME &&other) {
     SetNull();
     shm_weak_move_main(std::forward<CLASS_NAME>(other));
   }

  /** Move assignment operator */
  CLASS_NAME& operator=(CLASS_NAME &&other) {
    if (this == &other) {
      return *this;
    }
    shm_destroy();
    if (alloc_ == other.alloc_) {
      shm_weak_move_main(std::forward<CLASS_NAME>(other));
      other.SetNull();
    } else {
      shm_strong_copy_main(other);
      other.shm_destroy();
    }
    return *this;
  }

  /**====================================
   * Copy Operations
   * ===================================*/

   /** Copy constructor */
   CLASS_NAME(const CLASS_NAME &other) {
     shm_make_header(nullptr, other.alloc_);
     SetNull();
     shm_strong_copy_main(other);
   }

   /** Copy assignment operator */
   CLASS_NAME& operator=(const CLASS_NAME &other) {
     if (this == &other) {
       return *this;
     }
     shm_destroy();
     shm_strong_copy_main(other);
     return *this;
   }

   /** SHM Constructor. Copy operator. */
   void shm_init(const CLASS_NAME &other) {
     SetNull();
     shm_strong_copy_main(other);
   }

  /**====================================
   * Constructors
   * ===================================*/

  /** Constructor. Default allocator. */
  CLASS_NAME() {
    shm_make_header(nullptr, nullptr);
    shm_init();
  }

  /** Constructor. Default allocator with args. */
  template<typename ...Args>
  CLASS_NAME(Args&& ...args) {
    shm_make_header(nullptr, nullptr);
    shm_init(std::forward<Args>(args)...);
  }

  /** Constructor. Custom allocator. */
  template<typename ...Args>
  explicit CLASS_NAME(hipc::Allocator *alloc, Args&& ...args) {
    shm_make_header(nullptr, alloc);
    shm_init(std::forward<Args>(args)...);
  }

  /** Constructor. Header is pre-allocated. */
  template<typename ...Args>
  explicit CLASS_NAME(hipc::ShmInit,
                      hipc::ShmDeserialize<CLASS_NAME> ar,
                      Args&& ...args) {
    shm_make_header(ar.header_, ar.alloc_);
    shm_init(std::forward<Args>(args)...);
  }

  /** Constructor. Header is pre-allocated. */
  template<typename ...Args>
  explicit CLASS_NAME(TYPE_UNWRAP(TYPED_HEADER) *header,
                      hipc::Allocator *alloc,
                      Args&& ...args) {
    shm_make_header(header, alloc);
    shm_init(std::forward<Args>(args)...);
  }

   /** Initialize header + allocator */
  void shm_make_header(TYPE_UNWRAP(TYPED_HEADER) *header,
                       hipc::Allocator *alloc) {
    if (alloc == nullptr) {
      alloc = HERMES_MEMORY_REGISTRY->GetDefaultAllocator();
    }
    alloc_ = alloc;
    if (header == nullptr) {
      header_ = alloc_->template AllocateObjs<TYPE_UNWRAP(TYPED_HEADER)>(1);
      flags_.SetBits(SHM_PRIVATE_IS_DESTRUCTABLE | SHM_PRIVATE_OWNS_HEADER);
    } else {
      header_ = header;
      flags_.UnsetBits(SHM_PRIVATE_IS_DESTRUCTABLE | SHM_PRIVATE_OWNS_HEADER);
    }
  }

  /**====================================
   * Destructor
   * ===================================*/

  /** Destruction operation */
  void shm_destroy() {
    if (IsNull()) { return; }
    if (flags_.OrBits(SHM_PRIVATE_IS_DESTRUCTABLE)) {
      shm_destroy_main();
      if (flags_.OrBits(SHM_PRIVATE_OWNS_HEADER)) {
        alloc_->template FreePtr<TYPE_UNWRAP(TYPED_HEADER)>(header_);
      }
    }
    SetNull();
  }
  /**====================================
   * Serialization
   * ===================================*/

  /** Serialize into a Pointer */
  void shm_serialize(hipc::TypedPointer<TYPED_CLASS> &ar) const {
    ar = alloc_->template
      Convert<TYPED_HEADER, hipc::Pointer>(header_);
  }

  /** Serialize into an AtomicPointer */
  void shm_serialize(hipc::TypedAtomicPointer<TYPED_CLASS> &ar) const {
    ar = alloc_->template
      Convert<TYPED_HEADER, hipc::AtomicPointer>(header_);
  }

  /**====================================
   * Deserialization
   * ===================================*/

  /** Deserialize object from a raw pointer */
  bool shm_deserialize(const hipc::TypedPointer<TYPED_CLASS> &ar) {
    return shm_deserialize(
      HERMES_MEMORY_REGISTRY->GetAllocator(ar.allocator_id_),
      ar.ToOffsetPointer()
    );
  }

  /** Deserialize object from allocator + offset */
  bool shm_deserialize(hipc::Allocator *alloc, hipc::OffsetPointer header_ptr) {
    if (header_ptr.IsNull()) { return false; }
    return shm_deserialize(alloc->Convert<
                             TYPED_HEADER,
                             hipc::OffsetPointer>(header_ptr),
                           alloc);
  }

  /** Deserialize object from "Deserialize" object */
  bool shm_deserialize(hipc::ShmDeserialize<TYPED_CLASS> other) {
    return shm_deserialize(other.header_, other.alloc_);
  }

  /** Deserialize object from allocator + header */
  bool shm_deserialize(TYPED_HEADER *header,
                       hipc::Allocator *alloc) {
    shm_make_header(header, alloc);
    shm_deserialize_main();
    return true;
  }

  /** Constructor. Deserialize the object from the reference. */
  explicit CLASS_NAME(hipc::Ref<TYPED_CLASS> &obj) {
    shm_deserialize(obj->header_, obj->GetAllocator());
  }

  /** Constructor. Deserialize the object deserialize reference. */
  explicit CLASS_NAME(hipc::ShmDeserialize<TYPED_CLASS> other) {
    shm_deserialize(other);
  }

  /**====================================
   * Flag Operations
   * ===================================*/

  void SetHeaderOwned() {
    flags_.SetBits(SHM_PRIVATE_OWNS_HEADER);
  }

  void UnsetHeaderOwned() {
    flags_.UnsetBits(SHM_PRIVATE_OWNS_HEADER);
  }

  /**====================================
   * Header Operations
   * ===================================*/

  /** Set the header to null */
  void RemoveHeader() {
    header_ = nullptr;
  }

  /** Get a typed pointer to the object */
  template<typename POINTER_T>
  POINTER_T GetShmPointer() const {
    return alloc_->Convert<TYPED_HEADER, POINTER_T>(header_);
  }

  /** Get a ShmDeserialize object */
  hipc::ShmDeserialize<CLASS_NAME> GetShmDeserialize() const {
    return hipc::ShmDeserialize<CLASS_NAME>(header_, alloc_);
  }

  /**====================================
   * Query Operations
   * ===================================*/

  /** Get the allocator for this container */
  hipc::Allocator* GetAllocator() {
    return alloc_;
  }

  /** Get the allocator for this container */
  hipc::Allocator* GetAllocator() const {
    return alloc_;
  }

  /** Get the shared-memory allocator id */
  hipc::allocator_id_t GetAllocatorId() const {
    return alloc_->GetId();
  }
};

}  // namespace hermes_shm::ipc

#undef CLASS_NAME
#undef TYPED_CLASS
#undef TYPED_HEADER

#endif //HERMES_INCLUDE_HERMES_DATA_STRUCTURES_INTERNAL_SHM_CONTAINER_EXAMPLE_H_
