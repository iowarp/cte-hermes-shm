public:
/**====================================
 * Variables & Types
 * ===================================*/

typedef TYPED_HEADER header_t; /** Header type query */
header_t *header_; /**< Header of the shared-memory data structure */
hipc::Allocator *alloc_; /**< hipc::Allocator used for this data structure */
hermes_shm::bitfield32_t flags_; /**< Flags used data structure status */

/**====================================
 * Move Operations
 * ===================================*/

/** Move constructor */
explicit CLASS_NAME(CLASS_NAME &&other) {
shm_weak_move_main(std::forward<CLASS_NAME>(other));
}

/** SHM Constructor. Move operator. */
void shm_init(CLASS_NAME &&other) {
  SetNull();
  shm_strong_move_main(std::forward<CLASS_NAME>(other));
}

/** Weak move of simply pointers */
void shm_weak_move_main(CLASS_NAME &&other) {
  shm_make_header(other.header_, other.alloc_);
  flags_ = other.flags_;
  shm_deserialize_main();
  other.RemoveHeader();
}

/** Move assignment operator */
CLASS_NAME& operator=(CLASS_NAME &&other) {
  if (this == &other) {
    return *this;
  }
  shm_destroy();
  if (header_ == nullptr) {
    shm_weak_move_main(std::forward<CLASS_NAME>(other));
  } else if (alloc_ == other.alloc_) {
    shm_strong_move_main(std::forward<CLASS_NAME>(other));
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
  shm_erase_or_init(alloc_);
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
explicit CLASS_NAME(hipc::ShmArchive<CLASS_NAME> &header,
hipc::Allocator *alloc,
  Args&& ...args) {
shm_make_header(header.get(), alloc);
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

/** Destructor. */
~CLASS_NAME() {
  if (flags_.OrBits(SHM_PRIVATE_IS_DESTRUCTABLE)) {
    shm_destroy();
  }
}

/** Used by assignmnet operators */
void shm_erase_or_init(hipc::Allocator *alloc) {
  if (header_ != nullptr) {
    shm_erase();
  } else {
    shm_make_header(nullptr, alloc);
    SetNull();
  }
}

/** Erase operation */
void shm_erase() {
  if (!IsNull()) {
    shm_destroy_main();
    SetNull();
  }
}

/** Destruction operation */
void shm_destroy() {
  shm_erase();
  if (flags_.OrBits(SHM_PRIVATE_OWNS_HEADER) && header_) {
    alloc_->template FreePtr<TYPE_UNWRAP(TYPED_HEADER)>(header_);
    header_ = nullptr;
  }
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
  header_ = header;
  alloc_ = alloc;
  flags_.UnsetBits(SHM_PRIVATE_IS_DESTRUCTABLE | SHM_PRIVATE_OWNS_HEADER);
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