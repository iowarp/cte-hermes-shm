public:
/**====================================
 * Variables & Types
 * ===================================*/
HSHM_ALLOCATOR_INFO alloc_info_;

/**====================================
 * Constructors
 * ===================================*/
/** Initialize container */
HSHM_CROSS_FUN void init_shm_container(AllocT *alloc) {
  if constexpr (!IsPrivate) {
    alloc_info_ = alloc->GetId();
  } else {
    alloc_info_ = alloc;
  }
}

/**====================================
 * Destructor
 * ===================================*/
/** Destructor. */
HSHM_INLINE_CROSS_FUN ~TYPE_UNWRAP(CLASS_NAME)() {
  shm_destroy();
}

/** Destruction operation */
HSHM_INLINE_CROSS_FUN void shm_destroy() {
  if (IsNull()) { return; }
  shm_destroy_main();
  SetNull();
}

/**====================================
 * Header Operations
 * ===================================*/

/** Get a typed pointer to the object */
template<typename POINTER_T>
HSHM_INLINE_CROSS_FUN POINTER_T GetShmPointer() const {
  return GetAllocator()->template Convert<TYPE_UNWRAP(TYPED_CLASS), POINTER_T>(this);
}

/**====================================
 * Query Operations
 * ===================================*/

/** Get the allocator for this container */
HSHM_INLINE_CROSS_FUN AllocT* GetAllocator() const {
  if constexpr (!IsPrivate) {
    return (AllocT*)HERMES_MEMORY_MANAGER->GetAllocator(alloc_info_);
  } else {
    return alloc_info_;
  }
}

/** Get the shared-memory allocator id */
HSHM_INLINE_CROSS_FUN const hipc::AllocatorId& GetAllocatorId() const {
  if constexpr (!IsPrivate) {
    return alloc_info_;
  } else {
    return GetAllocator()->GetId();
  }
}