public:
/**====================================
 * Variables & Types
 * ===================================*/
HSHM_ALLOCATOR_INFO alloc_info_;

/**====================================
 * Constructors
 * ===================================*/
/** Initialize container */
HSHM_CROSS_FUN
void init_shm_container(AllocT *alloc) {
  if constexpr (!(HSHM_FLAGS & hipc::ShmFlag::kIsPrivate)) {
    alloc_info_ = alloc->GetId();
  } else {
    alloc_info_.alloc_ = alloc;
    alloc_info_.ctx_ = hipc::MemContext();
  }
}

/** Initialize container (thread-local) */
HSHM_CROSS_FUN
void init_shm_container(const hipc::MemContext &ctx, AllocT *alloc) {
  if constexpr (!(HSHM_FLAGS & hipc::ShmFlag::kIsPrivate)) {
    alloc_info_ = alloc->GetId();
  } else {
    alloc_info_.alloc_ = alloc;
    alloc_info_.ctx_ = ctx;
  }
}

/** Initialize container (thread-local) */
HSHM_CROSS_FUN
void init_shm_container(const hipc::CtxAllocator<AllocT> &tls_alloc) {
  init_shm_container(tls_alloc.ctx_, tls_alloc.alloc_);
}

/**====================================
 * Destructor
 * ===================================*/
/** Destructor. */
HSHM_INLINE_CROSS_FUN
~TYPE_UNWRAP(CLASS_NAME)() {
  if constexpr ((HSHM_FLAGS & hipc::ShmFlag::kIsUndestructable)) {
    shm_destroy();
  }
}

/** Destruction operation */
HSHM_INLINE_CROSS_FUN
void shm_destroy() {
  if (IsNull()) { return; }
  shm_destroy_main();
  SetNull();
}

/**====================================
 * Header Operations
 * ===================================*/

/** Get a typed pointer to the object */
template<typename POINTER_T>
HSHM_INLINE_CROSS_FUN
    POINTER_T GetShmPointer() const {
  return GetAllocator()->template Convert<TYPE_UNWRAP(TYPED_CLASS), POINTER_T>(this);
}

/**====================================
 * Query Operations
 * ===================================*/

/** Get the allocator for this container */
HSHM_INLINE_CROSS_FUN
    AllocT* GetAllocator() const {
  if constexpr (!(HSHM_FLAGS & hipc::ShmFlag::kIsPrivate)) {
    return (AllocT*)HERMES_MEMORY_MANAGER->GetAllocator(alloc_info_);
  } else {
    return alloc_info_.alloc_;
  }
}

/** Get the shared-memory allocator id */
HSHM_INLINE_CROSS_FUN
const hipc::AllocatorId& GetAllocatorId() const {
  if constexpr (!(HSHM_FLAGS & hipc::ShmFlag::kIsPrivate)) {
    return alloc_info_;
  } else {
    return GetAllocator()->GetId();
  }
}

/** Get the shared-memory allocator id */
HSHM_INLINE_CROSS_FUN
    hshm::ThreadId GetThreadId() const {
  if constexpr (!(HSHM_FLAGS & hipc::ShmFlag::kIsPrivate)) {
    return hshm::ThreadId::GetNull();
  } else {
    return alloc_info_.ctx_;
  }
}

/** Get the shared-memory allocator id */
HSHM_INLINE_CROSS_FUN
    hipc::CtxAllocator<AllocT> GetTlsAllocator() const {
  return hipc::CtxAllocator<AllocT>{GetThreadId(), GetAllocator()};
}