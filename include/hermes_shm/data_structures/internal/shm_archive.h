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

#ifndef HSHM_DATA_STRUCTURES_SHM_ARCHIVE_H_
#define HSHM_DATA_STRUCTURES_SHM_ARCHIVE_H_

#include "hermes_shm/constants/macros.h"
#include "hermes_shm/memory/allocator/allocator.h"
#include "hermes_shm/memory/memory.h"
#include "hermes_shm/types/argpack.h"
#include "shm_container.h"
#include "shm_macros.h"

namespace hshm::ipc {

/**
 * Represents the layout of a data structure in shared memory.
 * This class is used to delay the initialization of a data structure
 * until it is needed. It treats ShmContainer types differently from
 * regular types.
 * */
template <typename T>
class delay_ar {
 public:
  typedef T internal_t;
  char obj_[sizeof(T)];

  /** Default constructor */
  HSHM_INLINE_CROSS_FUN delay_ar() = default;

  /** Destructor */
  HSHM_INLINE_CROSS_FUN ~delay_ar() = default;

  /** Pointer to internal object */
  HSHM_INLINE_CROSS_FUN T* get() { return reinterpret_cast<T*>(obj_); }

  /** Pointer to internal object (const) */
  HSHM_INLINE_CROSS_FUN const T* get() const {
    return reinterpret_cast<const T*>(obj_);
  }

  /** Get pointer (alwyas used and never inlined) */
  HSHM_NO_INLINE_CROSS_FUN T* get_dbg() { return reinterpret_cast<T*>(obj_); }

  /** Reference to internal object */
  HSHM_INLINE_CROSS_FUN T& get_ref() { return *get(); }

  /** Reference to internal object (const) */
  HSHM_INLINE_CROSS_FUN const T& get_ref() const { return *get(); }

  /** Dereference operator */
  HSHM_INLINE_CROSS_FUN T& operator*() { return get_ref(); }

  /** Dereference operator */
  HSHM_INLINE_CROSS_FUN const T& operator*() const { return get_ref(); }

  /** Arrow operator */
  HSHM_INLINE_CROSS_FUN T* operator->() { return get(); }

  /** Arrow operator */
  HSHM_INLINE_CROSS_FUN const T* operator->() const { return get(); }

  /** Copy constructor */
  HSHM_INLINE_CROSS_FUN
  delay_ar(const delay_ar& other) = delete;

  /** Copy assignment operator */
  HSHM_INLINE_CROSS_FUN
  delay_ar& operator=(const delay_ar& other) = delete;

  /** Move constructor */
  HSHM_INLINE_CROSS_FUN
  delay_ar(delay_ar&& other) = delete;

  /** Move assignment operator */
  HSHM_INLINE_CROSS_FUN
  delay_ar& operator=(delay_ar&& other) = delete;

  /** Initialize */
  template <typename... Args>
  HSHM_INLINE_CROSS_FUN void shm_init(Args&&... args) {
    Allocator::ConstructObj<T>(get_ref(), std::forward<Args>(args)...);
  }

  /** Initialize with allocator for SHM_ARCHIVEABLE, without allocator for
   * others */
  template <typename AllocT, typename... Args>
  HSHM_INLINE_CROSS_FUN void shm_init(AllocT&& alloc, Args&&... args) {
    if constexpr (IS_SHM_ARCHIVEABLE(T)) {
      // For SHM archiveable types: pass allocator + args
      Allocator::ConstructObj<T>(get_ref(), std::forward<AllocT>(alloc),
                                 std::forward<Args>(args)...);
    } else {
      // For non-SHM archiveable types: skip the allocator, pass only remaining
      // args
      if constexpr (sizeof...(args) > 0) {
        Allocator::ConstructObj<T>(get_ref(), std::forward<Args>(args)...);
      } else {
        // For non-SHM_ARCHIVEABLE types with no extra args, default construct
        // This covers primitive types like size_t, int, etc.
        // Note: PageAllocator case will fail and needs special handling
        Allocator::ConstructObj<T>(get_ref());
      }
    }
  }

  /** Initialize piecewise */
  template <typename ArgPackT_1, typename ArgPackT_2>
  HSHM_INLINE_CROSS_FUN void shm_init_piecewise(ArgPackT_1&& args1,
                                                ArgPackT_2&& args2) {
    return hshm::PassArgPack::Call(
        MergeArgPacks::Merge(make_argpack(get_ref()),
                             std::forward<ArgPackT_1>(args1),
                             std::forward<ArgPackT_2>(args2)),
        [](auto&&... args) constexpr {
          Allocator::ConstructObj<T>(std::forward<decltype(args)>(args)...);
        });
  }

  /** Destroy */
  HSHM_INLINE_CROSS_FUN
  void shm_destroy() { Allocator::DestructObj<T>(get_ref()); }
};

/**
 * NOTE(llogan): Why use macros here instead of templates?
 * Good question!
 *
 * C++ templates are very annoying! Template types cannot be
 * inferred when they are nested inside of other templates.
 *
 * These macros avoid needing to constantly specify that
 * template parameter.
 * */

/** A macro to determine the type of AR automatically */
#define HSHM_AR_GET_TYPE(AR) \
  (typename std::remove_reference<decltype(AR)>::type::internal_t)

/** Construct a piecewise archive */
#define HSHM_MAKE_AR_PW(AR, ALLOC, ...)                        \
  if constexpr (IS_SHM_ARCHIVEABLE(HSHM_AR_GET_TYPE(AR))) {    \
    (AR).shm_init_piecewise(make_argpack(ALLOC), __VA_ARGS__); \
  } else {                                                     \
    (AR).shm_init_piecewise(make_argpack(), __VA_ARGS__);      \
  }

/** Destroy an archive */
#define HSHM_DESTROY_AR(AR) (AR).shm_destroy();

template <typename Ar, typename T>
void HSHM_CROSS_FUN save(Ar& ar, const delay_ar<T>& obj) {
  ar & obj.get_ref();
}

template <typename Ar, typename T>
void HSHM_CROSS_FUN load(Ar& ar, delay_ar<T>& obj) {
  obj.shm_init(HSHM_DEFAULT_ALLOC);
  ar & obj.get_ref();
}

}  // namespace hshm::ipc

#endif  // HSHM_DATA_STRUCTURES_SHM_ARCHIVE_H_
