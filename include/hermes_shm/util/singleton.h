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

#ifndef HSHM_SHM_SINGLETON_H
#define HSHM_SHM_SINGLETON_H

#include <memory>

#include "hermes_shm/constants/macros.h"
#include "hermes_shm/thread/lock/spin_lock.h"

namespace hshm {

/**
 * A class to represent singleton pattern
 * Does not require specific initialization of the static variable
 * */
template <typename T, bool WithLock>
class SingletonBase {
 public:
  static T *GetInstance() {
    if (GetObject() == nullptr) {
      if constexpr (WithLock) {
        hshm::ScopedSpinLock lock(GetSpinLock(), 0);
        new ((T *)GetData()) T();
        GetObject() = (T *)GetData();
      } else {
        new ((T *)GetData()) T();
        GetObject() = (T *)GetData();
      }
    }
    return GetObject();
  }

  static hshm::SpinLock &GetSpinLock() {
    static char spinlock_data_[sizeof(hshm::SpinLock)] = {0};
    return *(hshm::SpinLock *)spinlock_data_;
  }

  static T *GetData() {
    static char data_[sizeof(T)] = {0};
    return (T *)data_;
  }

  static T *&GetObject() {
    static T *obj_ = nullptr;
    return obj_;
  }
};

/** Singleton default case declaration */
template <typename T>
using Singleton = SingletonBase<T, true>;

/** Singleton without lock declaration */
template <typename T>
using LockfreeSingleton = SingletonBase<T, false>;

/**
 * A class to represent singleton pattern
 * Does not require specific initialization of the static variable
 * */
template <typename T, bool WithLock>
class CrossSingletonBase {
 public:
  HSHM_INLINE_CROSS_FUN
  static T *GetInstance() {
    if (GetObject() == nullptr) {
      if constexpr (WithLock) {
        hshm::ScopedSpinLock lock(GetSpinLock(), 0);
        new ((T *)GetData()) T();
        GetObject() = (T *)GetData();
      } else {
        new ((T *)GetData()) T();
        GetObject() = (T *)GetData();
      }
    }
    return GetObject();
  }

  HSHM_INLINE_CROSS_FUN
  static hshm::SpinLock &GetSpinLock() {
    static char spinlock_data_[sizeof(hshm::SpinLock)] = {0};
    return *(hshm::SpinLock *)spinlock_data_;
  }

  HSHM_INLINE_CROSS_FUN
  static T *GetData() {
    static char data_[sizeof(T)] = {0};
    return (T *)data_;
  }

  HSHM_INLINE_CROSS_FUN
  static T *&GetObject() {
    static T *obj_ = nullptr;
    return obj_;
  }
};

/** Singleton default case declaration */
template <typename T>
using CrossSingleton = CrossSingletonBase<T, true>;

/** Singleton without lock declaration */
template <typename T>
using LockfreeCrossSingleton = CrossSingletonBase<T, false>;

/**
 * Makes a singleton. Constructs during initialization of program.
 * Does not require specific initialization of the static variable.
 * */
template <typename T>
class GlobalSingleton {
 private:
  static T obj_;

 public:
  GlobalSingleton() = default;

  static T *GetInstance() { return &obj_; }
};
template <typename T>
T GlobalSingleton<T>::obj_;

/**
 * Makes a singleton. Constructs during initialization of program.
 * Does not require specific initialization of the static variable.
 * */
#ifdef HSHM_IS_HOST
template <typename T>
class GlobalCrossSingleton {
 private:
  static T obj_;

 public:
  GlobalCrossSingleton() = default;

  static T *GetInstance() { return &obj_; }
};
template <typename T>
T GlobalCrossSingleton<T>::obj_;
#else
template <typename T>
using GlobalCrossSingleton = LockfreeCrossSingleton<T>;
#endif

}  // namespace hshm

#endif  // HSHM_SHM_SINGLETON_H
