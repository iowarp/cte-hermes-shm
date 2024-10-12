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

#ifndef HERMES_SHM_INCLUDE_HERMES_SHM_UTIL_SINGLETON_SINGLETON_H_
#define HERMES_SHM_INCLUDE_HERMES_SHM_UTIL_SINGLETON_SINGLETON_H_

#include <memory>
#include "hermes_shm/thread/lock/mutex.h"
#include "hermes_shm/constants/macros.h"

namespace hshm {

/**
 * Makes a singleton. Constructs the first time GetInstance is called.
 * Requires user to define the static storage of obj_ in separate file.
 * @tparam T
 */
template<typename T>
class Singleton {
 private:
  static T *obj_;
  static hshm::Mutex *lock_;

 public:
  HSHM_CROSS_FUN Singleton() = default;

  /** Get or create an instance of type T */
  HSHM_CROSS_FUN
  inline static T *GetInstance() {
    if (!obj_) {
      hshm::ScopedMutex lock(*lock_, 0);
      if (obj_ == nullptr) {
        obj_ = new T();
      }
    }
    return obj_;
  }

  /** Static initialization method for obj */
  HSHM_CROSS_FUN
  static T *_GetObj();

  /** Static initialization method for lock */
  HSHM_CROSS_FUN
  static hshm::Mutex *_GetLock();
};

template<typename T>
T* Singleton<T>::obj_ = Singleton<T>::_GetObj();
template<typename T>
hshm::Mutex* Singleton<T>::lock_ = Singleton<T>::_GetLock();
#define DEFINE_SINGLETON_CC(T)\
  HSHM_CROSS_FUN template<> T* hshm::Singleton<T>::_GetObj() {\
    return nullptr;\
  }\
  HSHM_CROSS_FUN template<> hshm::Mutex* hshm::Singleton<T>::_GetLock() {\
    static hshm::Mutex lock;\
    return &lock;\
  }

}  // namespace hshm

#endif  // HERMES_SHM_INCLUDE_HERMES_SHM_UTIL_SINGLETON_SINGLETON_H_
