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

#ifndef HERMES_SHM_INCLUDE_HERMES_SHM_UTIL_EASY_LOCKFREE_SINGLETON_SINGLETON_H
#define HERMES_SHM_INCLUDE_HERMES_SHM_UTIL_EASY_LOCKFREE_SINGLETON_SINGLETON_H

#include <memory>
#include "hermes_shm/constants/macros.h"

namespace hshm {

/**
 * Makes a singleton. Constructs the first time GetInstance is called.
 * Requires user to define the static storage of obj_ in separate file.
 * @tparam T
 */
template<typename T>
class EasyLockfreeSingleton {
 public:
  /** Get or create an instance of type T */
  HSHM_CROSS_FUN static T* GetInstance() {
    static T *obj_;
    if (!obj_) {
      if (obj_ == nullptr) {
        obj_ = new T();
      }
    }
    return obj_;
  }
};

}  // namespace hshm

#endif  // HERMES_SHM_INCLUDE_HERMES_SHM_UTIL_EASY_LOCKFREE_SINGLETON_SINGLETON_H
