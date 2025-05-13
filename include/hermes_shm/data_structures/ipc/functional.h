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

#ifndef HSHM_SHM_SHM_DATA_STRUCTURES_CONTAINERS_CmpTIONAL_H_
#define HSHM_SHM_SHM_DATA_STRUCTURES_CONTAINERS_CmpTIONAL_H_

#include <hermes_shm/constants/macros.h>
#include <hermes_shm/types/numbers.h>

#include "algorithm.h"

namespace hshm {

/** Find a value \val between start and end */
template <typename T, typename IterT>
HSHM_CROSS_FUN IterT find(IterT start, const IterT &end, T &val) {
  for (; start != end; ++start) {
    T &ref = *start;
    if (ref == val) {
      return start;
    }
  }
  return end;
}

}  // namespace hshm

#endif  // HSHM_SHM_SHM_DATA_STRUCTURES_CONTAINERS_CmpTIONAL_H_
