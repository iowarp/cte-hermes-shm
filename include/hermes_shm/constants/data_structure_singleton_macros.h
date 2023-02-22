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

#ifndef HERMES_INCLUDE_HERMES_CONSTANTS_DATA_STRUCTURE_SINGLETON_MACROS_H_H
#define HERMES_INCLUDE_HERMES_CONSTANTS_DATA_STRUCTURE_SINGLETON_MACROS_H_H

#include <hermes_shm/util/singleton.h>

#define HERMES_SYSTEM_INFO scs::Singleton<hermes::SystemInfo>::GetInstance()
#define HERMES_SYSTEM_INFO_T hermes::SystemInfo*

#define HERMES_MEMORY_REGISTRY scs::Singleton<hermes::ipc::MemoryRegistry>::GetInstance()
#define HERMES_MEMORY_REGISTRY_T hermes::ipc::MemoryRegistry*

#define HERMES_MEMORY_MANAGER scs::Singleton<hermes::ipc::MemoryManager>::GetInstance()
#define HERMES_MEMORY_MANAGER_T hermes::ipc::MemoryManager*

#define HERMES_THREAD_MANAGER scs::Singleton<hermes::ThreadManager>::GetInstance()
#define HERMES_THREAD_MANAGER_T hermes::ThreadManager*

#endif  // include_labstor_constants_data_structure_singleton_macros_h
