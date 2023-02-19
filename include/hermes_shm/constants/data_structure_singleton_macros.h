#ifndef HERMES_INCLUDE_HERMES_CONSTANTS_DATA_STRUCTURE_SINGLETON_MACROS_H_H
#define HERMES_INCLUDE_HERMES_CONSTANTS_DATA_STRUCTURE_SINGLETON_MACROS_H_H

#include <hermes_shm/util/singleton.h>

#define HERMES_SYSTEM_INFO scs::Singleton<hermes::SystemInfo>::GetInstance()
#define HERMES_SYSTEM_INFO_T hermes::SystemInfo*

#define HERMES_MEMORY_MANAGER scs::Singleton<hermes::ipc::MemoryManager>::GetInstance()
#define HERMES_MEMORY_MANAGER_T hermes::ipc::MemoryManager*

#define HERMES_THREAD_MANAGER scs::Singleton<hermes::ThreadManager>::GetInstance()
#define HERMES_THREAD_MANAGER_T hermes::ThreadManager*

#endif  // include_labstor_constants_data_structure_singleton_macros_h
