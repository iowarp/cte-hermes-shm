#ifndef HERMES_INCLUDE_HERMES_CONSTANTS_SINGLETON_MACROS_H_H
#define HERMES_INCLUDE_HERMES_CONSTANTS_SINGLETON_MACROS_H_H

#include <hermes_shm/util/singleton.h>

#define HERMES_IPC_MANAGER scs::Singleton<hermes::IpcManager>::GetInstance()
#define HERMES_IPC_MANAGER_T hermes::IpcManager*

#define HERMES_CONFIGURATION_MANAGER scs::Singleton<hermes::ConfigurationManager>::GetInstance()
#define HERMES_CONFIGURATION_MANAGER_T hermes::ConfigurationManager*

#endif  // include_labstor_constants_singleton_macros_h
