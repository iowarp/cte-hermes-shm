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

#ifndef HSHM_SHM_INCLUDE_HSHM_SHM_UTIL_LOGGING_H_
#define HSHM_SHM_INCLUDE_HSHM_SHM_UTIL_LOGGING_H_

#include <climits>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "formatter.h"
#include "hermes_shm/introspect/system_info.h"
#include "singleton.h"
#include "timer.h"

namespace hshm {

/** Prints log verbosity at compile time */
#define XSTR(s) STR(s)
#define STR(s) #s
// #pragma message XSTR(HSHM_LOG_EXCLUDE)

/** Simplify access to Logger singleton */
#define HSHM_LOG hshm::CrossSingleton<hshm::Logger>::GetInstance()

/** Max number of log codes */
#define HSHM_MAX_LOGGING_CODES 256

/** Given Information Logging levels */
#ifndef kInfo
#define kInfo 251 /**< Useful information the user should know */
#endif
#ifndef kWarning
#define kWarning 252 /**< Something might be wrong */
#endif
#ifndef kError
#define kError 253 /**< A non-fatal error has occurred */
#endif
#ifndef kFatal
#define kFatal 254 /**< A fatal error has occurred */
#endif
#ifndef kDebug /**< Low-priority debugging information*/
#ifndef HSHM_DEBUG
#define kDebug -1
#else
#define kDebug 255
#endif
#endif

/**
 * Hermes Print. Like printf, except types are inferred
 * */
#define HIPRINT(...) HSHM_LOG->Print(__VA_ARGS__)

/**
 * Hermes SHM Log
 * */
#define HLOG(LOG_CODE, SUB_CODE, ...)                                 \
  do {                                                                \
    if constexpr (LOG_CODE >= 0 && SUB_CODE >= 0) {                   \
      HSHM_LOG->Log<LOG_CODE, SUB_CODE>(__FILE__, __func__, __LINE__, \
                                        __VA_ARGS__);                 \
    }                                                                 \
  } while (false)

/** Hermes info log */
#define HILOG(SUB_CODE, ...) HLOG(kInfo, SUB_CODE, __VA_ARGS__)

/** Hermes error log */
#define HELOG(LOG_CODE, ...) HLOG(LOG_CODE, LOG_CODE, __VA_ARGS__)

/** Periodic info log */
#define HILOG_PERIODIC(SUB_CODE, IDX, NSEC, ...)                            \
  do {                                                                      \
    HSHM_PERIODIC(IDX)->Run(NSEC, [&]() { HILOG(SUB_CODE, __VA_ARGS__); }); \
  } while (false)

class Logger {
 public:
  FILE *fout_;
  bool disabled_[HSHM_MAX_LOGGING_CODES];

 public:
  HSHM_CROSS_FUN
  Logger() {
#if HSHM_IS_HOST
    memset(disabled_, 0, sizeof(disabled_));
    // exe_name_ = std::filesystem::path(exe_path_).filename().string();
    std::string verbosity_env = hshm::SystemInfo::Getenv(
        "HSHM_LOG_EXCLUDE", hshm::Unit<size_t>::Megabytes(1));
    if (!verbosity_env.empty()) {
      std::vector<int> verbosity_levels;
      std::string verbosity_str(verbosity_env);
      std::stringstream ss(verbosity_str);
      std::string item;
      while (std::getline(ss, item, ',')) {
        int code = std::stoi(item);
        DisableCode(code);
      }
    }

    std::string env = hshm::SystemInfo::Getenv(
        "HSHM_LOG_OUT", hshm::Unit<size_t>::Megabytes(1));
    if (env.empty()) {
      fout_ = nullptr;
    } else {
      fout_ = fopen(env.c_str(), "w");
    }
#endif
  }

  HSHM_CROSS_FUN
  void DisableCode(int code) {
    if (code >= 0 && code < HSHM_MAX_LOGGING_CODES) {
      disabled_[code] = true;
    }
  }

  template <typename... Args>
  HSHM_CROSS_FUN void Print(const char *fmt, Args &&...args) {
#if HSHM_IS_HOST

    std::string msg = hshm::Formatter::format(fmt, std::forward<Args>(args)...);
    int tid = SystemInfo::GetTid();
    std::string out = hshm::Formatter::format("{}\n", msg);
    std::cout << out;
    if (fout_) {
      fwrite(out.data(), 1, out.size(), fout_);
    }
#endif
  }

  template <int LOG_CODE, int SUB_CODE, typename... Args>
  HSHM_CROSS_FUN void Log(const char *path, const char *func, int line,
                          const char *fmt, Args &&...args) {
#if HSHM_IS_HOST
    if (disabled_[LOG_CODE] || disabled_[SUB_CODE]) {
      return;
    }
    std::string level;
    switch (LOG_CODE) {
      case kInfo: {
        level = "INFO";
        break;
      }
      case kWarning: {
        level = "WARNING";
        break;
      }
      case kError: {
        level = "ERROR";
        break;
      }
      case kFatal: {
        level = "FATAL";
        break;
      }
      default: {
        level = "WARNING";
        break;
      }
    }

    std::string msg = hshm::Formatter::format(fmt, std::forward<Args>(args)...);
    int tid = SystemInfo::GetTid();
    std::string out = hshm::Formatter::format("{}:{} {} {} {} {}\n", path, line,
                                              level, tid, func, msg);
    if (LOG_CODE == kInfo) {
      std::cout << out;
      fflush(stdout);
    } else {
      std::cerr << out;
      fflush(stderr);
    }
    if (fout_) {
      fwrite(out.data(), 1, out.size(), fout_);
    }
    if (LOG_CODE == kFatal) {
      exit(1);
    }
#endif
  }
};

}  // namespace hshm

#endif  // HSHM_SHM_INCLUDE_HSHM_SHM_UTIL_LOGGING_H_
