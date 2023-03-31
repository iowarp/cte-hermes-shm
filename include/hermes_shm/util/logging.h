//
// Created by lukemartinlogan on 3/31/23.
//

#ifndef HERMES_SHM_INCLUDE_HERMES_SHM_UTIL_LOGGING_H_
#define HERMES_SHM_INCLUDE_HERMES_SHM_UTIL_LOGGING_H_

#include <unistd.h>
#include <climits>

#include <vector>
#include <iostream>
#include <filesystem>
#include "formatter.h"
#include "singleton.h"

namespace hshm {

/**
 * The verbosity of the logger indicates how much data will be printed.
 * The higher the verbosity, the more data will be printed and the lower
 * performance will be.
 * */
#ifndef HERMES_LOG_VERBOSITY
#define HERMES_LOG_VERBOSITY 10
#endif

/** Logging levels */
#define kDebug 10  /**< Low-priority debugging information*/
// ... may want to add more levels here
#define kInfo 0    /**< Useful information the user should know */

/** Simplify access to Logger singleton */
#define HERMES_LOG hshm::EasySingleton<hshm::Logger>::GetInstance()

/**
 * LOG_LEVEL indicates the priority of the log.
 * LOG_LEVEL 0 is maximum priority
 * LOG_LEVEL 10 is considered debugging priority.
 * */
#define HLOG(LOG_LEVEL, ...) \
  if constexpr(LOG_LEVEL <= HERMES_LOG_VERBOSITY) { \
    HERMES_LOG->Log(LOG_LEVEL, __func__, __LINE__, __VA_ARGS__); \
  }

class Logger {
 public:
  std::string exe_path_;
  std::string exe_name_;
  int verbosity_;

 public:
  Logger() {
    GetExePath();
    exe_name_ = std::filesystem::path(exe_path_).lexically_normal();
    verbosity_ = kDebug;
    auto verbosity_env = getenv("HERMES_LOG_VERBOSITY");
    if (verbosity_env && strlen(verbosity_env)) {
      try {
        std::stringstream(verbosity_env) >> verbosity_;
      } catch (...) {
        verbosity_ = kDebug;
      }
    }
  }

  void SetVerbosity(int LOG_LEVEL) {
    verbosity_ = LOG_LEVEL;
  }

  template<typename ...Args>
  void Log(int LOG_LEVEL,
           const char *func,
           int line,
           const char *fmt,
           Args&& ...args) {
    if (LOG_LEVEL > verbosity_) { return; }
    std::string text =
      hshm::Formatter::format(fmt, std::forward<Args>(args)...);
    int tid = gettid();
    std::cerr << hshm::Formatter::format(
      "{} \\033]8;url=file://{}&line={}\\a{}:{} {}\\033]8;;\\a {}",
      tid, exe_path_, line,
      exe_name_, line, func, text);
  }

 private:
  void GetExePath() {
    std::vector<char> exe_path(PATH_MAX, 0);
    if (readlink("/proc/self/exe", exe_path.data(), PATH_MAX - 1) == -1) {
      std::cerr
        << "Could not introspect the name of this program for logging"
        << std::endl;
    }
    exe_path_ = std::string(exe_path.data());
  }
};

}  // namespace hshm

#endif //HERMES_SHM_INCLUDE_HERMES_SHM_UTIL_LOGGING_H_
