#ifndef HERMES_SHM_DATA_STRUCTURES_IPC_STRINGSTREAM_H_
#define HERMES_SHM_DATA_STRUCTURES_IPC_STRINGSTREAM_H_

#include <sstream>
#include <string>
#include <vector>

namespace hhshm::ipc {

template <int SSO>
class stringstream {
 public:
  stringstream() = default;
  ~stringstream() = default;

  template <typename T>
  stringstream& operator<<(const T& value) {
    return *this;
  }

  void clear() {}
};

}  // namespace hhshm::ipc

#endif  // HERMES_SHM_DATA_STRUCTURES_IPC_STRINGSTREAM_H_