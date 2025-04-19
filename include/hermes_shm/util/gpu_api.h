#include "hermes_shm/constants/macros.h"
#include "hermes_shm/util/logging.h"

namespace hshm {

class GpuApi {
 public:
  static void SetDevice(int gpu_id) {
#if defined(HSHM_ENABLE_CUDA)
    CUDA_ERROR_CHECK(cudaSetDevice(gpu_id));
#elif defined(HSHM_ENABLE_ROCM)
    HIP_ERROR_CHECK(hipSetDevice(gpu_id));
#endif
  }

  static int GetDeviceCount() {
    int ngpu;
#ifdef HSHM_ENABLE_ROCM
    HIP_ERROR_CHECK(hipGetDeviceCount(&ngpu));
#endif
#ifdef HSHM_ENABLE_CUDA
    CUDA_ERROR_CHECK(cudaGetDeviceCount(&ngpu));
#endif
    return ngpu;
  }

  static void Synchronize() {
#ifdef HSHM_ENABLE_ROCM
    HIP_ERROR_CHECK(hipDeviceSynchronize());
#endif
#ifdef HSHM_ENABLE_CUDA
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
#endif
  }
};

}  // namespace hshm