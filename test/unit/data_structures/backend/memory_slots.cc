/*
 * Copyright (C) 2022  SCS Lab <scslab@iit.edu>,
 * Luke Logan <llogan@hawk.iit.edu>,
 * Jaime Cernuda Garcia <jcernudagarcia@hawk.iit.edu>
 * Jay Lofstead <gflofst@sandia.gov>,
 * Anthony Kougkas <akougkas@iit.edu>,
 * Xian-He Sun <sun@iit.edu>
 *
 * This file is part of hermes
 *
 * hermes is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public
 * License along with this program.  If not, see
 * <http://www.gnu.org/licenses/>.
 */

#include "basic_test.h"

#include <mpi.h>
#include <iostream>
#include "hermes_shm/memory/backend/posix_shm_mmap.h"

using hermes::ipc::PosixShmMmap;

TEST_CASE("MemorySlot") {
  int rank;
  char nonce = 8;
  std::string shm_url = "test_mem_backend";
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  HERMES_ERROR_HANDLE_START()

  PosixShmMmap backend;
  if (rank == 0) {
    {
      std::cout << "Creating SHMEM (rank 0)" << std::endl;
      if(!backend.shm_init(MEGABYTES(1), shm_url)) {
        throw std::runtime_error("Couldn't create backend");
      }
      std::cout << "Backend data: " << (void*)backend.data_ << std::endl;
      std::cout << "Backend sz: " << backend.data_size_ << std::endl;
      memset(backend.data_, nonce, backend.data_size_);
      std::cout << "Wrote backend data" << std::endl;
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank != 0) {
    {
      std::cout << "Attaching SHMEM (rank 1)" << std::endl;
      backend.shm_deserialize(shm_url);
      char *ptr = backend.data_;
      REQUIRE(VerifyBuffer(ptr, backend.data_size_, nonce));
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    {
      std::cout << "Destroying shmem (rank 1)" << std::endl;
      backend.shm_destroy();
    }
  }

  HERMES_ERROR_HANDLE_END()
}
