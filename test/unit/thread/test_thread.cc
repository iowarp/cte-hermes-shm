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

#include "basic_test.h"
#include "omp.h"
#include "hermes_shm/thread/thread_manager.h"

TEST_CASE("TestPthread") {
  HSHM_THREAD_MANAGER->SetThreadModel(hshm::ThreadType::kPthread);
  int x = 25, y = 60;
  int z = -1;
  auto thread = hshm::Pthread([&x, &y, &z]() {
    z = y + x;
  });
  thread.Join();
  REQUIRE(z == y + x);
}

TEST_CASE("TestArgobots") {
  HSHM_THREAD_MANAGER->SetThreadModel(hshm::ThreadType::kArgobots);
  int x = 25, y = 60;
  int z = -1;
  ABT_xstream xstream;
  ABT_xstream_create(ABT_SCHED_NULL, &xstream);
  auto thread = hshm::Argobots(xstream, [&x, &y, &z]() {
    z = y + x;
  });
  thread.Join();
  REQUIRE(z == y + x);
}
