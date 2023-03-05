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

#include <iostream>
#include "test_init.h"

namespace tl = thallium;
namespace hshm = hermes_shm;

bool started = false;

void StartThalliumServer() {
  tl::engine myEngine("ofi+sockets://127.0.0.1:8080",
                      THALLIUM_SERVER_MODE,
                      true, 1);
  std::cout << "Server running at address " << myEngine.self() << std::endl;
  started = true;
  myEngine.enable_remote_shutdown();
  myEngine.wait_for_finalize();
}

void MainPretest() {
  // Start server thread
  auto lambda = [] () {
    StartThalliumServer();
  };
  auto thread = hshm::ThreadFactory(hshm::ThreadType::kPthread, lambda).Get();
  while (!started) {
    HERMES_THREAD_MANAGER->GetThreadStatic()->Yield();
  }

  // Start client
  tl::engine myEngine("ofi+sockets://127.0.0.1:8080", THALLIUM_CLIENT_MODE);
}

void MainPosttest() {
}