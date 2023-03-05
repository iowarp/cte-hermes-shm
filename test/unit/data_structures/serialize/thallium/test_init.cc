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
#include "hermes_shm/data_structures/string.h"
#include "hermes_shm/data_structures/serialization/thallium.h"
#include "hermes_shm/types/charbuf.h"
#include <memory>

namespace tl = thallium;
namespace hshm = hermes_shm;
using thallium::request;

std::unique_ptr<tl::engine> client_;

void MainPretest() {
  client_ = std::make_unique<tl::engine>(
    "ofi+sockets",
    THALLIUM_CLIENT_MODE);
}

void MainPosttest() {
  tl::endpoint server = client_->lookup("ofi+sockets://127.0.0.1:8080");
  client_->shutdown_remote_engine(server);
}