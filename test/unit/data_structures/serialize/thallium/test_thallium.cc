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
#include "test_init.h"
#include "hermes_shm/data_structures/string.h"

TEST_CASE("SerializeString") {
  tl::endpoint server = client_->lookup(kServerName);
  tl::remote_procedure string0_proc = client_->define(kStringTest0);
  tl::remote_procedure string_large_proc = client_->define(kStringTestLarge);

  hipc::string empty_str("");
  hipc::string large_str(kTestString);

  REQUIRE(string0_proc.on(server)(empty_str));
  REQUIRE(string_large_proc.on(server)(large_str));
}

TEST_CASE("SerializeCharBuf") {
  tl::endpoint server = client_->lookup(kServerName);
  tl::remote_procedure string0_proc = client_->define(kCharbufTest0);
  tl::remote_procedure string_large_proc = client_->define(kCharbufTestLarge);

  hipc::string empty_str("");
  hipc::string large_str(kTestString);

  REQUIRE(string0_proc.on(server)(empty_str));
  REQUIRE(string_large_proc.on(server)(large_str));
}

TEST_CASE("SerializeVector") {
}