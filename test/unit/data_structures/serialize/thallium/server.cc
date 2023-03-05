//
// Created by lukemartinlogan on 3/5/23.
//

#include "test_init.h"
#include "hermes_shm/data_structures/string.h"
#include "hermes_shm/data_structures/serialization/thallium.h"
#include "hermes_shm/types/charbuf.h"
#include <memory>

std::unique_ptr<tl::engine> server_;

int main() {
  server_ = std::make_unique<tl::engine>(
    "ofi+sockets://127.0.0.1:8080",
    THALLIUM_SERVER_MODE,
    true, 1);
  std::cout << "Server running at address " << server_->self() << std::endl;

  // Test transfer of 0-length string
  auto string_test0 = [](const request &req,
                         hipc::string &text) {
    req.respond(text == "");
  };
  server_->define("string_test0", string_test0);

  // Test transfer of long string
  auto string_test1 = [](const request &req,
                         hipc::string &text) {
    req.respond(text == "012344823723642364723874623");
  };
  server_->define("string_test1", string_test0);

  // Test transfer of 0-length charbuf
  auto charbuf_test0 = [](const request &req,
                          hshm::charbuf &text) {
    req.respond(text == "");
  };
  server_->define("charbuf_test0", string_test0);

  // Test transfer of long charbuf
  auto charbuf_test1 = [](const request &req,
                          hshm::charbuf &text) {
    req.respond(text == "012344823723642364723874623");
  };
  server_->define("charbuf_test1", string_test0);

  // Start daemon
  server_->enable_remote_shutdown();
  server_->wait_for_finalize();
}