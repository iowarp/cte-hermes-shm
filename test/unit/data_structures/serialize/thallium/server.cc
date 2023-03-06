//
// Created by lukemartinlogan on 3/5/23.
//

#include "test_init.h"
#include "hermes_shm/data_structures/string.h"
#include "hermes_shm/data_structures/serialization/thallium.h"
#include "hermes_shm/types/charbuf.h"
#include <memory>

std::unique_ptr<tl::engine> client_;
std::unique_ptr<tl::engine> server_;

int main() {
  // Pretest
  ServerPretest<hipc::StackAllocator>();

  // Create thallium server
  server_ = std::make_unique<tl::engine>(
    kServerName,
    THALLIUM_SERVER_MODE,
    true, 1);
  std::cout << "Server running at address " << server_->self() << std::endl;

  // Test transfer of 0-length string
  auto string_test0 = [](const request &req,
                         hipc::string &text) {
    req.respond(text == "");
  };
  server_->define(kStringTest0, string_test0);

  // Test transfer of long string
  auto string_test1 = [](const request &req,
                         hipc::string &text) {
    req.respond(text == kTestString);
  };
  server_->define(kStringTestLarge, string_test0);

  // Test transfer of 0-length charbuf
  auto charbuf_test0 = [](const request &req,
                          hshm::charbuf &text) {
    req.respond(text == "");
  };
  server_->define("charbuf_test0", string_test0);

  // Test transfer of long charbuf
  auto charbuf_test1 = [](const request &req,
                          hshm::charbuf &text) {
    req.respond(text == kTestString);
  };
  server_->define("charbuf_test1", string_test0);

  // Start daemon
  server_->enable_remote_shutdown();
  server_->wait_for_finalize();
}