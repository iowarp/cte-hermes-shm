#pragma once
#include <string>
#include <utility>
#include <cstdint>

namespace hshm::lbm::utils {

/**
 * @brief Parse a URL into node and service components
 * @param url URL in format "protocol://host:port"
 * @return pair of (host, port)
 */
std::pair<std::string, std::string> parseUrl(const std::string& url);

/**
 * @brief Default buffer size for message operations
 */
constexpr size_t getDefaultBufferSize() { return 1024; }

/**
 * @brief Default completion queue size
 */
constexpr size_t getDefaultCqSize() { return 10; }

} // namespace hshm::lbm::utils 