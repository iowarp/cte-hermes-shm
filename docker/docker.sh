#!/bin/bash

# Build cte-hermes-shm Docker image

# Get the project root directory (parent of docker folder)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"

# Build the Docker image
docker build --no-cache -t iowarp/cte-hermes-shm:latest -f "${SCRIPT_DIR}/local.Dockerfile" "${PROJECT_ROOT}"
