cmake_minimum_required(VERSION 3.27)
project(ferrum_mlx_sdpa_bridge LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT APPLE)
  message(FATAL_ERROR "This diagnostic bridge measures the MLX Metal backend")
endif()
# Set CMAKE_PREFIX_PATH to `venv/bin/python -m mlx --cmake-dir`, or set
# MLX_DIR to that prefix's share/cmake/MLX. Never download or rebuild MLX here.
find_package(MLX CONFIG REQUIRED)
file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/bridge.cpp" BRIDGE_SOURCE_SHA256)
add_library(ferrum_mlx_sdpa_bridge SHARED bridge.cpp)
target_link_libraries(ferrum_mlx_sdpa_bridge PRIVATE mlx)
target_compile_definitions(ferrum_mlx_sdpa_bridge PRIVATE
  FERRUM_MLX_BRIDGE_SOURCE_SHA256="${BRIDGE_SOURCE_SHA256}")
set_target_properties(ferrum_mlx_sdpa_bridge PROPERTIES
  CXX_VISIBILITY_PRESET default BUILD_WITH_INSTALL_RPATH FALSE)
