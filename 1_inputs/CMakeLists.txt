cmake_minimum_required(VERSION 3.10)
project(LogicSynthesisTool)

# Define working directory (HybridSYN root)
set(WORK_DIR ${CMAKE_SOURCE_DIR}/..)
message(STATUS "WORK_DIR: ${WORK_DIR}")

# Mockturtle strictly requires the C++17 standard to work
set(CMAKE_CXX_STANDARD 17)

# Build executable
add_executable(import import.cpp)

# Use repository-local mockturtle headers (adjust if running CMake from elsewhere)
target_include_directories(import PRIVATE
	${CMAKE_SOURCE_DIR}/mockturtle/include          # for <mockturtle/...>
	${CMAKE_SOURCE_DIR}/mockturtle/lib/lorina        # for <lorina/...>
	${CMAKE_SOURCE_DIR}/mockturtle/lib/fmt           # for <fmt/...> when using bundled fmt
	${CMAKE_SOURCE_DIR}/mockturtle/lib/kitty         # for <kitty/...> when using bundled kitty
	${CMAKE_SOURCE_DIR}/mockturtle/lib/parallel_hashmap # for <parallel_hashmap/phmap.h>
)

# Prefer using system fmt if available; otherwise fall back to bundled headers
find_package(fmt QUIET)
if(TARGET fmt::fmt)
	target_link_libraries(import PRIVATE fmt::fmt)
else()
	# Use header-only mode for bundled fmt to avoid missing link errors
	target_compile_definitions(import PRIVATE FMT_HEADER_ONLY=1)
endif()