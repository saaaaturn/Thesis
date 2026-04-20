cmake_minimum_required(VERSION 3.10)
project(FeatureExtraction)

# Mockturtle requires C++17
set(CMAKE_CXX_STANDARD 17)

set(MOCKTURTLE_INCLUDE_DIRS
	${CMAKE_SOURCE_DIR}/../1_inputs/mockturtle/include          # for <mockturtle/...>
	${CMAKE_SOURCE_DIR}/../1_inputs/mockturtle/lib/lorina        # for <lorina/...>
	${CMAKE_SOURCE_DIR}/../1_inputs/mockturtle/lib/fmt           # for <fmt/...>
	${CMAKE_SOURCE_DIR}/../1_inputs/mockturtle/lib/kitty         # for <kitty/...>
	${CMAKE_SOURCE_DIR}/../1_inputs/mockturtle/lib/parallel_hashmap # for <parallel_hashmap/phmap.h>
)

# Build executable
add_executable(extract_features extract_features.cpp)
add_executable(interface mig_interface.cpp)

# Include mockturtle headers from 1_inputs folder
target_include_directories(extract_features PRIVATE ${MOCKTURTLE_INCLUDE_DIRS})
target_include_directories(interface PRIVATE ${MOCKTURTLE_INCLUDE_DIRS})

# Prefer system fmt; fall back to header-only
find_package(fmt QUIET)
if(TARGET fmt::fmt)
	target_link_libraries(extract_features PRIVATE fmt::fmt)
	target_link_libraries(interface PRIVATE fmt::fmt)
else()
	target_compile_definitions(extract_features PRIVATE FMT_HEADER_ONLY=1)
	target_compile_definitions(interface PRIVATE FMT_HEADER_ONLY=1)
endif()

target_compile_definitions(interface PRIVATE FMT_HEADER_ONLY=1)
