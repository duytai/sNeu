cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(sneu)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -fsanitize=address -std=c++17")

find_package(Torch REQUIRED)

add_executable(sneu sneu.cpp)

include_directories(".")

add_subdirectory(libfuzzer)
add_subdirectory(libutil)

target_link_libraries(sneu "${TORCH_LIBRARIES}" fuzzer) 
