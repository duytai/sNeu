cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(sneu)

find_package(Torch REQUIRED)

add_executable(sneu sneu.cpp)

include_directories(".")

add_subdirectory(libfuzzer)
add_subdirectory(libmutator)

target_link_libraries(sneu "${TORCH_LIBRARIES}" mutator fuzzer)
set_property(TARGET sneu PROPERTY CXX_STANDARD 17)
