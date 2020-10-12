cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(sneu)

find_package(Torch REQUIRED)

add_executable(sneu sneu.cpp)

add_subdirectory(libfuzzer)

target_link_libraries(sneu "${TORCH_LIBRARIES}" fuzzer)
set_property(TARGET sneu PROPERTY CXX_STANDARD 17)
