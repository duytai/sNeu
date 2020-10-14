set(sources
  types.h
  debug.h
  config.h
  hash.h
  fuzzer.h
  fuzzer.cpp
  testsuit.h
  testsuit.cpp
  net.h
  net.cpp
)

include_directories(${TORCH_INCLUDE_DIRS})
add_library(fuzzer ${sources})
