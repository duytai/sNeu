set(sources
  types.h
  debug.h
  config.h
  hash.h
  fuzzer.h
  fuzzer.cpp
  testsuit.h
  testsuit.cpp
)

add_library(fuzzer ${sources})
