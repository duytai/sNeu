project(libfuzzer)

set(sources
  types.h
  debug.h
  config.h
  hash.h
  fuzzer.h
  fuzzer.cpp
)

add_library(fuzzer STATIC ${sources})
