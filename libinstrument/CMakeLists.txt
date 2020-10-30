set(sources
  afl-llvm-rt.o.c
  config.h
  debug.h
  types.h
  hash.h
)

add_library(llvmrt ${sources})
