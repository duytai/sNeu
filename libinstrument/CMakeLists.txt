set(sources
  afl-llvm-rt.o.c
  config.h
  debug.h
  types.h
  hash.h
)
execute_process(COMMAND ${CC} -c ${CMAKE_CURRENT_SOURCE_DIR}/${sources} -o cmpcov.o)
