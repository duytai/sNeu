execute_process(
  COMMAND
  ${CMAKE_C_COMPILER} -c ${CMAKE_CURRENT_SOURCE_DIR}/afl-llvm-rt.o.c -o cmpcov.o
)
