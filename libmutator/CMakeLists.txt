set(sources
  mutator.h
  mutator.cpp
)

add_library(mutator ${sources})
target_link_libraries(mutator PUBLIC fuzzer)
