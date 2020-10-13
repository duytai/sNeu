set(sources
  net.h
  net.cpp
)

include_directories(${TORCH_INCLUDE_DIRS})
add_library(net ${sources})
