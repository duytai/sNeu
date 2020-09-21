# c-branch-distance

This is a simple instrument module for c/c++. It sums up branch distances and save them to disk

### 1. How does it work

It instruments a `.c` file with LLVM Pass `libcmppass.so` and adds a function call `__sn_cmp` to each comparison instruction followed by a conditional `JUMP`. Each comparison instruction has a unique ID. The function `__sn_cmp` is implemented in `cmpcov.c` file where we save branch distances to `.logs/<pid>.cov` file. 

- `cov_reader.py` is an example of reading, accumulating and generating traning label
- `clang.py` is a wrapper of `clang`. It replaces original `clang` to build `c/c++` project without editing `Makefile` or `CMakeLists`


### 2. Usage
- `make` to generate `libcmppass.so` and `cmpcov.o`
- update clang wrapper to `CC` flag, e.g., `CC=path_to_clang.py` and build your project
