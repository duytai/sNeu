# sNeu

This is a neural network assited fuzzer to efficiently generate testcases for AFL

### 1. Requirements

- AFL  : https://github.com/google/AFL
- Rust : https://www.rust-lang.org/tools/install
- Libtorch: https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.2.0.zip
- Clang/Clang++


### 2. Usage

```bash
# build sNeu project
cd sNeu/
mkdir build/ && cd build/
CC=clang CXX=clang++ cmake -DCMAKE_PREFIX_PATH=<path/to/libtorch> ..
```
### 3. Code instrument for c/c++ projects

```bash
# projects containing CMakeLists
mkdir build && cd build/
CC=<path/to/sNeu>/clang.py CXX=<path/to/sNeu>/clang++.py cmake ..
make

# projects containing ./configurate
CC=<path/to/sNeu>/clang.py CXX=<path/to/sNeu>/clang++.py ./configure
```
### 4. Fuzz your programs with sNeu

Fuzz your binary with AFL for 1 hour and continue with sneu
```bash
sneu -i in -o out <path/to/binary/file>
# in/ contains testcases under the folder queue of AFL
# out/ is an empty folder to store test cases
# <path/to/binary/file> syntax is similar to AFL (@@ to read from file and empty to read from stdin)
```
