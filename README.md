# sNeu

This is a neural network assited fuzzer to efficiently generate testcases for AFL

### 1. Requirements

- AFL  : https://github.com/google/AFL
- Rust : https://www.rust-lang.org/tools/install
- Clang/Clang++


### 2. Usage

```bash
# build a lightweight fuzzer + static libraries
make

# create dirs for AFL
mkdir in/ out/ bin/
echo "{}" > in/1.txt

# mutate testcases with Neural Network
./sneu.py -i in/ -o out/ -b bin/ -p programs/example/

# run AFL in master-slave mode
afl-fuzz -i in/ -o out/ -M fuzzer01 bin/target_afl
```