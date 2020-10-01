# sNeu

This is a neural network assited fuzzer to efficiently generate testcases for AFL

### 2. Usage

```bash
# build a lightweight fuzzer + static libraries
make
# fuzz the example program
./sneu.py -i ../in/ -o ../out/ -b ../bin/ -p programs/example/ 
```
