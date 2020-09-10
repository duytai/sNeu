# c-branch-distance

This is a simple instrument module for c/c++ and rust programs. It summarizes branch distances and save them to disk in the form
of standard `.sancov` files. It is based on [SanitizerCoverage](https://clang.llvm.org/docs/SanitizerCoverage.html)

### 1. How does it work

It sumarizes distances in `cmp` instructions and write them to `.sancov` when `__sanitizer_cov_trace_pc_guard` is invoked.
The branch distance of each `cmp` instruction is `8 - matching bytes between two operands`

### 2. Usage

```bash
make
ASAN_OPTIONS="coverage=1, coverage_dir=logs/" ./test
```
It writes coverage to `.sancov` located at `logs/`.
