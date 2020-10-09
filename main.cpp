#include "fuzzer.h"
#include "debug.h"

#include <stdio.h>

int main() {
  Fuzzer fuzzer;
  fuzzer.setup_fds();
}
