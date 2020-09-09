// trace-pc-guard-example.cc
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  if (argc < 10) {
    for (int i = 0; i < 100; i++) {
      if (rand() % 5 == 1) return 1;
      if (rand() % 7 == 1) return 1;
    }
  }
  fprintf(stdout, "argv[0] = %s\n", argv[0]);
  return 0;
}
