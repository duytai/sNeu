#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

void func(int x) {
  if (x > 8989) {
    printf("f\n");
  }
}

int main(int argc, char **argv) {
  if (argc < 230) {
    if (argc > 0) {
      printf("Hello \n");
    }
  }
  if (argc > 100) {
    if (argc < 1000) {
      printf("World\n");
    }
  }
  func(88888);
  return 0;
}
