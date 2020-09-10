#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  char* dict[] = {"hello", "world", "11", "22", "33"};
  uint8_t bytes[] = {1,2,3,4,4,4,4,4,4 };
  if (argc < 10) {
    for (int i = 0; i < 10; i++) {
      if (rand() % 5 == 1) {
        fprintf(stdout, "hit 5\n");
      };
      if (rand() % 7 == 1) {
        fprintf(stdout, "hit 7\n");
      };
      if (strcmp(dict[rand() % 5], "11")) {
        fprintf(stdout, "hit 11\n");
      }
      if (bytes[rand() % 9] == 99) {
        fprintf(stdout, "hit 99\n");
      }
    }
  }
  fprintf(stdout, "argv[0] = %s\n", argv[0]);
  return 0;
}
