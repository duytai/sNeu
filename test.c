#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  char data[44] = { /* zero padding */ };
  fread(data, 1, sizeof(data) - 1, stdin);
  if (data[0] != 'q') return 1;
  if (data[1] != 'w') return 1;
  if (data[2] != 'e') return 1;
  if (data[3] != 'r') return 1;
  if (data[4] != 't') return 1;
  if (data[5] != 'y') return 1;
  if (data[6] != '1') return 1;
  if (data[7] != '1') return 1;
  if (data[8] != '1') return 1;
  if (data[9] != '1') return 1;
  if (data[10] != '1') return 1;
  abort();
  return 0;
}
