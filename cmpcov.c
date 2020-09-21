#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>

#define u8 uint8_t
#define u16 uint16_t
#define u32 uint32_t
#define u64 uint64_t
#define s32 int32_t

#define cLGN "\x1b[1;92m"
#define cRST "\x1b[0m"
#define bSTOP "\x0f"
#define RESET_G1 "\x1b)B"
#define CURSOR_SHOW "\x1b[?25h"
#define cLRD "\x1b[1;91m"
#define cBRI "\x1b[1;97m"

#define SAYF(x...) printf(x)

#define FATAL(x...) do { \
  SAYF(bSTOP RESET_G1 CURSOR_SHOW cRST cLRD "\n[-] PROGRAM ABORT : " \
       cBRI x); \
  SAYF(cLRD "\n         Location : " cRST "%s(), %s:%u\n\n", \
       __FUNCTION__, __FILE__, __LINE__); \
  exit(1); \
} while (0)

#define FORMAT(_str...) ({ \
  char * _tmp; \
  s32 _len = snprintf(NULL, 0, _str); \
  if (_len < 0) FATAL("snprintf() failed"); \
  _tmp = malloc(_len + 1); \
  snprintf(_tmp, _len + 1, _str); \
  _tmp; \
})

#define to_bytes(num_bytes, value) ({ \
  u8 _tmp[num_bytes]; \
  for (int i = 0; i < num_bytes; i++) { \
    _tmp[i] = (value >> (i * 8)) & 0xFF; \
  } \
  _tmp; \
})

#define LOG_DIR ".logs"

/* 
 * type_size: 8, 16, 32, 64 
 * */
FILE* cov_file = NULL;

void __sn_cmp(u8 type_size, u16 branch_id, u64 left_value, u64 right_value) {
  fwrite(to_bytes(1, type_size), 1, 1, cov_file);
  fwrite(to_bytes(2, branch_id), 2, 1, cov_file);
  fwrite(to_bytes(8, left_value), 8, 1, cov_file);
  fwrite(to_bytes(8, right_value), 8, 1, cov_file);
}

__attribute__((constructor)) static void init() {
  static u8 init_done;
  struct stat st;
  if (!init_done) {
    if (stat(LOG_DIR, &st) == -1) mkdir(LOG_DIR, 0700);
    cov_file = fopen(FORMAT("%s/%d.cov", LOG_DIR, getpid()), "a+");
    if (cov_file == NULL) FATAL("fopen() failed");
    init_done = 1;
  }
};
