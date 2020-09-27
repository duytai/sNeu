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

#define OKF(x...) do { \
  SAYF(cLGN "[+] " cRST x); \
  SAYF(cRST "\n"); \
} while (0)

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

#define BYTES(num_bytes, value) ({ \
  u8 _tmp[num_bytes]; \
  for (int i = 0; i < num_bytes; i++) { \
    _tmp[i] = (value >> (i * 8)) & 0xFF; \
  } \
  _tmp; \
})

#define IGNORE(name) \
  void name {\
    OKF("%s", __func__); \
  }

#define COV(name, len, ty) \
  void name(ty x, ty y) {\
    u64 left = (u64) x;\
    u64 right = (u64) y;\
    u32 pc = (u32) __builtin_return_address(0);\
    fwrite(BYTES(1, len), 1, 1, cov_file);\
    fwrite(BYTES(4, pc), 4, 1, cov_file);\
    fwrite(BYTES(8, left), 8, 1, cov_file);\
    fwrite(BYTES(8, right), 8, 1, cov_file);\
    OKF("%s:%d: %lu - %lu", __func__, pc, left, right);\
  }

#define LOG_DIR ".logs"

static FILE* cov_file = NULL;

COV(__sanitizer_cov_trace_cmp1, 1, u8)
COV(__sanitizer_cov_trace_cmp2, 2, u16)
COV(__sanitizer_cov_trace_cmp4, 4, u32)
COV(__sanitizer_cov_trace_cmp8, 8, u64)
COV(__sanitizer_cov_trace_const_cmp1, 1, u8)
COV(__sanitizer_cov_trace_const_cmp2, 2, u16)
COV(__sanitizer_cov_trace_const_cmp4, 4, u32)
COV(__sanitizer_cov_trace_const_cmp8, 8, u64)

IGNORE(__sanitizer_cov_trace_switch(u64 Val, u64 *Cases))
IGNORE(__sanitizer_cov_trace_div4(u32 Val))
IGNORE(__sanitizer_cov_trace_div8(u64 Val))
IGNORE(__sanitizer_cov_trace_gep(uintptr_t Idx))

IGNORE(__sanitizer_cov_trace_pc_guard_init(u32* start, u32* stop))
IGNORE(__sanitizer_cov_trace_pc_guard(u32* guard))

__attribute__((constructor)) static void init() {
  static u8 init_done;
  struct stat st;
  if (!init_done) {
    if (stat(LOG_DIR, &st) == -1) mkdir(LOG_DIR, 0700);
    char* filename = FORMAT("%s/%d.cov", LOG_DIR, getpid());
    cov_file = fopen(filename, "a+");
    if (cov_file == NULL) FATAL("fopen() failed");
    init_done = 1;
    free(filename);
  }
};
