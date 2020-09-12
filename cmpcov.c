#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <ctype.h>
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
  if (asan_options.debug != 1) break; \
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

#define IGNORE(name) \
  void name {\
    OKF("%s", __func__); \
  }

#define COV(name, len, ty) \
  void name(ty x, ty y) {\
    u64 diff = x > y ? (u64)x - (u64)y : (u64)y - (u64)x;\
    diff_value = diff > U32_MAX ? U32_MAX : diff;\
  }

#define FORMAT(_str...) ({ \
    char * _tmp; \
    s32 _len = snprintf(NULL, 0, _str); \
    if (_len < 0) FATAL("snprintf() failed"); \
    _tmp = malloc(_len + 1); \
    snprintf(_tmp, _len + 1, _str); \
    _tmp; \
  })

#define U32_MAX 0xFFFFFFFF

typedef struct {
  u32 coverage;
  u32 debug;
  char* coverage_dir;
} AsanOptions;

static AsanOptions asan_options = { .coverage = 0, .coverage_dir = ".", .debug = 0 };
static u32 cur_fd;
static u32 diff_value = U32_MAX;

static char * trim_space(char *str) {
  char *end;
  while (isspace(*str)) {
    str = str + 1;
  }
  end = str + strlen(str) - 1;
  while (end > str && isspace(*end)) {
    end = end - 1;
  }
  *(end+1) = '\0';
  return str;
}

static void parse_asan_options() {
  char* asan_options_str = getenv("ASAN_OPTIONS");
  if (asan_options_str) {
    char* token;
    while ((token = strsep(&asan_options_str, ",")) != NULL) {
      char* key = strsep(&token, "=");
      char* value = strsep(&token, "=");
      if (key != NULL && value != NULL) {
        if (strcmp(trim_space(key), "coverage") == 0) {
          asan_options.coverage = atoi(value);
        }
        if (strcmp(trim_space(key), "coverage_dir") == 0) {
          asan_options.coverage_dir = trim_space(value);
        }
        if (strcmp(trim_space(key), "debug") == 0) {
          asan_options.debug = atoi(value);
        }
      }
    }
  }
}

static void prepare_files() {
  struct stat st;
  if (asan_options.coverage == 1) {
    /* create directory */
    if (stat(asan_options.coverage_dir, &st) == -1) {
      mkdir(asan_options.coverage_dir, 0700);
    }
    /* write header */
    char header[] = {0x64, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xBF, 0xC0};
    char* fpath = FORMAT("%s/cmp.%d.sancov", asan_options.coverage_dir, getpid());
    /* write header */
    cur_fd = open(fpath, O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
    if (cur_fd < 0) FATAL("open() failed");
    s32 offset = lseek(cur_fd, 0, SEEK_END);
    if (offset < 0) FATAL("lseek() failed");
    /* empy file */
    if (offset == 0) write(cur_fd, header, 8);
    free(fpath);
  }
}

static void append_sancov(u32 pc, u32 diff_value) {
  if (asan_options.coverage == 1) {
    u8 data[] =  {0, 0, 0, 0, 0, 0, 0, 0, 0};
    data[0] = pc & 0xFF;
    data[1] = (pc >> 8) & 0xFF;
    data[2] = (pc >> 16) & 0xFF;
    data[3] = (pc >> 24) & 0xFF;
    //
    data[4] = diff_value & 0xFF;
    data[5] = (diff_value >> 8) & 0xFF;
    data[6] = (diff_value >> 16) & 0xFF;
    data[7] = (diff_value >> 24) & 0xFF;
    // write
    write(cur_fd, data, 8);
  }
}

void __sanitizer_cov_trace_pc_guard_init(u32 *start, u32 *stop) {
  parse_asan_options();
  prepare_files();
  if (start == stop || *start) return;
  while (start < stop) {
    append_sancov((u32) start, diff_value);
    start ++;
  }
}

void __sanitizer_cov_trace_pc_guard(u32 *guard) {
  append_sancov((u32) guard, diff_value);
  diff_value = 0;
}

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
