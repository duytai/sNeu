// trace-pc-guard-cb.cc
#include <stdint.h>
#include <stdio.h>

#define u8 uint8_t
#define u16 uint16_t
#define u32 uint32_t
#define u64 uint64_t

#define cLGN "\x1b[1;92m"
#define cRST "\x1b[0m"

#define SAYF(x...) printf(x)
#define OKF(x...) do { \
  SAYF(cLGN "[+] " cRST x); \
  SAYF(cRST "\n"); \
} while (0)

#define IGNORE(name) \
  void name {\
    fprintf(stdout, "%s\n", __func__); \
  }

#define COV(name, len, ty) \
  void name(ty arg1, ty arg2) {\
    handle_cmp(arg1, arg2, len, __builtin_return_address(0)); \
  }

static u8 count_matching_bytes(u32 len, u64 x, u64 y) {
  u32 i;
  for (i = 0; i < len; i++) {
    if (((x >> (i * 8)) & 0xff) != ((y >> (i * 8)) & 0xff)) {
      break;
    }
  }
  return i;
}

static void handle_cmp(u64 x, u64 y, u32 len, void* pc) {
  u8 diff_bytes = len - count_matching_bytes(len, x, y);
  OKF("diff: %d\n", diff_bytes);
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
IGNORE(__sanitizer_cov_trace_pc_guard_init(u32 *start, u32 *stop))
IGNORE(__sanitizer_cov_trace_pc_guard(u32 *guard))
