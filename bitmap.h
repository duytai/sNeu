#include "types.h"
#include "config.h"

static const u8 count_class_lookup8[256] = {
  [0]           = 0,
  [1]           = 1,
  [2]           = 2,
  [3]           = 4,
  [4 ... 7]     = 8,
  [8 ... 15]    = 16,
  [16 ... 31]   = 32,
  [32 ... 127]  = 64,
  [128 ... 255] = 128
};

static u16 count_class_lookup16[65536];

static void init_count_class16(void) {
  u32 b1, b2;
  for (b1 = 0; b1 < 256; b1++)
    for (b2 = 0; b2 < 256; b2++)
      count_class_lookup16[(b1 << 8) + b2] =
        (count_class_lookup8[b1] << 8) |
        count_class_lookup8[b2];
}

static void classify_counts(uint64_t* mem) {
  u32 i = MAP_SIZE >> 3;
  while (i--) {
    /* Optimize for sparse bitmaps. */
    if (unlikely(*mem)) {
      u16 * mem16 = (u16*)mem;
      mem16[0] = count_class_lookup16[mem16[0]];
      mem16[1] = count_class_lookup16[mem16[1]];
      mem16[2] = count_class_lookup16[mem16[2]];
      mem16[3] = count_class_lookup16[mem16[3]];
    }
    mem++;
  }
}

u8 has_new_bits(u8* virgin_map, u8* trace_bits) {
  u64* current = (u64*)trace_bits;
  u64* virgin  = (u64*)virgin_map;
  u32 i = (MAP_SIZE >> 3);
  u8 ret = 0;

  while (i--) {
    if (unlikely(*current) && unlikely(*current & *virgin)) {
      if (likely(ret < 2)) {
        u8* cur = (u8*)current;
        u8* vir = (u8*)virgin;

        /* Looks like we have not found any new bytes yet; see if any non-zero
           bytes in current[] are pristine in virgin[]. */
        if ((cur[0] && vir[0] == 0xff) || (cur[1] && vir[1] == 0xff) ||
            (cur[2] && vir[2] == 0xff) || (cur[3] && vir[3] == 0xff) ||
            (cur[4] && vir[4] == 0xff) || (cur[5] && vir[5] == 0xff) ||
            (cur[6] && vir[6] == 0xff) || (cur[7] && vir[7] == 0xff)) ret = 2;
        else ret = 1;
      }
      *virgin &= ~*current;
    }
    current++;
    virgin++;
  }
  return ret;
}
