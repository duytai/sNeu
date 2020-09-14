#include <stdio.h>
#include <stdint.h>

#define u8 uint8_t
#define u16 uint16_t
#define u32 uint32_t
#define u64 uint64_t

void __sn_cmp(u8 type_size, u16 id, u64 x, u64 y) {
  printf("id: %d | size: %d | x:%lu | y:%lu\n", id, type_size, x, y);
}
