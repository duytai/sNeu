#include "mutator.h"
#include "config.h"
#include <stdio.h>
#include <string.h>

Mutator::Mutator(Fuzzer* fuzzer) {
  this->fuzzer = fuzzer;
}

void Mutator::byte_flip(char* mem, u32 len) {
  // u32 exec_tmout = EXEC_TIMEOUT;
  // for (u32 i = 0; i < len; i += 1) {
    // char tmp[len];
    // memcpy(tmp, mem, len);
    // mem[i] ^= 0xFF;
    // auto hnb = this->fuzzer->run_target(tmp, len, exec_tmout);
    // if (hnb) OKF("Found");
  // }
}

// TODO: add checksum to avoid re-flip
void Mutator::mutate() {
  // u32 exec_tmout = EXEC_TIMEOUT;
  // this->fuzzer->update_inst_branches();
  // for (auto fuzzer_pair: this->fuzzer->pairs) {
    // if (fuzzer_pair.min_loss != 255) {
      /*
       * This testcase touch one interesting branch at the least
       * we mutate to generate more testcases for training 
       * */
      // this->byte_flip(fuzzer_pair.mem, fuzzer_pair.len);
    // }
  // }
}
