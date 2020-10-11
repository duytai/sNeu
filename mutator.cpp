#include "mutator.h"
#include "config.h"
#include <stdio.h>
#include <string.h>
#include <vector>

Mutator::Mutator(Fuzzer* fuzzer) {
  this->fuzzer = fuzzer;
}

void Mutator::byte_flip(vector<char>& mem) {
  for (size_t i = 0; i < mem.size(); i++) {
    mem[i] ^= 0xFF;
    this->fuzzer->run_target(mem, EXEC_TIMEOUT);
    if (this->fuzzer->hnb) {
      OKF("Found");
    }
    mem[i] ^= 0xFF;
  }
}

// TODO: add checksum to avoid re-flip
void Mutator::mutate() {
  auto total_execs = this->fuzzer->total_execs;

  this->fuzzer->update_inst_branches();
  for (u32 i = 0; i < total_execs; i += 1) {
    auto fuzz_pair = this->fuzzer->pairs[i];
    if (fuzz_pair.min_loss != 255) {
      /*
       * This testcase touch one interesting branch at the least
       * we mutate to generate more testcases for training 
       * */
      ACTF("byte_flip %d", (u32)fuzz_pair.mem.size());
      this->byte_flip(fuzz_pair.mem);
    }
  }
  this->fuzzer->update_inst_branches();
  OKF("total_execs: %d", this->fuzzer->total_execs);

  // u32 total_inputs = 0;
  // for (auto fuzz_pair: this->fuzzer->pairs) {
    // if (fuzz_pair.min_loss != 255) {
      // total_inputs += 1;
    // }
  // }
  // OKF("total_inputs: %d", total_inputs);
}
