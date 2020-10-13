#include <libfuzzer/config.h>
#include <libmutator/mutator.h>
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

// TODO: add checksum to avoid refliping
void Mutator::mutate() {
  // auto total_execs = this->fuzzer->total_execs;
//
  // this->fuzzer->update_inst_branches();
  // OKF("total_execs: %d", total_execs);

  // for (u32 i = 0; i < total_execs; i += 1) {
    // auto fuzz_pair = this->fuzzer->pairs[i];
    // if (fuzz_pair.min_loss != 255) {
      /*
       * This testcase touch one interesting branch at the least
       * we mutate to generate more testcases for training 
       * */
      // ACTF("byte_flip %d", (u32)fuzz_pair.mem.size());
      // this->byte_flip(fuzz_pair.mem);
    // }
  // }

}
