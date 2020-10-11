#include "fuzzer.h"

class Mutator {
  private:
    Fuzzer* fuzzer = NULL;
  public:
    Mutator(Fuzzer* fuzzer);
    void mutate(void);
    void byte_flip(char* input, u32 len);
};
