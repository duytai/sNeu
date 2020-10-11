#include "fuzzer.h"

class Mutator {
  private:
    Fuzzer* fuzzer = NULL;

  public:
    Mutator(Fuzzer* fuzzer);
    void mutate(void);
    void byte_flip(vector<char>& mem);
};
