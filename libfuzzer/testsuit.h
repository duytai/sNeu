#ifndef _HAVE_TESTSUIT_H
#define _HAVE_TESTSUIT_H

#include <libfuzzer/types.h>
#include <libfuzzer/fuzzer.h>
#include <vector>

using namespace std;

typedef struct {
  vector<char> mem;
  vector<u8> loss_bits;
  u8 min_loss;
} TestCase;

class TestSuite {
  vector<TestCase> testcases;
  Fuzzer *fuzzer = NULL;
  public:
    TestSuite(Fuzzer* fuzzer);
    void load_from_dir(char* in_dir);
};

#endif
