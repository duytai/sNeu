#ifndef _HAVE_TESTSUIT_H
#define _HAVE_TESTSUIT_H

#include <libfuzzer/types.h>
#include <libfuzzer/fuzzer.h>
#include <vector>

using namespace std;

typedef struct {
  vector<char> buffer;
  vector<u8> loss_bits;
  u8 min_loss = 255;
  u8 hnb;
} TestCase;

class TestSuite {
  vector<TestCase> testcases;
  Fuzzer *fuzzer = NULL;
  SNeuOptions opt;
  public:
    TestSuite(Fuzzer* fuzzer, SNeuOptions opt);
    void load_from_in_dir();
    void load_from_dir(char* dir);
};

#endif
