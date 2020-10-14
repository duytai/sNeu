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
  bool executed = false;
} TestCase;

class TestSuite {
  vector<TestCase> testcases;
  Fuzzer *fuzzer = NULL;
  SNeuOptions opt;
  public:
    TestSuite(Fuzzer* fuzzer, SNeuOptions opt);
    void load_from_in_dir(void);
    void load_from_dir(char* dir);
    void exec_remaining(void);
    void compute_branch_loss(void);
    void smart_mutate(void);
};

#endif
