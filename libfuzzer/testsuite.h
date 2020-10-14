#ifndef _HAVE_TESTSUIT_H
#define _HAVE_TESTSUIT_H

#include <libfuzzer/types.h>
#include <libfuzzer/fuzzer.h>
#include <vector>

using namespace std;

class TestSuite {
  vector<TestCase> testcases;
  Fuzzer *fuzzer = NULL;
  SNeuOptions opt;
  public:
    TestSuite(Fuzzer* fuzzer, SNeuOptions opt);
    void load_from_in_dir(void);
    void load_from_dir(char* dir);
    void compute_branch_loss(void);
    void mutate(void);
    vector<TestCase> smart_mutate(void);
};

#endif
