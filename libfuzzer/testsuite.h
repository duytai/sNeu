#ifndef _HAVE_TESTSUIT_H
#define _HAVE_TESTSUIT_H

#include <libfuzzer/types.h>
#include <libfuzzer/fuzzer.h>
#include <vector>

using namespace std;

class TestSuite {
  Fuzzer *fuzzer = NULL;
  SNeuOptions opt;

  public:
    TestSuite(Fuzzer* fuzzer, SNeuOptions opt);
    void mutate(void);
    void compute_branch_loss(vector<TestCase>&);
    vector<TestCase> load_from_dir(char* dir);
    vector<TestCase> smart_mutate(vector<TestCase>&);
};

#endif
