#ifndef _HAVE_TESTSUIT_H
#define _HAVE_TESTSUIT_H

#include <libfuzzer/types.h>
#include <libfuzzer/fuzzer.h>
#include <vector>

using namespace std;

typedef enum {
  STAGE_FLIP1,
  STAGE_FLIP2,
  STAGE_FLIP4,
  STAGE_FLIP8,
  STAGE_FLIP16,
  STAGE_FLIP32,
  STAGE_ARITH8,
  STAGE_ARITH16,
  STAGE_ARITH32,
  STAGE_INTEREST8,
  STAGE_INTEREST16,
  STAGE_INTEREST32,
} Stage;

class TestSuite {
  Fuzzer *fuzzer = NULL;
  SNeuOptions opt;

  public:
    TestSuite(Fuzzer* fuzzer, SNeuOptions opt);
    void compute_branch_loss(vector<TestCase>&);
    vector<TestCase> load_from_dir(char* dir);
    vector<TestCase> smart_mutate(vector<TestCase>&);
    vector<TestCase> deterministic(vector<char>, u32);
    void save_if_interest(u8 ret,vector<char>&, vector<TestCase>& tcs);
    void write_testcase(string, vector<char>&);
    void mutate(void);
};

#endif
