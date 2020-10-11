#ifndef _HAVE_FUZZER_H
#define _HAVE_FUZZER_H

#include "config.h"
#include "types.h"
#include "debug.h"
#include "hash.h"
#include "fuzzer.h"

#include <vector>
#include <string>

using namespace std;

typedef struct {
  bool use_stdin = true;
  char* in_dir = NULL;
  char* out_file = ".cur_input";
  char** target_argv = NULL;
} FuzzerOpt;

typedef struct {
  vector<char> mem;
  vector<u8> loss_bits;
  u8 min_loss;
} FuzzerPair;

class Fuzzer {
  private:
    s32 forksrv_pid,
      fsrv_st_fd,
      fsrv_ctl_fd,
      shm_id,
      child_pid = -1,
      out_fd,
      dev_null_fd = -1;

    u8 child_timed_out,
      virgin_bits[MAP_SIZE],
      virgin_loss[MAP_SIZE],
      * loss_bits,
      * trace_bits,
      count_class_lookup8[256] = {
        [0]           = 0,
        [1]           = 1,
        [2]           = 2,
        [3]           = 4,
        [4 ... 7]     = 8,
        [8 ... 15]    = 16,
        [16 ... 31]   = 32,
        [32 ... 127]  = 64,
        [128 ... 255] = 128
      };

    u16 count_class_lookup16[65536];
    FuzzerOpt opt;

  public:
    u64 total_execs;
    u64 exec_ms;
    u8 hnb;
    vector<FuzzerPair> pairs;

    Fuzzer();
    ~Fuzzer();
    void load_opt(FuzzerOpt opt);
    void init_count_class16(void);
    void classify_counts(void);
    void handle_timeout(void);
    void setup_fds(void);
    void remove_shm(void);
    void setup_shm(void);
    void init_forkserver(void);
    void update_loss(void);
    void update_inst_branches(void);
    u8 has_new_bits(void);
    u8 run_target(vector<char>& mem, u32 exec_tmout);
};

#endif
