#ifndef _HAVE_FUZZER_H
#define _HAVE_FUZZER_H

#include <libfuzzer/config.h>
#include <libfuzzer/types.h>
#include <libfuzzer/debug.h>
#include <libfuzzer/hash.h>
#include <libfuzzer/fuzzer.h>
#include <libutil/util.h>

#include <vector>
#include <string>

using namespace std;

typedef struct {
  vector<char> buffer;
  vector<u8> loss_bits;
  u8 min_loss = 255;
  u8 hnb;
} TestCase;

typedef struct {
  u64 total_execs;
  u64 total_ints;
  u64 total_time;
  u64 start_time;
  u32 queue_size;
  u32 queue_idx;
  bool render_output;
  string stage = "init";
} FuzzStats;

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
    SNeuOptions opt;

  public:
    FuzzStats stats;
    TestCase tc;
    u8 virgin_bits[MAP_SIZE],
       virgin_loss[MAP_SIZE],
       * loss_bits,
       * trace_bits;

    Fuzzer();
    ~Fuzzer();
    void load_opt(SNeuOptions opt);
    void init_count_class16(void);
    void classify_counts(void);
    void handle_timeout(void);
    void setup_fds(void);
    void remove_shm(void);
    void setup_shm(void);
    void init_forkserver(void);
    void update_loss(void);
    void handle_stop_sig(void);
    void show_stats(u8 force); 
    u8 has_new_bits(void);
    u8 run_target(vector<char>& mem, u32 exec_tmout);
};

#endif
