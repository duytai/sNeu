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
      count_class_lookup8[256];

    u16 count_class_lookup16[65536];
    SNeuOptions opt;

  public:
    u64 total_execs;
    u64 exec_ms;
    u8 hnb,
       virgin_bits[MAP_SIZE],
       virgin_loss[MAP_SIZE],
       * loss_bits,
       * trace_bits;

    Fuzzer();
    ~Fuzzer();
    void load_opt(SNeuOptions opt);
    void init_count_class16(void);
    void init_count_class8(void);
    void classify_counts(void);
    void handle_timeout(void);
    void setup_fds(void);
    void remove_shm(void);
    void setup_shm(void);
    void init_forkserver(void);
    void update_loss(void);
    void handle_stop_sig(void);
    u8 has_new_bits(void);
    u8 run_target(vector<char>& mem, u32 exec_tmout);
};

#endif
