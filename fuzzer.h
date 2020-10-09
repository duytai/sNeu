#ifndef _HAVE_FUZZER_H
#define _HAVE_FUZZER_H

#include "config.h"
#include "types.h"
#include "debug.h"
#include "hash.h"
#include "fuzzer.h"

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
      * trace_bits;
      
    u64 total_execs;

    bool use_stdin = true;
    char * out_file = ".cur_input",
         * in_dir,
         ** target_argv;

  public:
    ~Fuzzer();
    void handle_timeout(void);
    void setup_fds(void);
    void remove_shm(void);
    void setup_shm(void);
    void parse_arguments(int argc, char** argv);
    void init_forkserver(void);
    void write_to_testcase(char* mem, u32 len);
    u8 run_target(u32 exec_tmout);
};

#endif
