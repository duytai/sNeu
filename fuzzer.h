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

  public:
    void setup_fds(void);
    void setup_shm(void);
    void setup_signal_handlers(void);
    void init_forkserver(void);
};

#endif
