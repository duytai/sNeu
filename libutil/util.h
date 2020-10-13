#ifndef _HAVE_UTIL_H
#define _HAVE_UTIL_H

typedef struct {
  bool use_stdin = true;
  char* in_dir = NULL;
  char* out_file = ".cur_input";
  char** target_argv = NULL;
} SNeuOptions;

#endif
