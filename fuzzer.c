#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/wait.h>
#include <dirent.h>
#include <string.h>
#include <stdint.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <sys/inotify.h>

#define MAP_SIZE 1 << 16
#define FORKSRV_FD 198
#define FORK_WAIT_MULT 10
#define EXEC_TIMEOUT 1000
#define SHM_ENV_VAR "__AFL_SHM_ID"
#define SA struct sockaddr
#define PORT 1234
#define IN_BUF_SIZE 1024

#define MEM_BARRIER() \
  __asm__ volatile("" ::: "memory")
#define likely(_x)   __builtin_expect(!!(_x), 1)
#define unlikely(_x)  __builtin_expect(!!(_x), 0)
#define BUF_LEN (10 * (sizeof(struct inotify_event) + 1024 + 1))

enum {
  /* 00 */ FAULT_NONE,
  /* 01 */ FAULT_TMOUT,
  /* 02 */ FAULT_CRASH,
  /* 03 */ FAULT_ERROR,
  /* 04 */ FAULT_NOINST,
  /* 05 */ FAULT_NOBITS
};

static int forksrv_pid, fsrv_st_fd, fsrv_ctl_fd, shm_id, child_pid = -1; 
static int out_fd, dev_null_fd, child_timed_out, total_execs;
static uint8_t* trace_bits;
static uint8_t virgin_bits[MAP_SIZE];

static const uint8_t count_class_lookup8[256] = {
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

static uint16_t count_class_lookup16[65536];

static void init_count_class16(void) {
  uint32_t b1, b2;
  for (b1 = 0; b1 < 256; b1++)
    for (b2 = 0; b2 < 256; b2++)
      count_class_lookup16[(b1 << 8) + b2] =
        (count_class_lookup8[b1] << 8) |
        count_class_lookup8[b2];
}

static void classify_counts(uint64_t* mem) {
  uint32_t i = MAP_SIZE >> 3;
  while (i--) {
    /* Optimize for sparse bitmaps. */
    if (unlikely(*mem)) {
      uint16_t * mem16 = (uint16_t*)mem;
      mem16[0] = count_class_lookup16[mem16[0]];
      mem16[1] = count_class_lookup16[mem16[1]];
      mem16[2] = count_class_lookup16[mem16[2]];
      mem16[3] = count_class_lookup16[mem16[3]];
    }
    mem++;
  }
}

static void fatal(char* msg) {
  fprintf(stderr, "%s\n", msg);
  exit(EXIT_FAILURE);
}

static void remove_shm(void) {
  shmctl(shm_id, IPC_RMID, NULL);
}

static void handle_timeout(int sig) {
  if (child_pid > 0) {
    child_timed_out = 1;
    kill(child_pid, SIGKILL);
  } else if (child_pid == -1 && forksrv_pid > 0) {
    child_timed_out = 1;
    kill(forksrv_pid, SIGKILL);
  }
}

static void setup_shm(void) {
  char shm_str[10];
  shm_id = shmget(IPC_PRIVATE, MAP_SIZE, IPC_CREAT | IPC_EXCL | 0600);
  if (shm_id < 0) fatal("shmget() failed");
  atexit(remove_shm);
  sprintf(shm_str, "%d", shm_id);
  setenv(SHM_ENV_VAR, shm_str, 1);
  memset(virgin_bits, 255, MAP_SIZE);
  trace_bits = shmat(shm_id, NULL, 0);
  if (trace_bits == (void *)-1) fatal("shmat() failed");
}

static void setup_signal_handlers() {
  struct sigaction sa;

  sa.sa_handler   = NULL;
  sa.sa_flags     = SA_RESTART;
  sa.sa_sigaction = NULL;

  sigemptyset(&sa.sa_mask);

  sa.sa_handler = handle_timeout;
  sigaction(SIGALRM, &sa, NULL);
}

static void init_forkserver(char* target_path) {
  static struct itimerval it;
  int st_pipe[2], ctl_pipe[2];
  int status, rlen;

  printf("[+] Spinning up the fork server...\n");
  if (pipe(st_pipe) || pipe(ctl_pipe)) fatal("pipe() failed");

  forksrv_pid = fork();
  if (forksrv_pid < 0) fatal("fork() failed");

  if (!forksrv_pid) {
    setsid();

    dup2(dev_null_fd, 1);
    dup2(dev_null_fd, 2);

    dup2(out_fd, 0);
    close(out_fd);

    if (dup2(ctl_pipe[0], FORKSRV_FD) < 0) fatal("dup2() failed");
    if (dup2(st_pipe[1], FORKSRV_FD + 1) < 0) fatal("dup2() failed");

    close(ctl_pipe[0]);
    close(ctl_pipe[1]);
    close(st_pipe[0]);
    close(st_pipe[1]);

    char** argv = { NULL };
    execv(target_path, argv);
    exit(EXIT_SUCCESS);
  }

  close(ctl_pipe[0]);
  close(st_pipe[1]);

  fsrv_ctl_fd = ctl_pipe[1];
  fsrv_st_fd = st_pipe[0]; 

  it.it_value.tv_sec = ((EXEC_TIMEOUT * FORK_WAIT_MULT) / 1000);
  it.it_value.tv_usec = ((EXEC_TIMEOUT * FORK_WAIT_MULT) % 1000) * 1000;

  setitimer(ITIMER_REAL, &it, NULL);

  rlen = read(fsrv_st_fd, &status, 4);

  it.it_value.tv_sec = 0;
  it.it_value.tv_usec = 0;

  setitimer(ITIMER_REAL, &it, NULL);

  if (rlen == 4) {
    printf("[+] All right - fork server is up.\n");
    return;
  }

  if (waitpid(forksrv_pid, &status, 0) <= 0) fatal("waitpid() failed");
  fatal("[+] Fork server handshake failed");
}

static uint8_t has_new_bits(uint8_t* virgin_map) {
  uint64_t* current = (uint64_t*)trace_bits;
  uint64_t* virgin  = (uint64_t*)virgin_map;
  int i = (MAP_SIZE >> 3);
  uint8_t ret = 0;

  while (i--) {
    if (unlikely(*current) && unlikely(*current & *virgin)) {
      if (likely(ret < 2)) {
        uint8_t* cur = (uint8_t*)current;
        uint8_t* vir = (uint8_t*)virgin;

        /* Looks like we have not found any new bytes yet; see if any non-zero
           bytes in current[] are pristine in virgin[]. */
        if ((cur[0] && vir[0] == 0xff) || (cur[1] && vir[1] == 0xff) ||
            (cur[2] && vir[2] == 0xff) || (cur[3] && vir[3] == 0xff) ||
            (cur[4] && vir[4] == 0xff) || (cur[5] && vir[5] == 0xff) ||
            (cur[6] && vir[6] == 0xff) || (cur[7] && vir[7] == 0xff)) ret = 2;
        else ret = 1;
      }
      *virgin &= ~*current;
    }
    current++;
    virgin++;
  }
  return ret;
}

static int run_target(char* used_mem, int len) {
  static struct itimerval it;
  static int prev_timed_out = 0;
  int status, res, kill_signal;

  lseek(out_fd, 0, SEEK_SET);
  if (write(out_fd, used_mem, len) != len) fatal("write() failed");
  if (ftruncate(out_fd, len)) fatal("ftruncate() failed");
  lseek(out_fd, 0, SEEK_SET);

  memset(trace_bits, 0, MAP_SIZE);
  MEM_BARRIER();
  child_timed_out = 0;

  if ((res = write(fsrv_ctl_fd, &prev_timed_out, 4)) != 4) {
    fatal("Unable to request new process from fork server (OOM?)");
  } 
  if ((res = read(fsrv_st_fd, &child_pid, 4)) != 4) {
    fatal("Unable to request new process from fork server (OOM?)");
  } 
  if (child_pid <= 0) fatal("Fork server is misbehaving (OOM?)");

  it.it_value.tv_sec = (EXEC_TIMEOUT / 1000);
  it.it_value.tv_usec = (EXEC_TIMEOUT % 1000) * 1000;
  setitimer(ITIMER_REAL, &it, NULL);

  if ((res = read(fsrv_st_fd, &status, 4)) != 4) { // get return status
    fatal("Unable to communicate with fork server (OOM?)");
  }

  if (!WIFSTOPPED(status)) child_pid = 0;

  getitimer(ITIMER_REAL, &it);
  it.it_value.tv_sec = 0;
  it.it_value.tv_usec = 0;
  setitimer(ITIMER_REAL, &it, NULL);

  prev_timed_out = child_timed_out;
  total_execs ++;
  MEM_BARRIER();
  classify_counts((uint64_t*) trace_bits);

  if (WIFSIGNALED(status)) {
    kill_signal = WTERMSIG(status);
    if (child_timed_out && kill_signal == SIGKILL) return FAULT_TMOUT;
    return FAULT_CRASH;
  }

  return FAULT_NONE;
}

static void server_up(void) {
  struct sockaddr_in server, client;
  int rlen, server_fd, client_fd, flag = 1;
  char in_buf[IN_BUF_SIZE];
  char out_buf[100];

  server_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (server_fd < 0) fatal("socket() failed");

  server.sin_family = AF_INET;
	server.sin_addr.s_addr = INADDR_ANY;
	server.sin_port = htons(PORT);

  setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &flag, sizeof(flag));

  if (bind(server_fd,(SA*)&server, sizeof(server)) < 0) fatal("bind() failed");
  if (listen(server_fd, 3) < 0) fatal("listen() failed");
  printf("[+] Waiting for incoming connections\n");

  while (1) {
    socklen_t tmp = sizeof(client);
    client_fd = accept(server_fd, (SA*)&client, &tmp);
    if (client_fd < 0) fatal("accept() failed");
    printf("[+] Connection is accepted\n");

    while ((rlen = recv(client_fd, in_buf, sizeof(in_buf), 0)) > 0) {
      int ret = run_target(in_buf, rlen);
      int hnb = has_new_bits(virgin_bits);
      sprintf(out_buf, "%d:%d\n", ret, hnb);
      send(client_fd, out_buf, strlen(out_buf), 0);
    }

    close(client_fd);
    printf("[+] Waiting for incoming connections\n");
  }

  close(server_fd);
}

static void run_one(char* fn) {
  struct stat st;

  if (lstat(fn, &st) || access(fn, R_OK)) return;
  if (!S_ISREG(st.st_mode) || !st.st_size) return;

  char used_mem[st.st_size];
  int fd = open(fn, O_RDONLY);

  if (fd < 0) fatal("open() failed");
  if (read(fd, used_mem, st.st_size) != st.st_size)
    fatal("read() failed");
  close(fd);

  run_target(used_mem, st.st_size);
  printf("[+] Sync %s\n", fn);
}

static void sync_bitmap(char* queue_dir) {
  int inotify_fd, wd, cur_cnt, new_cnt;
  char buf[BUF_LEN], *p;
  ssize_t num_read;

  struct inotify_event *event;
  struct dirent **nl;

  memset(virgin_bits, 255, MAP_SIZE);
  cur_cnt = scandir(queue_dir, &nl, NULL, alphasort);

  if (getenv("SKIP_WATCH")) {
    printf("[+] Start syncing with master\n");
    for (int i = 0; i < cur_cnt; i++) {
      char fn[255];
      sprintf(fn, "%s/%s", queue_dir, nl[i]->d_name);
      run_one(fn);
      free(nl[i]);
    }
    free(nl);
    return;
  }

  while (1) {
    new_cnt = scandir(queue_dir, &nl, NULL, alphasort);
    if (cur_cnt == -1 && new_cnt != -1) {
      cur_cnt = new_cnt;
      continue;
    }
    if (cur_cnt != -1 && new_cnt != -1) {
      if (cur_cnt != new_cnt) {
        printf("[+] Start syncing with master\n");
        for (int i = 0; i < new_cnt; i++) {
          char fn[255];
          sprintf(fn, "%s/%s", queue_dir, nl[i]->d_name);
          run_one(fn);
        }
        break;
      }
    }
    sleep(1);
  }

  free(nl);

  inotify_fd = inotify_init();
  if (inotify_fd < 0) fatal("inotify_init() failed");
  wd = inotify_add_watch(inotify_fd, queue_dir, IN_ALL_EVENTS);
  if (wd < 0) fatal("watch() failed");

  for (;;) {                                  /* Read events forever */
    num_read = read(inotify_fd, buf, BUF_LEN);
    if (num_read < 0) fatal("read() failed");
    for (p = buf; p < buf + num_read; ) {
      event = (struct inotify_event *) p;
      if (event->mask & IN_IGNORED) {
        sync_bitmap(queue_dir);
        return;
      } 
      if (event->mask & IN_CLOSE_WRITE) {
        char fn[255];
        sprintf(fn, "%s/%s", queue_dir, event->name);
        run_one(fn);
      }
      p += sizeof(struct inotify_event) + event->len;
    }
  }
}

static void usage(char* name) {
  printf("Usage: %s /path/to/fuzzed_app /path/to/queue \n", name);
  exit(EXIT_SUCCESS);
}

int main(int argv, char** argc) {

  if (argv < 3) usage(argc[0]);
  char* target_path = argc[1];
  char* target_queue = argc[2];

  unlink(".cur_input");
  out_fd = open(".cur_input", O_RDWR | O_CREAT | O_EXCL, 0600);
  if (out_fd < 0) fatal("Unable to create '.cur_input'");

  dev_null_fd = open("/dev/null", O_RDWR);
  if (dev_null_fd < 0) fatal("Unable to open /dev/null");

  setup_shm();
  init_count_class16();
  setup_signal_handlers();
  init_forkserver(target_path);
  
  int sync_pid = fork();
  if (!sync_pid) {
    sync_bitmap(target_queue);
    return 0;
  }
  server_up();
  return 0;
}
