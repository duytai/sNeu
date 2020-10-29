#include <torch/torch.h>
#include <libfuzzer/testsuite.h>
#include <libfuzzer/fuzzer.h>
#include <libfuzzer/net.h>
#include <libfuzzer/util.h>
#include <algorithm>
#include <fstream>
#include <set>
#include <unistd.h>
#include <fcntl.h>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>

using namespace std;

TestSuite::TestSuite(Fuzzer* fuzzer, SNeuOptions opt) {
  this->fuzzer = fuzzer;
  this->fuzzer->load_opt(opt);
  this->fuzzer->stats.render_output = true;
  this->opt = opt;
}

vector<TestCase> TestSuite::load_from_dir(char* dir) {
  vector<TestCase> tcs;
  struct dirent **nl;

  s32 nl_cnt = scandir(dir, &nl, NULL, alphasort);
  if (nl_cnt < 0) PFATAL("Unable to open '%s'", dir);

  for (auto i = 0; i < nl_cnt; i += 1) {

    struct stat st;
    auto fn = string(dir) + "/" + string(nl[i]->d_name);

    free(nl[i]);
    if (lstat(fn.c_str(), &st) || access(fn.c_str(), R_OK)) PFATAL("Unable to access '%s'", fn.c_str());
    if (!S_ISREG(st.st_mode) || !st.st_size || strstr(fn.c_str(), "/README.txt")) continue;

    s32 fd = open(fn.c_str(), O_RDONLY);
    if (fd < 0) PFATAL("Unable to open '%s'", fn.c_str());

    char use_mem[st.st_size];
    if (read(fd, use_mem, st.st_size) != st.st_size) FATAL("Short read from '%s'", fn.c_str());

    vector<char> buffer(use_mem, use_mem + st.st_size);
    auto tmp = this->fuzzer->run_target(buffer, EXEC_TIMEOUT);
    this->save_if_interest(tmp, buffer, tcs);

    close(fd);
  }

  free(nl);

  return tcs;
}

void TestSuite::compute_branch_loss(vector<TestCase>& testcases) {
  auto& stats = this->fuzzer->stats;
  vector<u32> inst_branches;
  u32 i = 0;

  for (; i < MAP_SIZE; i += 1) {
    if (this->fuzzer->virgin_loss[i] != 0 && this->fuzzer->virgin_loss[i] != 255) {
      inst_branches.push_back(i);
    }
  }

  for (auto& t: testcases) {
    u8 loss = 255;
    for (auto br : inst_branches) {
      if (likely(loss & t.loss_bits[br])) {
        loss &= t.loss_bits[br];
      }
    }
    t.min_loss = loss;
  }

  stats.uncovered_branches = inst_branches.size();

}

/*
 * t.min_loss = min(loss_of_br_1, loss_of_br_2, ....)
 * t.min_loss != 255 means that the current testcase
 * modified losses of uncover branches
 * */

vector<TestCase> TestSuite::smart_mutate(vector<TestCase>& testcases) {

  set<u8> losses;
  vector<torch::Tensor> xs, ys, mixed_xs;
  vector<TestCase> tcs;
  auto& stats = this->fuzzer->stats;
  u32 max_len = 0,
      train_epoch = 100;

  /* Compute labels */
  this->compute_branch_loss(testcases);
  for (auto t : testcases) {
    max_len = max((u32) t.buffer.size(), max_len);
  }
  if (!max_len) return tcs; 

  auto options = torch::TensorOptions().dtype(torch::kUInt8);
  stats.input_size = max_len;
  for (auto t : testcases) {
    torch::Tensor x = torch::zeros(max_len);
    torch::Tensor y = torch::zeros(1);
    x.slice(0, 0, t.buffer.size()) = torch::from_blob(t.buffer.data(), t.buffer.size(), options);
    x = x / 255.0;
    y[0] = t.min_loss / 255.0;
    xs.push_back(x);
    ys.push_back(y);
    losses.insert(t.min_loss);
  }
  stats.total_inputs = ys.size();
  stats.uniq_loss = losses.size();

  auto net = std::make_shared<Net>(max_len);
  torch::optim::SGD optimizer(net->parameters(), /*lr=*/0.01);

  /* Training */
  stats.stage = "train";
  this->fuzzer->show_stats(1);
  for (u32 epoch = 0; epoch < train_epoch; epoch += 1) {
    optimizer.zero_grad();
    torch::Tensor prediction = net->forward(torch::stack(xs));
    torch::Tensor loss = torch::mse_loss(prediction, torch::stack(ys));
    loss.backward();
    optimizer.step();
  }

  /* Add and Delete */
  if (testcases.size() > 2) {
    for (u32 i = 0; i < testcases.size(); i += 1) {
      auto buff = testcases[i].buffer;
      u32 half_size = buff.size() / 2;
      if (half_size > 0) {
        u32 delete_from = half_size + random() % half_size;
        auto off = torch::zeros(max_len);
        for (u32 j = 0; j < delete_from; j += 1) {
          off[j] = (u8) buff[j] / 255.0;
        }
        mixed_xs.push_back(off);
      }
    }
  }

  /* Splicing */
  if (testcases.size() > 2) {
    for (u32 i = 0; i < testcases.size(); i += 1) {
      u32 j = random() % testcases.size();
      while (j == i) {
        j = random() % testcases.size();
      }
      auto buff_0 = testcases[i].buffer; 
      auto buff_1 = testcases[j].buffer;
      auto min_len = min(buff_0.size(), buff_1.size());
      s32 f_diff = -1, l_diff = -1;
      for (u32 loc = 0; loc < min_len; loc += 1) {
        if (buff_0[loc] != buff_1[loc]) {
          if (f_diff == -1) f_diff = loc;
          l_diff = loc;
        }
      }
      if (f_diff < 0 || l_diff < 2 || f_diff == l_diff) continue;
      u32 split_at = f_diff + random() % (l_diff - f_diff + 1);

      torch::Tensor off_0 = torch::zeros(max_len);
      torch::Tensor off_1 = torch::zeros(max_len);

      for (u32 i = 0; i < max_len; i += 1) {
        /* Create first offspring */
        if (i < buff_1.size()) {
          if (i < split_at) {
            off_0[i] = (u8) buff_0[i] / 255.0;
          } else {
            off_0[i] = (u8) buff_1[i] / 255.0;
          }
        }
        /* Create second offspring */
        if (i < buff_0.size()) {
          if (i < split_at) {
            off_1[i] = (u8) buff_1[i] / 255.0;
          } else {
            off_1[i] = (u8) buff_0[i] / 255.0;
          }
        }
      }

      mixed_xs.push_back(off_0);
      mixed_xs.push_back(off_1);
    }
  }
  /* Append mixed_xs to xs */
  xs.insert(xs.end(), mixed_xs.begin(), mixed_xs.end());

  /* Compute grads for input x and mutate topk */
  stats.stage = "mtopk";
  this->fuzzer->show_stats(1);
  for (auto x : xs) {
    x = x.clone();
    x.set_requires_grad(true);
    for (u32 epoch = 0; epoch < 100; epoch += 1) {
      optimizer.zero_grad();
      torch::Tensor prediction = net->forward(x);
      torch::Tensor loss = torch::mse_loss(prediction, torch::zeros(1));
      loss.backward();
      auto topk = x.grad().abs().topk(5);
      torch::Tensor values = get<0>(topk);
      torch::Tensor indices = get<1>(topk);

      /* Compute an update */
      torch::Tensor extra = torch::zeros(x.sizes()[0]);
      for (u32 i = 0; i < 5; i += 1) {
        auto idx = indices[i].item<int>();
        extra[idx] = values[i]; 
      }
      x.set_requires_grad(false);
      x.add_(extra);

      /* Got result, run target */
      auto temp = x.mul(255.0).to(torch::kUInt8);
      vector buffer((char*) temp.data_ptr(), (char*) temp.data_ptr() + temp.numel());
      auto tmp = this->fuzzer->run_target(buffer, EXEC_TIMEOUT);
      this->save_if_interest(tmp, buffer, tcs);

      /* Zero grad for next round */
      x.grad().zero_();
      x.set_requires_grad(true);
    }
  }

  /* Compute grads for input x and muate the whole input */
  stats.stage = "mall";
  this->fuzzer->show_stats(1);
  for (auto x : xs) {
    x = x.clone();
    x.set_requires_grad(true);
    for (u32 epoch = 0; epoch < 100; epoch += 1) {
      optimizer.zero_grad();
      torch::Tensor prediction = net->forward(x);
      torch::Tensor loss = torch::mse_loss(prediction, torch::zeros(1));
      loss.backward();
      x.set_requires_grad(false);
      x.add_(x.grad());

      /* Got result, run target */
      auto temp = x.mul(255.0).to(torch::kUInt8);
      vector buffer((char*) temp.data_ptr(), (char*) temp.data_ptr() + temp.numel());
      auto tmp = this->fuzzer->run_target(buffer, EXEC_TIMEOUT);
      this->save_if_interest(tmp, buffer, tcs);

      /* Zero grad for next round */
      x.grad().zero_();
      x.set_requires_grad(true);
    }
  }

  return tcs;
}

vector<TestCase> TestSuite::deterministic(vector<char> buffer, u32 cksum) {
  vector<TestCase> tcs;
  auto& stats = this->fuzzer->stats;

  /* FLIP1 */
  stats.stage = "flip1";
  for (u32 i = 0; i < buffer.size() << 3; i += 1) {
    FLIP_BIT(buffer.data(), i);
    auto tmp = this->fuzzer->run_target(buffer, EXEC_TIMEOUT);
    this->save_if_interest(tmp, buffer, tcs);
    FLIP_BIT(buffer.data(), i);
  }

  /* FLIP2 */
  stats.stage = "flip2";
  for (u32 i = 0; i < (buffer.size() << 3) - 1; i += 1) {
    FLIP_BIT(buffer.data(), i);
    FLIP_BIT(buffer.data(), i + 1);
    auto tmp = this->fuzzer->run_target(buffer, EXEC_TIMEOUT);
    this->save_if_interest(tmp, buffer, tcs);
    FLIP_BIT(buffer.data(), i);
    FLIP_BIT(buffer.data(), i + 1);
  }

  /* FLIP4 */
  stats.stage = "flip4";
  for (u32 i = 0; i < (buffer.size() << 3) - 3; i += 1) {
    FLIP_BIT(buffer.data(), i);
    FLIP_BIT(buffer.data(), i + 1);
    FLIP_BIT(buffer.data(), i + 2);
    FLIP_BIT(buffer.data(), i + 3);
    auto tmp = this->fuzzer->run_target(buffer, EXEC_TIMEOUT);
    this->save_if_interest(tmp, buffer, tcs);
    FLIP_BIT(buffer.data(), i);
    FLIP_BIT(buffer.data(), i + 1);
    FLIP_BIT(buffer.data(), i + 2);
    FLIP_BIT(buffer.data(), i + 3);
  }

  u8 eff_map[EFF_ALEN(buffer.size())];
  memset(eff_map, 0, EFF_ALEN(buffer.size()));

  u32 eff_cnt = 1;
  eff_map[0] = 1;

  if (EFF_APOS(buffer.size() - 1) != 0) {
    eff_map[EFF_APOS(buffer.size() - 1)] = 1;
    eff_cnt += 1;
  }

  /* FLIP8 */
  stats.stage = "flip8";
  for (u32 i = 0; i < buffer.size(); i += 1) {
    buffer.data()[i] ^= 0xFF;
    auto tmp = this->fuzzer->run_target(buffer, EXEC_TIMEOUT);
    this->save_if_interest(tmp, buffer, tcs);

    if (!eff_map[EFF_APOS(i)]) {
      u32 new_cksum = hash32(this->fuzzer->trace_bits, MAP_SIZE, HASH_CONST);
      if (new_cksum != cksum) {
        eff_map[EFF_APOS(i)] = 1;
        eff_cnt += 1;
      }
    }

    buffer.data()[i] ^= 0xFF;
  }

  if (eff_cnt != EFF_ALEN(buffer.size()) && eff_cnt * 100 / EFF_ALEN(buffer.size()) > EFF_MAX_PERC) {
    memset(eff_map, 1, EFF_ALEN(buffer.size()));
  }

  /* FLIP16 */
  if (buffer.size() >= 2) {
    stats.stage = "flip16";
    for (u32 i = 0; i < buffer.size() - 1; i += 1) {
      if (!eff_map[EFF_APOS(i)] && !eff_map[EFF_APOS(i + 1)]) continue;
      *(u16*)(buffer.data() + i) ^= 0xFFFF;
      auto tmp = this->fuzzer->run_target(buffer, EXEC_TIMEOUT);
      this->save_if_interest(tmp, buffer, tcs);
      *(u16*)(buffer.data() + i) ^= 0xFFFF;
    }
  }

  /* FLIP32 */
  if (buffer.size() >= 4) {
    stats.stage = "flip32";
    for (u32 i = 0; i < buffer.size() - 3; i += 1) {
      if (!eff_map[EFF_APOS(i)] && !eff_map[EFF_APOS(i + 1)] &&
          !eff_map[EFF_APOS(i + 2)] && !eff_map[EFF_APOS(i + 3)]) continue;
      *(u32*)(buffer.data() + i) ^= 0xFFFFFFFF;
      auto tmp = this->fuzzer->run_target(buffer, EXEC_TIMEOUT);
      this->save_if_interest(tmp, buffer, tcs);
      *(u32*)(buffer.data() + i) ^= 0xFFFFFFFF;
    }
  }

  /* ARITH8 */
  stats.stage = "arith8";
  for (u32 i = 0; i < buffer.size(); i += 1) {
    if (!eff_map[EFF_APOS(i)]) continue;

    u8 orig = buffer[i];
    for (u32 j = 0; j < ARITH_MAX; j += 1) {
      u8 r1 = orig ^ (orig + j);
      u8 r2 = orig ^ (orig - j);
      if (!could_be_bitflip(r1)) {
        buffer[i] = orig + j;
        auto tmp = this->fuzzer->run_target(buffer, EXEC_TIMEOUT);
        this->save_if_interest(tmp, buffer, tcs);
      }
      if (!could_be_bitflip(r2)) {
        buffer[i] = orig - j;
        auto tmp = this->fuzzer->run_target(buffer, EXEC_TIMEOUT);
        this->save_if_interest(tmp, buffer, tcs);
      }
      buffer[i] = orig;
    }
  }

  /* ARITH16 */
  if (buffer.size() >= 2) {
    stats.stage = "arith16";
    for (u32 i = 0; i < buffer.size() - 1; i += 1) {
      if (!eff_map[EFF_APOS(i)] && !eff_map[EFF_APOS(i + 1)]) continue;
      u16 orig = *(u16*)(buffer.data() + i);
      for (u32 j = 0; j < ARITH_MAX; j += 1) {
        u16 r1 = orig ^ (orig + j),
            r2 = orig ^ (orig - j),
            r3 = orig ^ SWAP16(SWAP16(orig) + j),
            r4 = orig ^ SWAP16(SWAP16(orig) - j);

        if ((orig & 0xff) + j > 0xff && !could_be_bitflip(r1)) {
          *(u16*)(buffer.data() + i) = orig + j;
          auto tmp = this->fuzzer->run_target(buffer, EXEC_TIMEOUT);
          this->save_if_interest(tmp, buffer, tcs);
        }
        if ((orig & 0xff) < j && !could_be_bitflip(r2)) {
          *(u16*)(buffer.data() + i) = orig - j;
          auto tmp = this->fuzzer->run_target(buffer, EXEC_TIMEOUT);
          this->save_if_interest(tmp, buffer, tcs);
        }
        if ((orig >> 8) + j > 0xff && !could_be_bitflip(r3)) {
          *(u16*)(buffer.data() + i) = SWAP16(SWAP16(orig) + j);
          auto tmp = this->fuzzer->run_target(buffer, EXEC_TIMEOUT);
          this->save_if_interest(tmp, buffer, tcs);
        }
        if ((orig >> 8) < j && !could_be_bitflip(r4)) {
          *(u16*)(buffer.data() + i) = SWAP16(SWAP16(orig) - j);
          auto tmp = this->fuzzer->run_target(buffer, EXEC_TIMEOUT);
          this->save_if_interest(tmp, buffer, tcs);
        }
        *(u16*)(buffer.data() + i) = orig;
      }
    }
  }

  /* ARITH32 */
  if (buffer.size() >= 4) {
    stats.stage = "arith32";
    for (u32 i = 0; i < buffer.size() - 3; i += 1) {
      if (!eff_map[EFF_APOS(i)] && !eff_map[EFF_APOS(i + 1)] &&
          !eff_map[EFF_APOS(i + 2)] && !eff_map[EFF_APOS(i + 3)]) continue;
      u32 orig = *(u32*)(buffer.data() + i);
      for (u32 j = 0; j < ARITH_MAX; j += 1) {
        u32 r1 = orig ^ (orig + j),
            r2 = orig ^ (orig - j),
            r3 = orig ^ SWAP32(SWAP32(orig) + j),
            r4 = orig ^ SWAP32(SWAP32(orig) - j);

        if ((orig & 0xffff) + j > 0xffff && !could_be_bitflip(r1)) {
          *(u32*)(buffer.data() + i) = orig + j;
          auto tmp = this->fuzzer->run_target(buffer, EXEC_TIMEOUT);
          this->save_if_interest(tmp, buffer, tcs);
        }
        if ((orig & 0xffff) < j && !could_be_bitflip(r2)) {
          *(u32*)(buffer.data() + i) = orig - j;
          auto tmp = this->fuzzer->run_target(buffer, EXEC_TIMEOUT);
          this->save_if_interest(tmp, buffer, tcs);
        }
        if ((SWAP32(orig) & 0xffff) + j > 0xffff && !could_be_bitflip(r3)) {
          *(u32*)(buffer.data() + i) = SWAP32(SWAP32(orig) + j);
          auto tmp = this->fuzzer->run_target(buffer, EXEC_TIMEOUT);
          this->save_if_interest(tmp, buffer, tcs);
        }
        if ((SWAP32(orig) & 0xffff) < j && !could_be_bitflip(r4)) {
          *(u32*)(buffer.data() + i) = SWAP32(SWAP32(orig) - j);
          auto tmp = this->fuzzer->run_target(buffer, EXEC_TIMEOUT);
          this->save_if_interest(tmp, buffer, tcs);
        }
        *(u32*)(buffer.data() + i) = orig;
      }
    }
  }

  /* INST8 */
  stats.stage = "inst8";
  for (u32 i = 0; i < buffer.size(); i += 1) {
    if (!eff_map[EFF_APOS(i)]) continue;
    u8 orig = buffer[i];
    for (u32 j = 0; j < sizeof(interesting_8); j += 1) {
      if (could_be_bitflip(orig ^ (u8)interesting_8[j]) ||
          could_be_arith(orig, (u8)interesting_8[j], 1)) {
        continue;
      }
      buffer[i] = interesting_8[j];
      auto tmp = this->fuzzer->run_target(buffer, EXEC_TIMEOUT);
      this->save_if_interest(tmp, buffer, tcs);
      buffer[i] = orig;
    }
  }

  /* INST16 */
  if (buffer.size() >= 2) {
    stats.stage = "inst16";
    for (u32 i = 0; i < buffer.size() - 1; i += 1) {
      if (!eff_map[EFF_APOS(i)] && !eff_map[EFF_APOS(i + 1)]) continue;
      u16 orig = *(u16*)(buffer.data() + i);

      for (u32 j = 0; j < sizeof(interesting_16) / 2; j += 1) {
        if (!could_be_bitflip(orig ^ (u16)interesting_16[j]) &&
            !could_be_arith(orig, (u16)interesting_16[j], 2) &&
            !could_be_interest(orig, (u16)interesting_16[j], 2, 0)) {
          *(u16*)(buffer.data() + i) = interesting_16[j];
          auto tmp = this->fuzzer->run_target(buffer, EXEC_TIMEOUT);
          this->save_if_interest(tmp, buffer, tcs);
        }
        if ((u16)interesting_16[j] != SWAP16(interesting_16[j]) &&
            !could_be_bitflip(orig ^ SWAP16(interesting_16[j])) &&
            !could_be_arith(orig, SWAP16(interesting_16[j]), 2) &&
            !could_be_interest(orig, SWAP16(interesting_16[j]), 2, 1)) {
          *(u16*)(buffer.data() + i) = SWAP16(interesting_16[j]);
          auto tmp = this->fuzzer->run_target(buffer, EXEC_TIMEOUT);
          this->save_if_interest(tmp, buffer, tcs);
        }
      }

      *(u16*)(buffer.data() + i) = orig;
    }
  }

  /* INST32 */
  if (buffer.size() >= 4) {
    stats.stage = "inst32";
    for (u32 i = 0; i < buffer.size() - 3; i += 1) {
      if (!eff_map[EFF_APOS(i)] && !eff_map[EFF_APOS(i + 1)] &&
          !eff_map[EFF_APOS(i + 2)] && !eff_map[EFF_APOS(i + 3)]) continue;

      u32 orig = *(u32*)(buffer.data() + i);

      for (u32 j = 0; j < sizeof(interesting_32) / 4; j += 1) {
        if (!could_be_bitflip(orig ^ (u32)interesting_32[j]) &&
            !could_be_arith(orig, interesting_32[j], 4) &&
            !could_be_interest(orig, interesting_32[j], 4, 0)) {
          *(u32*)(buffer.data()+ i) = interesting_32[j];
          auto tmp = this->fuzzer->run_target(buffer, EXEC_TIMEOUT);
          this->save_if_interest(tmp, buffer, tcs);
        } 

        if ((u32)interesting_32[j] != SWAP32(interesting_32[j]) &&
            !could_be_bitflip(orig ^ SWAP32(interesting_32[j])) &&
            !could_be_arith(orig, SWAP32(interesting_32[j]), 4) &&
            !could_be_interest(orig, SWAP32(interesting_32[j]), 4, 1)) {
          *(u32*)(buffer.data() + i) = SWAP32(interesting_32[j]);
          auto tmp = this->fuzzer->run_target(buffer, EXEC_TIMEOUT);
          this->save_if_interest(tmp, buffer, tcs);
        }
      }

      *(u32*)(buffer.data() + i) = orig;
    }
  }

  return tcs;
}

void TestSuite::save_if_interest(u8 result, vector<char>& mem, vector<TestCase>& tcs) {
  auto& stats = this->fuzzer->stats;
  u8 hnb = this->fuzzer->has_new_bits(this->fuzzer->virgin_bits);
  u8 crash_hnb = (result == FAULT_CRASH) ? this->fuzzer->has_new_bits(this->fuzzer->virgin_crash) : 0;
  string status = "success:" + to_string(hnb) + ",crash:" + to_string(crash_hnb);

  stats.total_crashes += (result == FAULT_CRASH);
  stats.uniq_crashes += crash_hnb > 0;

  if (hnb || crash_hnb) {
    tcs.push_back(this->fuzzer->tc);
    this->write_testcase(status, mem);
    if (hnb) {
      this->fuzzer->stats.queued_with_cov += (hnb == 2 ? 1 : 0);
      stats.total_ints += 1;
    }
  }
}

// TODO: mall, mtopk dont have src:000000
void TestSuite::write_testcase(string status, vector<char>& mem) {
  auto& stats = this->fuzzer->stats;
  string out_dir = string(this->opt.out_dir);

  char idx_str[7];
  char src_str[7];
  snprintf(idx_str, 7, "%06d", stats.test_idx);
  snprintf(src_str, 7, "%06d", stats.queue_idx);

  string fname = out_dir + "/id:" + idx_str + ",src:" + src_str + ",op:" + stats.stage + "," + status;
  unlink(fname.c_str());
  auto fd = open(fname.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
  if (fd < 0) PFATAL("Unable to create '%s'", fname.c_str());
  write(fd, mem.data(), mem.size());
  stats.test_idx += 1;
}

void TestSuite::mutate(void) {

  auto tcs = this->load_from_dir(this->opt.in_dir);
  u32 idx = tcs.size();
  auto& stats = this->fuzzer->stats;

  stats.test_idx = tcs.size();

  this->fuzzer->show_stats(1);
  while (1) {
    auto tmp = this->smart_mutate(tcs);
    tcs.insert(tcs.end(), tmp.begin(), tmp.end());

    while (idx < tcs.size()) {
      stats.queue_size = tcs.size();
      stats.queue_idx = idx + 1;
      this->fuzzer->show_stats(1);

      this->fuzzer->run_target(tcs[idx].buffer, EXEC_TIMEOUT);
      u32 cksum = hash32(this->fuzzer->trace_bits, MAP_SIZE, HASH_CONST);
      auto tmp = this->deterministic(tcs[idx].buffer, cksum);
      tcs.insert(tcs.end(), tmp.begin(), tmp.end());

      idx += 1;
    }

    stats.cycles += 1;
  }
}
