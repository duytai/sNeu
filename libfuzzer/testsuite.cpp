#include <torch/torch.h>
#include <libfuzzer/testsuite.h>
#include <libfuzzer/fuzzer.h>
#include <libfuzzer/net.h>
#include <libfuzzer/util.h>
#include <filesystem>
#include <algorithm>
#include <fstream>
#include <set>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>

using namespace std;
using namespace std::filesystem;

TestSuite::TestSuite(Fuzzer* fuzzer, SNeuOptions opt) {
  this->fuzzer = fuzzer;
  this->fuzzer->load_opt(opt);
  this->fuzzer->stats.render_output = true;
  this->opt = opt;
}

vector<TestCase> TestSuite::load_from_dir(char* dir) {
  vector<TestCase> tcs;
  vector<directory_entry> files((directory_iterator(dir)), directory_iterator());
  sort(files.begin(), files.end());

  for (auto &file: files) {
    if (file.is_regular_file() && file.file_size() > 0) {
      ifstream st(file.path(), ios::binary);
      vector<char> buffer((istreambuf_iterator<char>(st)), istreambuf_iterator<char>());
      this->fuzzer->run_target(buffer, EXEC_TIMEOUT);
      tcs.push_back(this->fuzzer->tc);
    }
  }

  return tcs;
}

void TestSuite::compute_branch_loss(vector<TestCase>& testcases) {
  auto& stats = this->fuzzer->stats;
  vector<u32> inst_branches;
  u32 i = 0;

  for (; i < MAP_SIZE; i += 1) {
    if (this->fuzzer->virgin_loss[i] != 0 && this->fuzzer->virgin_loss[i] != 255) {
      set<u8> losses;
      for (auto t: testcases) {
        losses.insert(t.loss_bits[i]);
      }
      /*
       * Value in branch $i changed twice
       * TODO: increase to reduce the number of uncover branches in dataset
       * */
      if (losses.size() >= 2) inst_branches.push_back(i);
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
  vector<torch::Tensor> xs, ys;
  vector<TestCase> tcs;
  auto& stats = this->fuzzer->stats;
  u32 max_len = 0,
      train_epoch = 1000;

  /* Splicing */
  u32 round = 0;
  if (testcases.size() > 2) {
    stats.stage = "splicing";
    while (++ round < 255) {
      u32 idx_0 = random() % testcases.size();
      u32 idx_1 = random() % testcases.size();
      while (idx_0 == idx_1) {
        idx_1 = random() % testcases.size();
      }
      auto& buff_0 = testcases[idx_0].buffer;
      auto& buff_1 = testcases[idx_1].buffer;

      u32 min_len = min(buff_0.size(), buff_1.size());
      s32 f_loc = -1, l_loc = -1;
      for (u32 i = 0; i < min_len; i += 1) {
        if (buff_0[i] != buff_1[i]) {
          if (f_loc == -1) f_loc = i;
          l_loc = i;
        }
      }
      if (f_loc < 2 || l_loc < 2 || f_loc == l_loc) continue;

      /* Choose place to split */
      u32 split_at = f_loc + random() % (l_loc - f_loc + 1);

      /* head + tail of child 0 */
      vector<char> child_0(buff_1.size());
      memcpy(child_0.data(), buff_0.data(), split_at); 
      memcpy(child_0.data() + split_at, buff_1.data() + split_at, buff_1.size() - split_at); 

      /* head + tail of child 1 */
      vector<char> child_1(buff_0.size());
      memcpy(child_1.data(), buff_1.data(), split_at);
      memcpy(child_1.data() + split_at, buff_0.data() + split_at, buff_0.size() - split_at);

      /* run target and save if interest */
      for (auto buffer : {child_0, child_1}) {
        this->fuzzer->run_target(buffer, EXEC_TIMEOUT);
        if (this->fuzzer->tc.hnb) {
          tcs.push_back(this->fuzzer->tc);
          this->write_testcase(buffer);
        }
      }
    }
  }

  /* Compute labels */
  this->compute_branch_loss(testcases);
  for (auto t : testcases) {
    if (t.min_loss != 255) {
      max_len = max((u32) t.buffer.size(), max_len);
    }
  }
  stats.input_size = max_len;

  for (auto t : testcases) {
    if (t.min_loss != 255) {
      torch::Tensor x = torch::zeros(max_len);
      torch::Tensor y = torch::zeros(1);
      for (size_t i = 0; i < t.buffer.size(); i += 1) {
        x[i] = (u8) t.buffer[i] / 255.0;
      }
      y[0] = t.min_loss / 64.0;
      xs.push_back(x);
      ys.push_back(y);
      losses.insert(t.min_loss);
    }
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

  /* Compute grads for input x and mutate topk */
  stats.stage = "mtopk";
  this->fuzzer->show_stats(1);
  for (auto& x : xs) {
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
      this->fuzzer->run_target(buffer, EXEC_TIMEOUT);
      if (this->fuzzer->tc.hnb) {
        tcs.push_back(this->fuzzer->tc);
        this->write_testcase(buffer);
      }

      /* Zero grad for next round */
      x.grad().zero_();
      x.set_requires_grad(true);
    }
  }

  /* Compute grads for input x and muate the whole input */
  stats.stage = "mall";
  this->fuzzer->show_stats(1);
  for (auto& x : xs) {
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
      this->fuzzer->run_target(buffer, EXEC_TIMEOUT);
      if (this->fuzzer->tc.hnb) {
        tcs.push_back(this->fuzzer->tc);
        this->write_testcase(buffer);
      }

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
    this->fuzzer->run_target(buffer, EXEC_TIMEOUT);
    if (this->fuzzer->tc.hnb) {
      tcs.push_back(this->fuzzer->tc);
      this->write_testcase(buffer);
    }
    FLIP_BIT(buffer.data(), i);
  }

  /* FLIP2 */
  stats.stage = "flip2";
  for (u32 i = 0; i < (buffer.size() << 3) - 1; i += 1) {
    FLIP_BIT(buffer.data(), i);
    FLIP_BIT(buffer.data(), i + 1);
    this->fuzzer->run_target(buffer, EXEC_TIMEOUT);
    if (this->fuzzer->tc.hnb) {
      tcs.push_back(this->fuzzer->tc);
      this->write_testcase(buffer);
    }
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
    this->fuzzer->run_target(buffer, EXEC_TIMEOUT);
    if (this->fuzzer->tc.hnb) {
      tcs.push_back(this->fuzzer->tc);
      this->write_testcase(buffer);
    }
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
    this->fuzzer->run_target(buffer, EXEC_TIMEOUT);
    if (this->fuzzer->tc.hnb) {
      tcs.push_back(this->fuzzer->tc);
      this->write_testcase(buffer);
    }

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
      this->fuzzer->run_target(buffer, EXEC_TIMEOUT);
      if (this->fuzzer->tc.hnb) {
        tcs.push_back(this->fuzzer->tc);
        this->write_testcase(buffer);
      }
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
      this->fuzzer->run_target(buffer, EXEC_TIMEOUT);
      if (this->fuzzer->tc.hnb) {
        tcs.push_back(this->fuzzer->tc);
        this->write_testcase(buffer);
      }
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
        this->fuzzer->run_target(buffer, EXEC_TIMEOUT);
        if (this->fuzzer->tc.hnb) {
          tcs.push_back(this->fuzzer->tc);
          this->write_testcase(buffer);
        }
      }
      if (!could_be_bitflip(r2)) {
        buffer[i] = orig - j;
        this->fuzzer->run_target(buffer, EXEC_TIMEOUT);
        if (this->fuzzer->tc.hnb) {
          tcs.push_back(this->fuzzer->tc);
          this->write_testcase(buffer);
        }
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
          this->fuzzer->run_target(buffer, EXEC_TIMEOUT);
          if (this->fuzzer->tc.hnb) {
            tcs.push_back(this->fuzzer->tc);
            this->write_testcase(buffer);
          }
        }
        if ((orig & 0xff) < j && !could_be_bitflip(r2)) {
          *(u16*)(buffer.data() + i) = orig - j;
          this->fuzzer->run_target(buffer, EXEC_TIMEOUT);
          if (this->fuzzer->tc.hnb) {
            tcs.push_back(this->fuzzer->tc);
            this->write_testcase(buffer);
          }
        }
        if ((orig >> 8) + j > 0xff && !could_be_bitflip(r3)) {
          *(u16*)(buffer.data() + i) = SWAP16(SWAP16(orig) + j);
          this->fuzzer->run_target(buffer, EXEC_TIMEOUT);
          if (this->fuzzer->tc.hnb) {
            tcs.push_back(this->fuzzer->tc);
            this->write_testcase(buffer);
          }
        }
        if ((orig >> 8) < j && !could_be_bitflip(r4)) {
          *(u16*)(buffer.data() + i) = SWAP16(SWAP16(orig) - j);
          this->fuzzer->run_target(buffer, EXEC_TIMEOUT);
          if (this->fuzzer->tc.hnb) {
            tcs.push_back(this->fuzzer->tc);
            this->write_testcase(buffer);
          }
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
          this->fuzzer->run_target(buffer, EXEC_TIMEOUT);
          if (this->fuzzer->tc.hnb) {
            tcs.push_back(this->fuzzer->tc);
            this->write_testcase(buffer);
          }
        }
        if ((orig & 0xffff) < j && !could_be_bitflip(r2)) {
          *(u32*)(buffer.data() + i) = orig - j;
          this->fuzzer->run_target(buffer, EXEC_TIMEOUT);
          if (this->fuzzer->tc.hnb) {
            tcs.push_back(this->fuzzer->tc);
            this->write_testcase(buffer);
          }
        }
        if ((SWAP32(orig) & 0xffff) + j > 0xffff && !could_be_bitflip(r3)) {
          *(u32*)(buffer.data() + i) = SWAP32(SWAP32(orig) + j);
          this->fuzzer->run_target(buffer, EXEC_TIMEOUT);
          if (this->fuzzer->tc.hnb) {
            tcs.push_back(this->fuzzer->tc);
            this->write_testcase(buffer);
          }
        }
        if ((SWAP32(orig) & 0xffff) < j && !could_be_bitflip(r4)) {
          *(u32*)(buffer.data() + i) = SWAP32(SWAP32(orig) - j);
          this->fuzzer->run_target(buffer, EXEC_TIMEOUT);
          if (this->fuzzer->tc.hnb) {
            tcs.push_back(this->fuzzer->tc);
            this->write_testcase(buffer);
          }
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
      this->fuzzer->run_target(buffer, EXEC_TIMEOUT);
      if (this->fuzzer->tc.hnb) {
        tcs.push_back(this->fuzzer->tc);
        this->write_testcase(buffer);
      }
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
          this->fuzzer->run_target(buffer, EXEC_TIMEOUT);
          if (this->fuzzer->tc.hnb) {
            tcs.push_back(this->fuzzer->tc);
            this->write_testcase(buffer);
          }
        }
        if ((u16)interesting_16[j] != SWAP16(interesting_16[j]) &&
            !could_be_bitflip(orig ^ SWAP16(interesting_16[j])) &&
            !could_be_arith(orig, SWAP16(interesting_16[j]), 2) &&
            !could_be_interest(orig, SWAP16(interesting_16[j]), 2, 1)) {
          *(u16*)(buffer.data() + i) = SWAP16(interesting_16[j]);
          this->fuzzer->run_target(buffer, EXEC_TIMEOUT);
          if (this->fuzzer->tc.hnb) {
            tcs.push_back(this->fuzzer->tc);
            this->write_testcase(buffer);
          }
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
          this->fuzzer->run_target(buffer, EXEC_TIMEOUT);
          if (this->fuzzer->tc.hnb) {
            tcs.push_back(this->fuzzer->tc);
            this->write_testcase(buffer);
          }
        } 

        if ((u32)interesting_32[j] != SWAP32(interesting_32[j]) &&
            !could_be_bitflip(orig ^ SWAP32(interesting_32[j])) &&
            !could_be_arith(orig, SWAP32(interesting_32[j]), 4) &&
            !could_be_interest(orig, SWAP32(interesting_32[j]), 4, 1)) {
          *(u32*)(buffer.data() + i) = SWAP32(interesting_32[j]);
          this->fuzzer->run_target(buffer, EXEC_TIMEOUT);
          if (this->fuzzer->tc.hnb) {
            tcs.push_back(this->fuzzer->tc);
            this->write_testcase(buffer);
          }
        }
      }

      *(u32*)(buffer.data() + i) = orig;
    }
  }

  return tcs;
}

void TestSuite::write_testcase(vector<char>& mem) {
  auto& stats = this->fuzzer->stats;
  string out_dir = string(this->opt.out_dir);

  char idx_str[7];
  char src_str[7];
  snprintf(idx_str, 7, "%06d", stats.test_idx);
  snprintf(src_str, 7, "%06d", stats.queue_idx);

  // TODO: mall, mtopk dont have src:000000
  string fname = out_dir + "/id:" + idx_str + ",src:" + src_str + ",op:" + stats.stage;
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
