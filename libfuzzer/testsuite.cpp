#include <torch/torch.h>
#include <libfuzzer/testsuite.h>
#include <libfuzzer/fuzzer.h>
#include <libfuzzer/net.h>
#include <filesystem>
#include <algorithm>
#include <fstream>
#include <set>

using namespace std;
using namespace std::filesystem;

TestSuite::TestSuite(Fuzzer* fuzzer, SNeuOptions opt) {
  this->fuzzer = fuzzer;
  this->fuzzer->load_opt(opt);
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

  OKF("\tInst branches\t: %lu", inst_branches.size());

  for (auto& t: testcases) {
    u8 loss = 255;
    for (auto br : inst_branches) {
      if (likely(loss & t.loss_bits[br])) {
        loss &= t.loss_bits[br];
      }
    }
    t.min_loss = loss;
  }
}

/*
 * t.min_loss = min(loss_of_br_1, loss_of_br_2, ....)
 * t.min_loss != 255 means that the current testcase
 * modified losses of uncover branches
 * */

vector<TestCase> TestSuite::smart_mutate(vector<TestCase>& testcases) {

  vector<torch::Tensor> xs, ys;
  vector<TestCase> tcs;
  u32 max_len = 0,
      train_epoch = 1000,
      mutate_epoch = 50,
      num_found = 0,
      num_execs  = 0;

  ACTF("Smart Mutation");
  this->compute_branch_loss(testcases);
  for (auto t : testcases) {
    if (t.min_loss != 255) {
      max_len = max((u32) t.buffer.size(), max_len);
    }
  }
  OKF("\tMaxLen\t: %d", max_len);

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
    }
  }
  OKF("\tHas\t: %lu ts", testcases.size());
  OKF("\tTrain\t: %lu ts", xs.size());

  auto net = std::make_shared<Net>(max_len);
  torch::optim::SGD optimizer(net->parameters(), /*lr=*/0.01);

  /* Training */
  for (u32 epoch = 0; epoch < train_epoch; epoch += 1) {
    optimizer.zero_grad();
    torch::Tensor prediction = net->forward(torch::stack(xs));
    torch::Tensor loss = torch::mse_loss(prediction, torch::stack(ys));
    loss.backward();
    optimizer.step();
    if (epoch + 1 == mutate_epoch) {
      OKF("\tEpoch\t: %d", epoch);
      OKF("\tLoss\t: %.4f", loss.item<float>());
    }
  }

  OKF("\tTrain finished, its time to mutate");

  for (auto x : xs) {
    /* Compute grads for input x */
    x.set_requires_grad(true);
    for (u32 epoch = 0; epoch < mutate_epoch; epoch += 1) {
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
        num_found += 1;
        tcs.push_back(this->fuzzer->tc);
      }
      num_execs += 1;

      /* Zero grad for next round */
      x.grad().zero_();
      x.set_requires_grad(true);
    }
  }

  OKF("\tExecs\t: %d/%d", num_found, num_execs);
  return tcs;
}

vector<TestCase> TestSuite::flip(vector<char> buffer, Stage stage) {

  vector<TestCase> tcs;

  switch (stage) {
    case Stage::STAGE_FLIP8:
      for (u32 i = 0; i < buffer.size(); i += 1) {
        buffer.data()[i] ^= 0xFF;
        this->fuzzer->run_target(buffer, EXEC_TIMEOUT);
        if (this->fuzzer->tc.hnb) {
          tcs.push_back(this->fuzzer->tc);
        }
        buffer.data()[i] ^= 0xFF;
      }
      break;

    case Stage::STAGE_FLIP16:
      if (buffer.size() < 2) break;
      for (u32 i = 0; i < buffer.size() - 1; i += 1) {
        *(u16*)(buffer.data() + i) ^= 0xFFFF;
        this->fuzzer->run_target(buffer, EXEC_TIMEOUT);
        if (this->fuzzer->tc.hnb) {
          tcs.push_back(this->fuzzer->tc);
        }
        *(u16*)(buffer.data() + i) ^= 0xFFFF;
      }
      break;

    case Stage::STAGE_FLIP32:
      if (buffer.size() < 3) break;
      for (u32 i = 0; i < buffer.size() - 3; i += 1) {
        *(u32*)(buffer.data() + i) ^= 0xFFFFFFFF;
        this->fuzzer->run_target(buffer, EXEC_TIMEOUT);
        if (this->fuzzer->tc.hnb) {
          tcs.push_back(this->fuzzer->tc);
        }
        *(u32*)(buffer.data() + i) ^= 0xFFFFFFFF;
      }
      break;
  }

  return tcs;
}

vector<TestCase> TestSuite::arith(vector<char> buffer, Stage stage) {

  vector<TestCase> tcs;

  switch (stage) {
    case Stage::STAGE_ARITH8:
      for (u32 i = 0; i < buffer.size(); i += 1) {
        u8 orig = buffer[i];
        for (u32 j = 0; j < ARITH_MAX; j += 1) {
          u8 r1 = orig ^ (orig + j);
          u8 r2 = orig ^ (orig - j);
          if (r1) {
            buffer[i] = orig + j;
            this->fuzzer->run_target(buffer, EXEC_TIMEOUT);
            if (this->fuzzer->tc.hnb) {
              tcs.push_back(this->fuzzer->tc);
            }
          }
          if (r2) {
            buffer[i] = orig - j;
            this->fuzzer->run_target(buffer, EXEC_TIMEOUT);
            if (this->fuzzer->tc.hnb) {
              tcs.push_back(this->fuzzer->tc);
            }
          }
        }
        buffer[i] = orig;
      }
      break;
    case Stage::STAGE_ARITH16:
      for (u32 i = 0; i < buffer.size() - 1; i += 1) {
        u16 orig = *(u16*)(buffer.data() + i);
        for (u32 j = 0; j < ARITH_MAX; j += 1) {
          u16 r1 = orig ^ (orig + j),
              r2 = orig ^ (orig - j),
              r3 = orig ^ SWAP16(SWAP16(orig) + j),
              r4 = orig ^ SWAP16(SWAP16(orig) - j);

          if ((orig & 0xff) + j > 0xff && r1) {
            *(u16*)(buffer.data() + i) = orig + j;
            this->fuzzer->run_target(buffer, EXEC_TIMEOUT);
            if (this->fuzzer->tc.hnb) {
              tcs.push_back(this->fuzzer->tc);
            }
          }
          if ((orig & 0xff) < j && r2) {
            *(u16*)(buffer.data() + i) = orig - j;
            this->fuzzer->run_target(buffer, EXEC_TIMEOUT);
            if (this->fuzzer->tc.hnb) {
              tcs.push_back(this->fuzzer->tc);
            }
          }
          if ((orig >> 8) + j > 0xff && r3) {
            *(u16*)(buffer.data() + i) = SWAP16(SWAP16(orig) + j);
            this->fuzzer->run_target(buffer, EXEC_TIMEOUT);
            if (this->fuzzer->tc.hnb) {
              tcs.push_back(this->fuzzer->tc);
            }
          }
          if ((orig >> 8) < j && r4) {
            *(u16*)(buffer.data() + i) = SWAP16(SWAP16(orig) - j);
            this->fuzzer->run_target(buffer, EXEC_TIMEOUT);
            if (this->fuzzer->tc.hnb) {
              tcs.push_back(this->fuzzer->tc);
            }
          }
        }
        *(u16*)(buffer.data() + i) = orig;
      }
      break;
    case Stage::STAGE_ARITH32:
      for (u32 i = 0; i < buffer.size() - 3; i += 1) {
        u32 orig = *(u32*)(buffer.data() + i);
        for (u32 j = 0; j < ARITH_MAX; j += 1) {
          u16 r1 = orig ^ (orig + j),
              r2 = orig ^ (orig - j),
              r3 = orig ^ SWAP32(SWAP32(orig) + j),
              r4 = orig ^ SWAP32(SWAP32(orig) - j);

          if ((orig & 0xffff) + j > 0xffff && r1) {
            *(u32*)(buffer.data() + i) = orig + j;
            this->fuzzer->run_target(buffer, EXEC_TIMEOUT);
            if (this->fuzzer->tc.hnb) {
              tcs.push_back(this->fuzzer->tc);
            }
          }
          if ((orig & 0xffff) < j && r2) {
            *(u32*)(buffer.data() + i) = orig - j;
            this->fuzzer->run_target(buffer, EXEC_TIMEOUT);
            if (this->fuzzer->tc.hnb) {
              tcs.push_back(this->fuzzer->tc);
            }
          }
          if ((SWAP32(orig) & 0xffff) + j > 0xffff && r3) {
            *(u32*)(buffer.data() + i) = SWAP32(SWAP32(orig) + j);
            this->fuzzer->run_target(buffer, EXEC_TIMEOUT);
            if (this->fuzzer->tc.hnb) {
              tcs.push_back(this->fuzzer->tc);
            }
          }
          if ((SWAP32(orig) & 0xffff) < j && r4) {
            *(u32*)(buffer.data() + i) = SWAP32(SWAP32(orig) - j);
            this->fuzzer->run_target(buffer, EXEC_TIMEOUT);
            if (this->fuzzer->tc.hnb) {
              tcs.push_back(this->fuzzer->tc);
            }
          }
        }
        *(u32*)(buffer.data() + i) = orig;
      }
      break;
  }

  return tcs;
}

void TestSuite::mutate(void) {

  vector<TestCase> tcs;
  char* tmp = "+------+[------------------------,,,,,,]";
  vector<char> buffer(tmp, tmp + strlen(tmp));

  u32 total_execs = this->fuzzer->total_execs;
  tcs = this->flip(buffer, Stage::STAGE_FLIP8);
  OKF("FLIP8 %lu", tcs.size());
  OKF("total_execs: %d", this->fuzzer->total_execs - total_execs);

  total_execs = this->fuzzer->total_execs;
  tcs = this->flip(buffer, Stage::STAGE_FLIP16);
  OKF("FLIP16 %lu", tcs.size());
  OKF("total_execs: %d", this->fuzzer->total_execs - total_execs);

  total_execs = this->fuzzer->total_execs;
  tcs = this->flip(buffer, Stage::STAGE_FLIP32);
  OKF("FLIP32 %lu", tcs.size());
  OKF("total_execs: %d", this->fuzzer->total_execs - total_execs);

  total_execs = this->fuzzer->total_execs;
  tcs = this->arith(buffer, Stage::STAGE_ARITH8);
  OKF("ARITH8 %lu", tcs.size());
  OKF("total_execs: %d", this->fuzzer->total_execs - total_execs);

  total_execs = this->fuzzer->total_execs;
  tcs = this->arith(buffer, Stage::STAGE_ARITH16);
  OKF("ARITH16 %lu", tcs.size());
  OKF("total_execs: %d", this->fuzzer->total_execs - total_execs);

  total_execs = this->fuzzer->total_execs;
  tcs = this->arith(buffer, Stage::STAGE_ARITH32);
  OKF("ARITH32 %lu", tcs.size());
  OKF("total_execs: %d", this->fuzzer->total_execs - total_execs);
}
