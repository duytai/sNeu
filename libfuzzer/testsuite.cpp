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

void TestSuite::load_from_dir(char* dir) {
  vector<directory_entry> files((directory_iterator(dir)), directory_iterator());
  sort(files.begin(), files.end());

  for (auto &file: files) {
    if (file.is_regular_file() && file.file_size() > 0) {
      ifstream st(file.path(), ios::binary);
      vector<char> buffer((istreambuf_iterator<char>(st)), istreambuf_iterator<char>());
      this->fuzzer->run_target(buffer, EXEC_TIMEOUT);
      vector<u8> loss_bits(this->fuzzer->loss_bits, this->fuzzer->loss_bits + MAP_SIZE);
      TestCase t;
      t.buffer = buffer;
      t.loss_bits = loss_bits;
      t.hnb = this->fuzzer->hnb;
      this->testcases.push_back(t);
    }
  }
}

void TestSuite::load_from_in_dir(void) {
  this->load_from_dir(this->opt.in_dir);
}

void TestSuite::compute_branch_loss(void) {
  vector<u32> inst_branches;
  u32 i = 0;

  for (; i < MAP_SIZE; i += 1) {
    if (this->fuzzer->virgin_loss[i] != 0 && this->fuzzer->virgin_loss[i] != 255) {
      set<u8> losses;
      for (auto t: this->testcases) {
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

void TestSuite::smart_mutate(void) {
  ACTF("Smart Mutation");
  vector<torch::Tensor> xs, ys;
  u32 max_len = 0,
      train_epoch = 1000,
      mutate_epoch = 50,
      num_found = 0,
      num_execs  = 0;

  this->compute_branch_loss();
  for (auto t : this->testcases) {
    if (t.min_loss != 255) {
      max_len = max((u32) t.buffer.size(), max_len);
    }
  }
  OKF("\tMaxLen\t: %d", max_len);

  for (auto t : this->testcases) {
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
  OKF("\tHas\t: %lu ts", this->testcases.size());
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
    if (epoch + 1 == 1000) {
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
      if (this->fuzzer->hnb) {
        vector<u8> loss_bits(this->fuzzer->loss_bits, this->fuzzer->loss_bits + MAP_SIZE);
        TestCase t;
        t.buffer = buffer;
        t.loss_bits = loss_bits;
        t.hnb = this->fuzzer->hnb;
        num_found += 1;
      }
      num_execs += 1;

      /* Zero grad for next round */
      x.grad().zero_();
      x.set_requires_grad(true);
    }
  }

  OKF("\tExecs\t: %d/%d", num_found, num_execs);
}

void TestSuite::mutate(void) {
  this->smart_mutate();
}
