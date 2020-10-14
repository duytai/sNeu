#include <torch/torch.h>
#include <libfuzzer/testsuit.h>
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
      TestCase t = {.buffer = buffer};
      this->testcases.push_back(t);
    }
  }
}

void TestSuite::exec_remaining(void) {
  for (auto& t: this->testcases) {
    if (!t.executed) {
      this->fuzzer->run_target(t.buffer, EXEC_TIMEOUT);
      vector<u8> loss_bits(this->fuzzer->loss_bits, this->fuzzer->loss_bits + MAP_SIZE);
      t.loss_bits = loss_bits;
      t.hnb = this->fuzzer->hnb;
      t.executed = true;
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
  this->compute_branch_loss();
  
  u32 max_len = 0;

  for (auto t : this->testcases) {
    if (t.min_loss != 255) {
      max_len = t.buffer.size() > max_len ? t.buffer.size() : max_len;
    }
  }
  OKF("max_len: %d", max_len);

  vector<torch::Tensor> xs;
  vector<torch::Tensor> ys;

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

  OKF("Num testcases : %d", this->testcases.size());
  OKF("Num train     : %d", xs.size());

  auto net = std::make_shared<Net>(max_len);
  torch::optim::SGD optimizer(net->parameters(), /*lr=*/0.01);

  /* Training */
  for (u32 epoch = 0; epoch < 1000; epoch += 1) {
    optimizer.zero_grad();
    torch::Tensor prediction = net->forward(torch::stack(xs));
    torch::Tensor loss = torch::mse_loss(prediction, torch::stack(ys));
    loss.backward();
    optimizer.step();
    if (epoch + 1 == 1000) {
      OKF("Epoch : %d | Loss: %.4f", epoch, loss.item<float>());
    }
  }

  ACTF("Train finished, its time to mutate");

  /* Mutation */
  u32 num_found = 0;
  u32 num_total = 0;
  for (auto x : xs) {
    /* Compute grads for input x */
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
      vector t((char*) temp.data_ptr(), (char*) temp.data_ptr() + temp.numel());
      this->fuzzer->run_target(t, EXEC_TIMEOUT);
      if (this->fuzzer->hnb) {
        num_found += 1;
        OKF("Perf %d/%d", num_found, num_total);
      }
      num_total += 1;
      /* Zero grad for next round */
      x.grad().zero_();
      x.set_requires_grad(true);
    }
  }
  OKF("Perf %d/%d", num_found, num_total);

  /* Predict */
  // auto ins = torch::stack(xs);
  // auto outs = torch::stack(ys);
  // ins.set_requires_grad(true);
  // optimizer.zero_grad();
  // torch::Tensor prediction = net->forward(ins);
  // torch::Tensor loss = torch::mse_loss(prediction, outs);
  // loss.backward();
  // cout << ins.grad()[0] << endl;
}
