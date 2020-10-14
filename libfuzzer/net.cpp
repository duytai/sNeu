#include <torch/torch.h>
#include <libfuzzer/net.h>

Net::Net(u64 n) {
  this->in = register_module("in", torch::nn::Linear(n, 64));
  this->h = register_module("h", torch::nn::Linear(64, 16));
  this->out = register_module("out", torch::nn::Linear(16, 1));
}

torch::Tensor Net::forward(torch::Tensor X) {
  X = torch::relu(in->forward(X));
  X = torch::relu(h->forward(X));
  X = torch::sigmoid(out->forward(X));
  return X;
}
