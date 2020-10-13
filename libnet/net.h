#include <torch/torch.h>
#include <libfuzzer/types.h>

class Net: torch::nn::Module {
  public:
    Net(u64 n);
    torch::Tensor forward(torch::Tensor X);
    torch::nn::Linear in{nullptr}, h{nullptr}, out{nullptr};
};
