#include <libfuzzer/testsuit.h>
#include <libfuzzer/fuzzer.h>
#include <filesystem>
#include <algorithm>
#include <fstream>

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
void TestSuite::load_from_in_dir() {
  this->load_from_dir(this->opt.in_dir);
}

