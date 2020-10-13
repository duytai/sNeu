#include <libfuzzer/testsuit.h>
#include <libfuzzer/fuzzer.h>
#include <filesystem>
#include <algorithm>
#include <fstream>

using namespace std;
using namespace std::filesystem;

TestSuite::TestSuite(Fuzzer* fuzzer) {
  this->fuzzer = fuzzer;
}

void TestSuite::load_from_dir(char* in_dir) {
  vector<directory_entry> files((directory_iterator(in_dir)), directory_iterator());
  sort(files.begin(), files.end());

  for (auto &file: files) {
    if (file.is_regular_file() && file.file_size() > 0) {
      ifstream st(file.path(), ios::binary);
      vector<char> buffer((istreambuf_iterator<char>(st)), istreambuf_iterator<char>());
    }
  }
}

