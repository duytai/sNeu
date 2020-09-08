// trace-pc-guard-example.cc
void foo() { }
int main(int argc, char **argv) {
  if (argc > 10000) foo();
}
