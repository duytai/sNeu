CC = clang
CXX = clang++
CXXFLAGS += -Wno-pointer-sign -Wno-write-strings -Wc99-designator -std=c++17
CFLAGS += -Wno-pointer-sign -Wno-write-strings -Wc99-designator

target=bin/fuzzer

all: main instrument
	./bin/fuzzer -i ./example/out/queue ./example/brainfuck @@

%.o: %.cpp
	$(CXX) -c -o $@ $< $(CXXFLAGS)

instrument: afl-llvm-rt.c
	$(CC) -c afl-llvm-rt.c $(CFLAGS)
	$(CC) example/brainfuck.c -o example/brainfuck afl-llvm-rt.o -fsanitize-coverage=trace-pc-guard,trace-cmp,no-prune

main: main.cpp fuzzer.o mutator.o
	@mkdir -p bin/
	$(CXX) -o $(target) $^ -fsanitize=address $(CXXFLAGS)

clean:
	@rm -rf bin/ *.o $(target) .cur_input
