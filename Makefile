
CXX= clang++
CXXFLAGS += -Wno-pointer-sign -pedantic

all: fuzzer

fuzzer: fuzzer.cpp
	$(CXX) -o $@ $^ -fsanitize=address $(CXXFLAGS)

clean:
	rm -f fuzzer
