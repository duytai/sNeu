
CXX = clang++
CXXFLAGS += -Wno-pointer-sign -Wno-write-strings

target=bin/fuzzer

all: main
	cd bin/ && ./fuzzer -i /root/neuzz/programs/nm/neuzz_in /root/neuzz/programs/nm/nm-new -C @@

%.o: %.cpp
	$(CXX) -c -o $@ $< $(CXXFLAGS)

main: main.cpp fuzzer.o
	@mkdir -p bin/
	$(CXX) -o $(target) $^ -fsanitize=address $(CXXFLAGS)

clean:
	@rm -rf bin/ *.o $(target)
