AR=ar
CC=clang
CXX=clang++

all: test

%.o: %.c
	$(CC) -c -o $@ $<

libcmppass.so: cmppass.cpp
	$(CXX) -shared -o $@ -fPIC $<

libafl-llvm-rt.a: afl-llvm-rt.o
	$(AR) cr $@ afl-llvm-rt.o

test: libcmppass.so cmpcov.o test.c
	./clang.py -c -g $@.c -o $@.o
	./clang.py $@.o -o $@

clean:
	rm -f *.so *.o *.a test
	rm -rf .logs
