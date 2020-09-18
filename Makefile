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

test: libcmppass.so cmpcov.o
	./clang.py -c $@.c -o $@.o
	./clang.py $@.o -o $@

clean:
	rm -f *.so *.o *.a test
