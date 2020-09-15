AR=ar
CC=clang
CXX=clang++

CFLAGS=-Xclang -load -Xclang ./libcmppass.so
LDFLAGS=

all: test

%.o: %.c
	$(CC) -c -o $@ $<

libcmppass.so: cmppass.cpp
	$(CXX) -shared -o $@ -fPIC $<

libafl-llvm-rt.a: afl-llvm-rt.o
	$(AR) cr $@ afl-llvm-rt.o

test: libcmppass.so cmpcov.o
	$(CC) -c $@.c -o $@.o $(CFLAGS)
	$(CC) $@.o cmpcov.o -o $@ $(LDFLAGS)

clean:
	rm -f *.so *.o *.a test
