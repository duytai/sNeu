AR=ar
CC=clang
CXX=clang++

CFLAGS=-Xclang -load -Xclang ./libcmppass.so
LDFLAGS=-L./ -lcmpcov

all: test

%.o: %.c
	$(CC) -c -o $@ $<

libcmppass.so: cmppass.cpp
	$(CXX) -shared -o $@ -fPIC $<

libcmpcov.a: cmpcov.o
	$(AR) cr $@ cmpcov.o

test: libcmppass.so libcmpcov.a
	$(CC) -c $@.c -o $@.o $(CFLAGS)
	$(CC) $@.o -o $@ $(LDFLAGS)

clean:
	rm -f *.so *.o *.a test
