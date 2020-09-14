AR=ar
CC=clang
CXX=clang++

all: test

libcmppass.so: cmppass.cpp
	$(CXX) -shared -o $@ -fPIC $<

%.o: %.c
	$(CC) -c -o $@ $<

libcmpcov.a: cmpcov.o
	$(AR) cr $@ cmpcov.o

test: libcmppass.so libcmpcov.a
	$(CC) -Xclang -load -Xclang ./libcmppass.* $@.c -L./ -lcmpcov -o $@

clean:
	rm -f *.so *.o *.a test
