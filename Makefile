CC=clang

all: test

%.o: %.c
	$(CC) -c -o $@ $< -fPIE -Wno-pointer-sign

libcmpcov.a: cmpcov.o
	$(AR) cr $@ $^

test: libcmpcov.a test.c
	./clang.py -c -g $@.c -o $@.o
	./clang.py $@.o -o $@

clean:
	rm -f *.so *.o *.a test
