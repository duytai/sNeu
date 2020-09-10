AR=ar
CC=clang

all: test

test: libcmpcov.a libafl-llvm-rt.a test.c
	$(CC) -c test.c -o test.o -fsanitize=address -fsanitize-coverage=trace-pc-guard,trace-cmp
	$(CC) test.o -o test -fsanitize=address -Wl,--whole-archive -L./ -lcmpcov -Wl,--no-whole-archive
	$(CC) test.c -o asan -fsanitize=address -fsanitize-coverage=trace-pc-guard

%.o: %.c
	$(CC) -c -o $@ $< -fPIE -Wno-pointer-sign

libcmpcov.a: cmpcov.o 
	$(AR) cr $@ cmpcov.o

libafl-llvm-rt.a: afl-llvm-rt.o
	$(AR) cr $@ afl-llvm-rt.o

clean:
	rm -f *.o test asan
	rm -f libcmpcov.a libafl-llvm-rt.a
	rm -rf logs/
