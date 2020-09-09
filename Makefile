AR=ar
CC=clang
OBJS=cmpcov.o

all: test

test: libcmpcov.a test.c
	$(CC) -c test.c -o test.o -fsanitize=address -fsanitize-coverage=trace-pc-guard,trace-cmp
	$(CC) test.o -o test -fsanitize=address -Wl,--whole-archive -L./ -lcmpcov -Wl,--no-whole-archive
	$(CC) test.c -o asan -fsanitize=address -fsanitize-coverage=trace-pc-guard

%.o: %.c
	$(CC) -c -o $@ $< -fPIE

libcmpcov.a: $(OBJS)
	$(AR) cr $@ $(OBJS)

clean:
	rm -f *.o test asan
	rm -f libcmpcov.a
	rm -rf logs/
