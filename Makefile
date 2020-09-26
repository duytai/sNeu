AR=ar
CC=clang
CFLAGS=-fsanitize-coverage=trace-cmp

all: test

%.o: %.c
	$(CC) -c -o $@ $<

test: cmpcov.o test.c
	$(CC) -c -g $@.c -o $@.o $(CFLAGS)
	$(CC) $@.o -o $@ cmpcov.o

clean:
	rm -f *.so *.o *.a test
