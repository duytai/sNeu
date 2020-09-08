AR=ar
CC=clang
OBJS=cmpcov.o

all: test

test: libcmpcov.a test.c
	$(CC) -c test.c -o test.o -fsanitize=address -fsanitize-coverage=trace-pc-guard,trace-cmp
	$(CC) test.o -o test -fsanitize=address -Wl,--whole-archive -L./ -lcmpcov -Wl,--no-whole-archive

%.o: %.c
	$(CC) -c -o $@ $< -fPIE

libcmpcov.a: $(OBJS)
	$(AR) cr $@ $(OBJS)

clean:
	rm -f *.o test
	rm libcmpcov.a
