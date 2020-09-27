CC=clang

all: test

%.o: %.c
	$(CC) -c -o $@ $<

test: cmpcov.o test.c
	./clang.py -c -g $@.c -o $@.o
	./clang.py $@.o -o $@

clean:
	rm -f *.so *.o *.a test
