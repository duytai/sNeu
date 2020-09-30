CC=clang

all: test fuzzer

%.o: %.c
	$(CC) -c -o $@ $< -Wno-pointer-sign

libcmpcov.a: cmpcov.o 
	$(AR) cr $@ $^

libllvmrt.a: llvmrt.o
	$(AR) cr $@ $^

fuzzer: fuzzer.c
	$(CC) -o $@ $^ 
	#-fsanitize=address 

test: libcmpcov.a libllvmrt.a test.c
	./clang.py -c -g $@.c -o $@.o
	./clang.py $@.o -o $@

clean:
	rm -f *.so *.o *.a fuzzer
