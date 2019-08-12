all: run

test: main.c
	gcc -fopenmp -Wall -ansi -pedantic -lpthread -o main main.c
run: test

clean:
	rm -f main
