all: run

matrixEasy: matrixEasy.c
	gcc -o matrixEasy matrixEasy.c

matrixHard: matrixHard.c
	gcc -o matrixHard matrixHard.c

run: matrixEasy matrixHard
	./matrixEasy
	./matrixHard

clean:
	rm -f matrixEasy
	rm -f matrixHard
