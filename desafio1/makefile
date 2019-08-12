all: run

predictEasy: predictEasy.c
	gcc -o predictEasy predictEasy.c

predictHard: predictHard.c
	gcc -o predictHard predictHard.c

run: predictHard predictEasy
	./predictEasy
	./predictHard

clean:
	rm -f predictEasy
	rm -f predictHard