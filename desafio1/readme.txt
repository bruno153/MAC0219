O objetivo dos dois programas é transportar um valor do índice
[0] até a última posição do vetor de tamanho [100000], passando
por todas as posições.

Em ambos os programas estão presentes if-else, entretanto, na 
o programa realiza a mesma operação de transporte de valor
independente do resultado da condicional. Entretanto, a condicional
if do predictEasy é [true] sempre; enquanto que no predictHard, 
a condição [true] e [false] alternam em cada iteração.

Pior performance é esperado no caso do predictHard devido a
forma como branch predictor funciona. Assumindo que branch
predictor faz o uso da frequência da decisão tomada, escolhendo
executar a branch mais comum até então, predictEasy acertará todas
as predictions no caso do predictEasy, pois todas as condições
são [true], enquanto que no predictHard, devido a alternância
da condicional, é esperado que predictor erre mais, levando
a mais bolhas e uma execução mais lenta.

Exemplo de execução:
gcc -o predictHard predictHard.c
gcc -o predictEasy predictEasy.c
./predictEasy
Optimized execution took 13.666809 seconds.
./predictHard
Unoptimized execution took 15.034015 seconds.