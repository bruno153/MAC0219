O objetivo dos dois programas é somar todos os valores de uma matriz 1000x1000,
cujos valores são inicializados como índice da sua coluna. 

Ambos os programas iteram buscando o valor de cada posição e somando na variável sum,
entretanto, matrixEasy itera por linha primeiro, enquanto que matrixHard itera
primeiro em coluna. matrixEasy faz melhor uso do cache pois ao acessar o primeiro
valor de uma coluna, é provavel que no cache seja armazenado a linha toda, minimizando
o número de acessos a memória lenta. Iterando por coluna, por outro lado, diminui a chance
dos acessos consecutivos estarem perto na memória, lotando o cache e maximizando o número
de acesos a memória lenta.

Exemplo de execução:
gcc     matrixEasy.c   -o matrixEasy
gcc     matrixHard.c   -o matrixHard
./matrixEasy
Optimized execution took 12.661821 seconds.
./matrixHard
Optimized execution took 23.503084 seconds.

