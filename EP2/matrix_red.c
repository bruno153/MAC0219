#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#define SIZE 3

int main(int argc, char* argv[]){
	int i, j, iAmount, **iiM, iMin, *iAns;
	char buffer[100];
	FILE *file;
	/* check input */
	if (argc != 2){
		printf("Usage: %s <caminho_lista_matriz>\n", argv[0]);
		return 3;
	}

	file = fopen(argv[1], "r");

	/* read amount of matrices */
	fscanf(file, "%d\n", &iAmount);

	/* allocate matrices */
	iiM = malloc(SIZE*SIZE*sizeof(int*));
	for (i = 0; i < SIZE*SIZE; ++i){
		iiM[i] = malloc(iAmount*sizeof(int));
	}
	iAns = malloc(SIZE*SIZE*sizeof(int));

	/* read user input */
	for (i = 0; i < iAmount; ++i){
		/*throw first line*/
		fgets(buffer, 100, file);
		for (j = 0; j < SIZE; ++j){
			fscanf(file, "%d %d %d \n", &iiM[0+j*3][i], &iiM[1+j*3][i], &iiM[2+j*3][i]);
		}
	}

	/*------------------------------------------------------------*/

	for (i = 0; i < SIZE*SIZE; ++i){
		iMin = iiM[i][0];
		for (j = 1; j < iAmount; ++j) {
			if (iMin > iiM[i][j]){
				iMin = iiM[i][j];
			}
		}
		iAns[i] = iMin;
	}

	for (i = 0; i < SIZE; ++i){
		for (j = 0; j < SIZE; ++j){
			printf("%d ", iAns[i*3 + j]);
		}
		printf("\n");
	}
	return 0;
}