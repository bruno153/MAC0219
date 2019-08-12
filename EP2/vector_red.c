#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#define SIZE 3

int main(int argc, char* argv[]){
	int i, j, iAmount, *iiM, iMin, *iAns;
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
	iiM = malloc(iAmount*sizeof(int));

	/* read user input */
	for (i = 0; i < iAmount; ++i){
		/*throw first line*/
		fgets(buffer, 100, file);
		fscanf(file, "%d \n", &iiM[i]);
	}

	/*------------------------------------------------------------*/

	iMin = iiM[0];
	for (i = 1; i < iAmount; ++i){
		if (iMin > iiM[i]){
			iMin = iiM[i];
		}
	}

	printf("The result is %d\n", iMin);
	return 0;
}