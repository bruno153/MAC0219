#include <stdio.h>
#include <time.h>
#define size 1000
#define iter 5000

int main(){
	int matrix[size][size];
	int i = 0, j = 0, k = 0;
	int sum = 0;

	clock_t begin = clock();

	for(i = 0; i<size; i++){
		for(j = 0; j<size; j++){
			matrix[i][j] = i;
		}
	}

	for(k = 0; k < iter; k++){
		sum = 0;
		for(i = 0; i < size; i++){
			for(j = 0; j < size; j++){
				sum += matrix[i][j];
			}
		}
	}

	clock_t end = clock();

	double execTime = (double)(end-begin)/CLOCKS_PER_SEC;

	printf("Optimized execution took %f seconds.\n", execTime);

}