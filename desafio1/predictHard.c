#include <stdio.h>
#include <time.h>

int main(){
	int i = 0;
	int j = 0;
	int array[100000];
	int foo = 0;

	clock_t begin = clock();

	for(i = 0; i<100000; i++){
		array[i] = i;
	}

	for (j = 0; j < 50000; j++){
		for(i = 0; i < 99999; i++){
			if(i&1){
				array[i+1] = array[i];
			}
			else{
				array[i+1] = array[i];
			}
		}
	}

	clock_t end = clock();

	double execTime = (double)(end-begin)/CLOCKS_PER_SEC;

	printf("Unoptimized execution took %f seconds.\n", execTime);

	
}