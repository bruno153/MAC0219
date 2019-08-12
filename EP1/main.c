#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <pthread.h>


#define SIZE_THRES 150
#define NUM_THREADS	8

pthread_mutex_t mutexnum;

double *freeMatrix(double* m);
double *allocateMatrix(uint64_t l, uint64_t c);
void sumMatrix(double* mA, double* mB, double* mC, uint64_t l, uint64_t c);
double* transposeMatrix(double* mA, uint64_t l, uint64_t c);
void multiplyMatrix(double* mA, double* mB, double* mC, uint64_t lA, uint64_t cA, uint64_t cB);
void multiplyMatrixP(double* mA, double* mB, double* mC, uint64_t lA, uint64_t cA, uint64_t cB);
double* transposeMatrixP(double* mA, uint64_t l, uint64_t c);
void multiplyMatrixRec(double* mA, double* mB, double* mC, uint64_t lA, uint64_t cA, uint64_t cB);
void multiplyMatrixOpenMP(double* mA, double* mB, double* mC, uint64_t lA, uint64_t cA, uint64_t cB);
void multiplyMatrixRecOpenMP(double* mA, double* mB, double* mC, uint64_t lA, uint64_t cA, uint64_t cB);



int main (int argc, char* argv[]){
	uint64_t  lA,  cA, i, j, lB, cB, lC, cC;
	double dNumber, *mA, *mB, *mC, execTime;
	FILE *fA;
	FILE *fB;
	FILE *fC;
	char mode;
	double begin, end;

	if (argc != 5){
		printf("Usage: %s <Modo_p_ou_o> <caminho_matriz_A> <caminho_matriz_B> <caminho_matriz_C>\n", argv[0]);
		return 5;
	}
	mode = argv[1][0];
	fA = fopen(argv[2], "r");
	fB = fopen(argv[3], "r");
	fC = fopen(argv[4], "w");

	/* get matrix A dimension */
	fscanf(fA, "%lu %lu", &lA, &cA);
	/* allocate matrix A */
	mA = allocateMatrix(lA, cA);
	/* write on matrix A */
	while (fscanf(fA, "%lu %lu %lf", &i, &j, &dNumber) != EOF){
		mA[(i-1)*cA + (j-1)] = dNumber;
	}
	/* close file */
	fclose(fA);

	/* get matrix B dimension */
	fscanf(fB, "%lu %lu", &lB, &cB);
	/* allocate matrix B */
	mB = allocateMatrix(lB, cB);
	/* write on matrix B */
	while (fscanf(fB, "%lu %lu %lf", &i, &j, &dNumber) != EOF){
		mB[(i-1)*cB + (j-1)] = dNumber;
	}
	/* close file */
	fclose(fB);

	/*allocate matrix C*/
	lC = lA;
	cC = cB;
	mC = allocateMatrix(lC, cC);
	multiplyMatrix(mA, mB, mC, lA, cA, cB);

	printf("Read done. Begin multiplication.\n");
	
	/*
	begin = omp_get_wtime();
	
	multiplyMatrix(mA, mB, mC, lA, cA, cB);
	
	end = omp_get_wtime();
	execTime = (double)(end-begin);
	printf("Unoptimized execution took %f seconds.\n", execTime);
	*/
	if (mode=='p')
	{
		/* code */
	
		begin = omp_get_wtime();
		
		multiplyMatrixRec(mA, mB, mC, lA, cA, cB);
		end = omp_get_wtime();
		execTime = (double)(end-begin);
		printf("Optimized execution took %f seconds.\n", execTime);
	
	}
	else if (mode=='o')
	{
		begin = omp_get_wtime();
		
		multiplyMatrixRecOpenMP(mA, mB, mC, lA, cA, cB);
		end = omp_get_wtime();
		execTime = (double)(end-begin);
		printf("Optimized execution took %f seconds.\n", execTime);
	}
	/*
	printf("Matrix A\n");
	for (i = 0; i < lA; ++i){
		for (j = 0; j < cA; ++j){
			printf("%.5f ", mA[(i)*cA + j]);
		}
		printf("\n");
	}
	printf("Matrix B\n");
	for (i = 0; i < lB; ++i){
		for (j = 0; j < cB; ++j){
			printf("%.5f ", mB[i*cB + j]);
		}
		printf("\n");
	}
	printf("Matrix C\n");
	for (i = 0; i < lC; ++i){
		for (j = 0; j < cC; ++j){
			printf("%.5f ", mC[i*cC + j]);
		}
		printf("\n");
	}
	*/
	
	fprintf(fC, "%lu %lu\n", lC, cC);
	for (i = 0; i < lC; ++i){
		for (j = 0; j < cC; ++j){
			if (mC[i*cC + j] != 0.0)
				fprintf(fC, "%lu %lu %f\n", i+1, j+1, mC[i*cC + j]);
		}
	}

	fclose(fC);
	free(mA);
	free(mB);
	free(mC);

	return 0;
}

double *freeMatrix(double* m){
	/* if matrix is alredy NULL end operation */
	if (m == NULL){
		return NULL;
	}
	free(m);
	m = NULL;

	return NULL;
}

double *allocateMatrix(uint64_t l, uint64_t c){
	double *m;
	m = calloc(l*c, sizeof(double));

	if (m == NULL){
		printf("MALLOC FAILED\n");
		exit(-1);
	}

	return m;
}

void sumMatrix(double* mA, double* mB, double* mC, uint64_t l, uint64_t c){
	uint64_t i;
	for (i = 0; i < l*c; ++i){
		mC[i] = mA[i] + mB[i];
	}
}

double* transposeMatrix(double* mA, uint64_t l, uint64_t c){
	double* mB = allocateMatrix(c, l);
	uint64_t i , j;
	for (i = 0; i < l; ++i){
		for (j = 0; j < c; ++j){
			mB[j*l + i] = mA[i*c + j];
		}
	}
	return mB;
}
/*
double* transposeMatrixP(double* mA, uint64_t l, uint64_t c){
	double* mB = allocateMatrix(c, l);
	#pragma omp parallel
	{
		uint64_t i , j;
		#pragma omp for 
		for (i = 0; i < l; ++i){
			for (j = 0; j < c; ++j){
				mB[j*l + i] = mA[i*c + j];
			}
		}
	}
	return mB;
}
*/
void multiplyMatrix(double* mA, double* mB, double* mC, uint64_t lA, uint64_t cA, uint64_t cB){
	uint64_t i, j, k;
	double sum = 0;
	for (i = 0; i < lA; ++i){
		for (j = 0; j < cB; ++j){
			for (k = 0; k < cA; ++k){
				sum = sum + mA[i*cA + k]*mB[k*cB + j];
			}
			mC[i*cB + j] = sum;
			sum = 0;
		}
	}
}


typedef struct infos {
	int indice;
	uint64_t n,m,p;
	double *mA, *mB, *mC;
	
}SUBMATRIZ_MULTIP;



void* multiplica (void* structs){

	uint64_t i, j, k ,n,m,p;
	double sum;
	int indice;
	double *mA,*mB,*mC;

	SUBMATRIZ_MULTIP *argumentos = structs;

	indice = argumentos->indice;
	n = argumentos->n;
	p = argumentos->p;
	m = argumentos->m;
	mA = argumentos->mA;
	mB = argumentos->mB;
	mC = argumentos->mC;

	for (i = indice; (i <  n ) ; i+=NUM_THREADS){
		
		for (j = 0; j < m; ++j){
			sum = 0;	
			for (k = 0; k < p; ++k){

				
				sum += mA[i*p + k]*mB[k*m + j];
			}
			mC[i*m + j] = sum;
		}
	}
	pthread_exit(NULL);
}

void multiplyMatrixP(double* mA, double* mB, double* mC, uint64_t lA, uint64_t cA, uint64_t cB){
	
/*cria threads e structs das funcoes*/
	int z;
	pthread_t threads[NUM_THREADS];
	

	SUBMATRIZ_MULTIP structs[NUM_THREADS];
	

/*loop q inicializa structs e threads*/
	for(z = 0 ; z< NUM_THREADS; z++){

		structs[z].indice = z;
		structs[z].n = lA;
		structs[z].p = cA;
		structs[z].m = cB;
		structs[z].mA = mA;
		structs[z].mB = mB;
		structs[z].mC = mC;
		

		pthread_create(&threads[z], NULL, multiplica, (void*) &structs[z] );
		
	}


	for (z = 0; z < NUM_THREADS; ++z)
	{
		pthread_join(threads[z], NULL);


	}


}


/*

###############################################################################################

*/

void multiplyMatrixOpenMP(double* mA, double* mB, double* mC, uint64_t lA, uint64_t cA, uint64_t cB){

	#pragma omp parallel 
	{
		uint64_t i, j, k;
		double sum;
		#pragma omp for
		for (i = 0; i < lA; ++i){
			for (j = 0; j < cB; ++j){
				sum = 0;
				for (k = 0; k < cA; ++k){
					sum += mA[i*cA + k]*mB[k*cB + j];
				}
				mC[i*cB + j] = sum;
			}
		}
	}
	
}

void multiplyMatrixRec(double* mA, double* mB, double* mC, uint64_t lA, uint64_t cA, uint64_t cB){
	uint64_t lA1, lA2, cA1, cA2, lB1, lB2, cB1, cB2, i , j;
	double *mA11, *mA12, *mA21, *mA22, *mB11, *mB12, *mB21, *mB22, *mC11, *mC12, *mC21, *mC22; 
	double *mT111, *mT112, *mT121, *mT122, *mT211, *mT212, *mT221, *mT222;
	/*if the matrix fits on the cache and all measures are greater than 1
	stop recursive calls*/
	if(lA < SIZE_THRES && cA < SIZE_THRES && cB < SIZE_THRES){

		
		multiplyMatrixP(mA, mB, mC, lA, cA, cB);
		
		return;
	}

	/* commence divide and conquer in 4 block matrices*/
	/* get block matrix dimension */
	lA1 = lA/2;
	lA2 = lA-lA1;
	cA1 = cA/2;
	cA2 = cA-cA1;
	lB1 = cA/2;
	lB2 = cA-lB1;
	cB1 = cB/2;
	cB2 = cB-cB1;
	/* allocate matrices */
	mA11 = allocateMatrix(lA1, cA1);
	mA12 = allocateMatrix(lA1, cA2);
	mA21 = allocateMatrix(lA2, cA1);
	mA22 = allocateMatrix(lA2, cA2);
	mB11 = allocateMatrix(lB1, cB1);
	mB12 = allocateMatrix(lB1, cB2);
	mB21 = allocateMatrix(lB2, cB1);
	mB22 = allocateMatrix(lB2, cB2);
	mC11 = allocateMatrix(lA1, cB1);
	mC12 = allocateMatrix(lA1, cB2);
	mC21 = allocateMatrix(lA2, cB1);
	mC22 = allocateMatrix(lA2, cB2);
	mT111 = allocateMatrix(lA1, cB1);
	mT112 = allocateMatrix(lA1, cB1);
	mT121 = allocateMatrix(lA1, cB2);
	mT122 = allocateMatrix(lA1, cB2);
	mT211 = allocateMatrix(lA2, cB1);
	mT212 = allocateMatrix(lA2, cB1);
	mT221 = allocateMatrix(lA2, cB2);
	mT222 = allocateMatrix(lA2, cB2);
	/* fill matrices */
	for (i = 0; i < lA; ++i){
		for (j = 0; j < cA; ++j){
			if (i < lA1)
				if (j < cA1)
					mA11[i*cA1 + j] = mA[i*cA + j];
				else
					mA12[i*cA2 + j - cA1] = mA[i*cA + j];
			else
				if (j < cA1)
					mA21[(i-lA1)*cA1 + j] = mA[i*cA + j];
				else
					mA22[(i-lA1)*cA2 + j - cA1] = mA[i*cA + j];
		}
	}

	for (i = 0; i < cA; ++i){
		for (j = 0; j < cB; ++j){
			if (i < lB1)
				if (j < cB1)
					mB11[i*cB1 + j] = mB[i*cB + j];
				else
					mB12[i*cB2 + j - cB1] = mB[i*cB + j];
			else
				if (j < cB1)
					mB21[(i-lB1)*cB1 + j] = mB[i*cB + j];
				else
					mB22[(i-lB1)*cB2 + j - cB1] = mB[i*cB + j];
		}
	}
	/* multiplication */
	multiplyMatrixRec(mA11, mB11, mT111, lA1, cA1, cB1);
	multiplyMatrixRec(mA12, mB21, mT112, lA1, cA2, cB1);
	multiplyMatrixRec(mA11, mB12, mT121, lA1, cA1, cB2);
	multiplyMatrixRec(mA12, mB22, mT122, lA1, cA2, cB2);
	multiplyMatrixRec(mA21, mB11, mT211, lA2, cA1, cB1);
	multiplyMatrixRec(mA22, mB21, mT212, lA2, cA2, cB1);
	multiplyMatrixRec(mA21, mB12, mT221, lA2, cA1, cB2);
	multiplyMatrixRec(mA22, mB22, mT222, lA2, cA2, cB2);
	/* sum */
	sumMatrix(mT111, mT112, mC11, lA1, cB1);
	sumMatrix(mT121, mT122, mC12, lA1, cB2);
	sumMatrix(mT211, mT212, mC21, lA2, cB1);
	sumMatrix(mT221, mT222, mC22, lA2, cB2);

	/* build the answer */
	for (i = 0; i < lA; ++i){
		for (j = 0; j < cB; ++j){
			if (i < lA1){
				if (j < cB1)
					mC[i*cB + j] = mC11[i*cB1 + j];
				else
					mC[i*cB + j] = mC12[i*cB2 + j - cB1];
			}
			else{
				if (j < cB1)
					mC[i*cB + j] = mC21[(i-lA1)*cB1 + j];
				else
					mC[i*cB + j] = mC22[(i-lA1)*cB2 + j - cB1];
			}
		}
	}

	free(mA11);
	free(mA12);
	free(mA21);
	free(mA22);
	free(mB11);
	free(mB12);
	free(mB21);
	free(mB22);
	free(mC11);
	free(mC12);
	free(mC21);
	free(mC22);
	free(mT111);
	free(mT112);
	free(mT121);
	free(mT122);
	free(mT211);
	free(mT212);
	free(mT221);
	free(mT222);
}



void multiplyMatrixRecOpenMP(double* mA, double* mB, double* mC, uint64_t lA, uint64_t cA, uint64_t cB){
	uint64_t lA1, lA2, cA1, cA2, lB1, lB2, cB1, cB2, i , j;
	double *mA11, *mA12, *mA21, *mA22, *mB11, *mB12, *mB21, *mB22, *mC11, *mC12, *mC21, *mC22; 
	double *mT111, *mT112, *mT121, *mT122, *mT211, *mT212, *mT221, *mT222;
	/*if the matrix fits on the cache and all measures are greater than 1
	stop recursive calls*/
	if(lA < SIZE_THRES && cA < SIZE_THRES && cB < SIZE_THRES){

		
		multiplyMatrixOpenMP(mA, mB, mC, lA, cA, cB);
		
		return;
	}

	/* commence divide and conquer in 4 block matrices*/
	/* get block matrix dimension */
	lA1 = lA/2;
	lA2 = lA-lA1;
	cA1 = cA/2;
	cA2 = cA-cA1;
	lB1 = cA/2;
	lB2 = cA-lB1;
	cB1 = cB/2;
	cB2 = cB-cB1;
	/* allocate matrices */
	mA11 = allocateMatrix(lA1, cA1);
	mA12 = allocateMatrix(lA1, cA2);
	mA21 = allocateMatrix(lA2, cA1);
	mA22 = allocateMatrix(lA2, cA2);
	mB11 = allocateMatrix(lB1, cB1);
	mB12 = allocateMatrix(lB1, cB2);
	mB21 = allocateMatrix(lB2, cB1);
	mB22 = allocateMatrix(lB2, cB2);
	mC11 = allocateMatrix(lA1, cB1);
	mC12 = allocateMatrix(lA1, cB2);
	mC21 = allocateMatrix(lA2, cB1);
	mC22 = allocateMatrix(lA2, cB2);
	mT111 = allocateMatrix(lA1, cB1);
	mT112 = allocateMatrix(lA1, cB1);
	mT121 = allocateMatrix(lA1, cB2);
	mT122 = allocateMatrix(lA1, cB2);
	mT211 = allocateMatrix(lA2, cB1);
	mT212 = allocateMatrix(lA2, cB1);
	mT221 = allocateMatrix(lA2, cB2);
	mT222 = allocateMatrix(lA2, cB2);
	/* fill matrices */
	for (i = 0; i < lA; ++i){
		for (j = 0; j < cA; ++j){
			if (i < lA1)
				if (j < cA1)
					mA11[i*cA1 + j] = mA[i*cA + j];
				else
					mA12[i*cA2 + j - cA1] = mA[i*cA + j];
			else
				if (j < cA1)
					mA21[(i-lA1)*cA1 + j] = mA[i*cA + j];
				else
					mA22[(i-lA1)*cA2 + j - cA1] = mA[i*cA + j];
		}
	}

	for (i = 0; i < cA; ++i){
		for (j = 0; j < cB; ++j){
			if (i < lB1)
				if (j < cB1)
					mB11[i*cB1 + j] = mB[i*cB + j];
				else
					mB12[i*cB2 + j - cB1] = mB[i*cB + j];
			else
				if (j < cB1)
					mB21[(i-lB1)*cB1 + j] = mB[i*cB + j];
				else
					mB22[(i-lB1)*cB2 + j - cB1] = mB[i*cB + j];
		}
	}
	/* multiplication */
	multiplyMatrixRec(mA11, mB11, mT111, lA1, cA1, cB1);
	multiplyMatrixRec(mA12, mB21, mT112, lA1, cA2, cB1);
	multiplyMatrixRec(mA11, mB12, mT121, lA1, cA1, cB2);
	multiplyMatrixRec(mA12, mB22, mT122, lA1, cA2, cB2);
	multiplyMatrixRec(mA21, mB11, mT211, lA2, cA1, cB1);
	multiplyMatrixRec(mA22, mB21, mT212, lA2, cA2, cB1);
	multiplyMatrixRec(mA21, mB12, mT221, lA2, cA1, cB2);
	multiplyMatrixRec(mA22, mB22, mT222, lA2, cA2, cB2);
	/* sum */
	sumMatrix(mT111, mT112, mC11, lA1, cB1);
	sumMatrix(mT121, mT122, mC12, lA1, cB2);
	sumMatrix(mT211, mT212, mC21, lA2, cB1);
	sumMatrix(mT221, mT222, mC22, lA2, cB2);

	/* build the answer */
	for (i = 0; i < lA; ++i){
		for (j = 0; j < cB; ++j){
			if (i < lA1){
				if (j < cB1)
					mC[i*cB + j] = mC11[i*cB1 + j];
				else
					mC[i*cB + j] = mC12[i*cB2 + j - cB1];
			}
			else{
				if (j < cB1)
					mC[i*cB + j] = mC21[(i-lA1)*cB1 + j];
				else
					mC[i*cB + j] = mC22[(i-lA1)*cB2 + j - cB1];
			}
		}
	}

	free(mA11);
	free(mA12);
	free(mA21);
	free(mA22);
	free(mB11);
	free(mB12);
	free(mB21);
	free(mB22);
	free(mC11);
	free(mC12);
	free(mC21);
	free(mC22);
	free(mT111);
	free(mT112);
	free(mT121);
	free(mT122);
	free(mT211);
	free(mT212);
	free(mT221);
	free(mT222);
}
