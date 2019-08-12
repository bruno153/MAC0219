#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <unistd.h>
#define SIZE 3
#define BLOCKSIZE 512
__global__ void minvec(int **idata, int size){
	__shared__ int sdata[BLOCKSIZE];
	int s;
	int tid = threadIdx.x;
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int v = blockIdx.y;
	int pseudoIf = i < size;
	/*if (i < size)
		sdata[tid] = idata[v][i];
	else{
		sdata[tid] = idata[v][0];
	}*/
	sdata[tid] = pseudoIf*idata[v][i] + (1 - pseudoIf)*idata[v][i];
	printf("1 sdata = %d\n", sdata[tid]);
	__syncthreads();
	printf("2 sdata = %d\n", sdata[tid]);
	printf("3 sdata = %d\n", sdata[tid]);
	
	for (s = 0; s < 10000; ++s){
		printf("%d\n", s);
	}
	for (s = blockDim.x/2; s > 0; s >>= 1){
		if (tid < s){
			if (sdata[tid] > sdata[tid+s]){
				sdata[tid] = sdata[tid+s];
			}
		}
		__syncthreads();
	}
	printf("t/assadasdid = %d\n", tid);
	if (tid == 0){
		printf("OMG HET THI min = %d Id =  %d\n",sdata[0], i);
		idata[v][blockIdx.x] = sdata[0];
		
	}

}
int main (int argc, char* argv[]){
	int iQnt, i, j, **m, devID = 0, **mv, min;
	char ignore[100];
	cudaError_t error;
	cudaDeviceProp deviceProp;
	FILE* f;

	if (argc != 2){
		printf("Usage: %s <caminho_lista_matriz>\n", argv[0]);
		return 2;
	}



	printf("3x3 matrix reduction using CUDA\n");

	error = cudaGetDevice(&devID);
	if (error != cudaSuccess){
		printf("cudaGetDevice returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
	}

	error = cudaGetDeviceProperties(&deviceProp,devID);
	if (deviceProp.computeMode == cudaComputeModeProhibited){
		fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
		exit(EXIT_SUCCESS);
	}
	if (error != cudaSuccess){
		printf("cudaGetDeviceProperties returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
	}
	else{
		printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
	}

	
	f = fopen(argv[1], "r");

	/* Read amount of matrices */
	fscanf(f, "%d", &iQnt);

	/* Allocate matrix on host*/
	m = (int**) malloc(sizeof(int*)*SIZE*SIZE);
	for (i = 0; i < SIZE*SIZE; ++i){
		m[i] = (int*)malloc(sizeof(int)*iQnt);
	}
	/* Read user input */
	for (i = 0; i < iQnt; ++i){
		/*throw first line*/
		fgets(ignore, 100, f);
		for (j = 0; j < SIZE; ++j){
			fscanf(f, "%d %d %d \n", &m[0+j*3][i], &m[1+j*3][i], &m[2+j*3][i]);
		}
	}

	/* Allocate matrix on device */
	mv = (int**) malloc(sizeof(int*)*SIZE*SIZE);
	for (i = 0; i < SIZE*SIZE; ++i){
		error = cudaMalloc(&mv[i], sizeof(int)*iQnt);
		if (error != cudaSuccess){
			printf("cudaMalloc returned error %s (code %d), line(%d) \n", cudaGetErrorString(error), error, __LINE__);
		}
		error = cudaMemcpy(mv[i], m[i], sizeof(int)*iQnt, cudaMemcpyHostToDevice);
		if (error != cudaSuccess){
			printf("cudaMemcpy returned error %s (code %d), line(%d) \n", cudaGetErrorString(error), error, __LINE__);
		}
	}
	/* Begin reduction */

	printf("iQnt = %d\n", iQnt);
	for (i = iQnt; i > 1; i = 1+(i/BLOCKSIZE)){
		printf("Size of reduction = %d\n", i);
		dim3 blocks((1+(i/BLOCKSIZE)), SIZE*SIZE);
		printf("BLOCK CALL (%d, %d)\n", (1+(i/BLOCKSIZE)), SIZE*SIZE);
		minvec<<<blocks, BLOCKSIZE>>>(mv, i);
		
	}
	sleep(10);
	error = cudaDeviceSynchronize();
	if (error != cudaSuccess){
		printf("cudaDeviceSynchronize returned error %s (code %d), line(%d) \n", cudaGetErrorString(error), error, __LINE__);
	}

	for (i = 0; i < SIZE*SIZE; ++i){
		error = cudaMemcpy(&min, &mv[i][0], sizeof(int), cudaMemcpyDeviceToHost);
		if (error != cudaSuccess){
			printf("cudaMemcpy returned error %s (code %d), line(%d) \n", cudaGetErrorString(error), error, __LINE__);
		}
		printf("The min for %d is %d\n", i, min);
	}

	free(m);
	cudaFree(mv);


	return 0;

}