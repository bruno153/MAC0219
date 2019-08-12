#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <unistd.h>
#define SIZE 3
#define BLOCKSIZE 512
__global__ void minvec(int *idata, int size){
	__shared__ int sdata[BLOCKSIZE];
	int s;
	int tid = threadIdx.x;
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int pseudoIf = i < size;
	sdata[tid] = pseudoIf*idata[i] + (1 - pseudoIf)*idata[0];
	
	
	__syncthreads();
	for (s = blockDim.x/2; s > 0; s >>= 1){
		if (tid < s){
			pseudoIf = sdata[tid] < sdata[tid+s];
			sdata[tid] = pseudoIf*sdata[tid] + (1 - pseudoIf)*sdata[tid+s];
			
		}
		__syncthreads();
	}
	if (tid == 0){
		idata[blockIdx.x] = sdata[0];
		
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
	fgets(ignore, 100, f);

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
	for (i = iQnt; i > 1; i = 1+(i/BLOCKSIZE)){
		for (j = 0; j < SIZE*SIZE; ++j){
			minvec<<<(1+(i/BLOCKSIZE)), BLOCKSIZE>>>(mv[j], i);
			error = cudaDeviceSynchronize();
		}
	}
	
	if (error != cudaSuccess){
		printf("cudaDeviceSynchronize returned error %s (code %d), line(%d) \n", cudaGetErrorString(error), error, __LINE__);
	}

	for (i = 0; i < SIZE*SIZE; ++i){
		error = cudaMemcpy(&min, &mv[i][0], sizeof(int), cudaMemcpyDeviceToHost);
		if (error != cudaSuccess){
			printf("cudaMemcpy returned error %s (code %d), line(%d) \n", cudaGetErrorString(error), error, __LINE__);
		}
		printf("%d ", min);
		if ((i+1)%3 == 0){
			printf("\n");
		}
	}

	free(m);
	cudaFree(mv);


	return 0;

}