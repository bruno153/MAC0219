#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>
#define SIZE 3
#define BLOCKSIZE 512
__global__ void minvec(int *idata, int size){
	__shared__ int sdata[BLOCKSIZE];
	int s;
	int tid = threadIdx.x;
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int pseudoIf = i < size;
	/*
	if (i < size)
		sdata[tid] = idata[i];
	else{
		sdata[tid] = idata[0];
	}*/

	sdata[tid] = pseudoIf * idata[i] + (1-pseudoIf) * idata[0];

	__syncthreads();

	/*printf(" sdata = %d\n", sdata[tid]);*/

	for (s = blockDim.x/2; s > 0; s >>= 1){
		if (tid < s){
			if (sdata[tid] > sdata[tid+s]){
				sdata[tid] = sdata[tid+s];
			}
		}
		__syncthreads();
	}
	if (tid == 0){
		idata[blockIdx.x] = sdata[0];
		/*printf("min = %d Id =  %d\n",sdata[0], i);*/
	}
}

int main (int argc, char* argv[]){
	int iQnt, i, *m, devID = 0, *mv, min;
	char ignore[100];
	cudaError_t error;
	cudaDeviceProp deviceProp;

	FILE* f;

	if (argc != 2){
		printf("Usage: %s <caminho_lista>\n", argv[0]);
		return 2;
	}

	/*printf("vector reduction using CUDA\n");*/

	/* Boring CUDA stuff */
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



	/* Open file */
	f = fopen(argv[1], "r");

	/* Read amount of matrices */
	fscanf(f, "%d", &iQnt);

	/* Allocate matrix on host*/
	m = (int *) malloc(sizeof(int)*iQnt);

	/* Allocate matrix on device */
	error = cudaMalloc(&mv, sizeof(int)*iQnt);
	if (error != cudaSuccess){
		printf("cudaMalloc returned error %s (code %d), line(%d) \n", cudaGetErrorString(error), error, __LINE__);
	}
	fgets(ignore, 100, f);
	/* Get values from file */
	for (i = 0; i < iQnt; ++i){
		/*throw first line*/
		fgets(ignore, 100, f);
		fscanf(f, "%d \n", &m[i]);
		/*printf("%d\n", m[i]);*/
	}

	/*printf("m[0] = %d\n", m[0]);*/
	error = cudaMemcpy(mv, m, sizeof(int)*iQnt, cudaMemcpyHostToDevice);
	if (error != cudaSuccess){
		printf("cudaMemcpy returned error %s (code %d), line(%d) \n", cudaGetErrorString(error), error, __LINE__);
	}

	for (i = iQnt; i > 1; i = 1+(i/BLOCKSIZE)){
		printf("Size of reduction = %d\n", i);
		minvec<<<(1+(iQnt/BLOCKSIZE)), BLOCKSIZE>>>(mv, i);
		cudaDeviceSynchronize();
	}

	

	/*error = cudaMemcpy(m, mv, (sizeof(int)*(1+(iQnt/BLOCKSIZE))), cudaMemcpyDeviceToHost);*/
	error = cudaMemcpy(&min, &mv[0], sizeof(int), cudaMemcpyDeviceToHost);
	if (error != cudaSuccess){
		printf("cudaMemcpy returned error %s (code %d), line(%d) \n", cudaGetErrorString(error), error, __LINE__);
	}

	printf("CABO! - THE MIN IS %d \n", min);

	free(m);
	cudaFree(mv);
}