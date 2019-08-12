#include <stdio.h>
#include <assert.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <math.h>
#define SIZE 3
#define BLOCKSIZE 512
#define PI 3.1415926535897932384626433


extern "C" void hello_world(){
	printf("oi, fui importada com sucesso. \n");

}

extern "C" void randomize(float *dV, int size){
	curandGenerator_t prng;
	/* create generator*/
	curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_XORWOW);
	/* generate seed */
	curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) 1337);
	/* randomize */
	curandGenerateUniform(prng, dV, size);
}



extern "C" __global__ void applyfunction(float *dV, float *dV2, int size, float k, float M){	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < size){
		dV[i] = ((float)sin((2*M + 1)*PI*dV[i])*cos(2*PI*k*dV[i]))/sin(PI*dV[i]);
		dV2[i] = dV[i]*dV[i];
	}
}

extern "C" __global__ void sumvec(float *idata, int size){
	__shared__ float sdata[BLOCKSIZE];
	int s;
	int tid = threadIdx.x;
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int pseudoIf = i < size;
	/*
	if (blockIdx.x == 1 && threadIdx.x == 0){
		printf("i = %d; size = %d; pseudoIf = %d\n", i, size, pseudoIf);
	}*/

	sdata[tid] = pseudoIf*idata[i];
	/*
	__syncthreads();
	if (tid == 0){
		for (s = 0; s < size; s++){
			printf("id = %d sdata[%d] = %f; %f\n", blockIdx.x, s, sdata[s], idata[blockIdx.x*blockDim.x + s]);
		}
	}*/
	
	__syncthreads();
	for (s = blockDim.x/2; s > 0; s >>= 1){
		if (tid < s){
			sdata[tid] = sdata[tid] + sdata[tid+s];
		}
		__syncthreads();
	}
	if (tid == 0){
		printf("id = %d, stored = %f\n", blockIdx.x, sdata[0]);
		idata[blockIdx.x] = sdata[0];
	}

}

extern "C" float* MC_CUDA(int N, float k, float M){

	int i, devID = 0;
	float *dV, *dV2, f, f2;
	cudaError_t error;
	cudaDeviceProp deviceProp;
	static float resultados[2];


	/*CUDA boring stuff */
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

	/* Allocate array on device */
	error = cudaMalloc(&dV, sizeof(float)*N);
	if (error != cudaSuccess){
		printf("cudaMalloc returned error %s (code %d), line(%d) \n", cudaGetErrorString(error), error, __LINE__);
	}
	error = cudaMalloc(&dV2, sizeof(float)*N);
	if (error != cudaSuccess){
		printf("cudaMalloc returned error %s (code %d), line(%d) \n", cudaGetErrorString(error), error, __LINE__);
	}

	/* Generate array */
	randomize(dV, N);
	/* Apply function */
	applyfunction<<<(1 + (N/BLOCKSIZE)), BLOCKSIZE>>>(dV, dV2, N, k, M);
	/* Sum all values */
	for (i = N; i > 1; i = 1+(i/BLOCKSIZE)){
		printf("Number of blocks = %d\n", 1+(i/BLOCKSIZE));
		printf("Size of array = %d\n", i);
		sumvec<<<(1+(i/BLOCKSIZE)), BLOCKSIZE>>>(dV, i);
		error = cudaDeviceSynchronize();
		if (error != cudaSuccess){
			printf("cudaDeviceSynchronize returned error %s (code %d), line(%d) \n", cudaGetErrorString(error), error, __LINE__);
		}
		printf("WAIT!\n");
		sumvec<<<(1+(i/BLOCKSIZE)), BLOCKSIZE>>>(dV2, i);
		error = cudaDeviceSynchronize();
		if (error != cudaSuccess){
			printf("cudaDeviceSynchronize returned error %s (code %d), line(%d) \n", cudaGetErrorString(error), error, __LINE__);
		}
	}
	/* Copy values from device */
	error = cudaMemcpy(&f, &dV[0], sizeof(float), cudaMemcpyDeviceToHost);
	if (error != cudaSuccess){
		printf("cudaMemcpy returned error %s (code %d), line(%d) \n", cudaGetErrorString(error), error, __LINE__);
	}
	error = cudaMemcpy(&f2, &dV2[0], sizeof(float), cudaMemcpyDeviceToHost);
	if (error != cudaSuccess){
		printf("cudaMemcpy returned error %s (code %d), line(%d) \n", cudaGetErrorString(error), error, __LINE__);
	}
	/* Calculate results */
	printf("SOMA = %f\n", f);
	f /= N;
	f2 /= N;

	resultados[0] = f;
	resultados[1] = f2;
	cudaFree(dV);
	cudaFree(dV2);
	return resultados;
}