linka_tudo: gpu.o ep3.o
	mpicc -I /usr/local/cuda-9.1/include -L /usr/local/cuda-9.1/lib64 -lcudart -lcurand -o pf_funfa gpu.o ep3.o  -lm -fopenmp -lstdc++

gpu.o: gpu.cu
	nvcc -c gpu.cu -lcurand -lm

ep3.o: ep3.c header.h
	mpicc -c ep3.c -lm -fopenmp
