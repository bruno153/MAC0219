#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>
#include <assert.h>





#include <unistd.h>

#include "header.h"

#define SIZE 3
#define BLOCKSIZE 512
#define PI 3.1415926535897932384626433





float function(float x, float k, float M){
	return ((float)sin((2*M + 1)*PI*x)*cos(2*PI*k*x))/sin(PI*x);
}

void seqIntegrate(int N, float k, float M, float* result, float* error){
	int i;
	double f = 0, f2 = 0, tmp, x;

	srand(1055);

	for (i = 0; i < N; ++i){
		x = ((double)rand()/(RAND_MAX));

		tmp = function(x/2, k, M);
		/*printf("%f %f\n", tmp, x);*/
		f += tmp;
		f2 += tmp*tmp;
	}

	f /= N;
	f2 /= N;

	*result = (float)f;
	*error = (float)sqrt((f2 - (f*f))/N);

}


void seqIntegrate_parallel(int N, float k, float M, float* result, float* error){
	int i;
	double f = 0, f2 = 0, x;

	srand(1055);

	#pragma omp parallel for private(x) reduction(+:f) reduction(+:f2)
	for (i = 0; i < N; ++i){
		#pragma omp critical
		x = ((float)rand()/(RAND_MAX));

		double tmp = function(x/2, k, M);
		/*printf("%f %f\n", tmp, x);*/
		f += tmp;
		f2 += tmp*tmp;
		
		
	}

	f /= N;
	f2 /= N;

	*result = (float)f;
	*error = (float) sqrt((f2 - (f*f))/N);

}



int main (int argc, char* argv[]){
	float result, error_seq, error_par, error_balanced, error_cuda;
	static double time, t0,t1,t2,t3;
	int cont;
	int i_loadbalanced;

	/* 3 vetores de tempo, um pra cpu seq, outro pra cpu 
	paralelizada, outro pra gpu. Todos com valores 
	correspondentes a 25 50 75 100% da carga. */
	float  time_parallel[4], time_cuda[4]; //time_seq[4],
	float candidatos_loadbalance[5];
	float tempo_inicial, tempo_final;
	float menor_tempo;

	float *funfa;

	int i, devID = 0, N, N_iteracoes;
	float *dV, M, k, *dV2, f, f2;
	
	if (argc != 4){
		printf("Usage: %s <N> <k> <M>\n", argv[0]);
		return 3;
	}

	/* get parameters */
	N = atoi(argv[1]);
	k = atof(argv[2]);
	M = atof(argv[3]);
	

 	MPI_Init(NULL, NULL);

	/* Get the number of processes*/
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    /* Get the rank of the process*/
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    if (rank == 0)
    {

    	printf("oi sou o rank/processo 0, vou comecar \n");
    	N_iteracoes = N;

    	

		cont=1;
		while(cont<=4){
			tempo_inicial = omp_get_wtime();
			
			funfa = MC_CUDA(  (N_iteracoes/4)*(cont) ,k,M);


			f = funfa[0];
			f2 = funfa[1];
			error_cuda = sqrt((f2 - (f*f))/ (N_iteracoes/4)*(cont) );

			/* Record time */
			time_cuda[cont-1] = omp_get_wtime() - tempo_inicial;

			cont +=1;
		}
		printf("rank 0 fez todas as 4 iteracoes do codigo em cuda \n");

		MPI_Recv(&time_parallel, 4, MPI_FLOAT, 1, MPI_ANY_TAG,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

		printf("rank 0 acabou de receber os tempos do rank1 \n");
		/*printf("rank 0 feito!\n");*/

		/* Load Balancing */

		candidatos_loadbalance[0] = time_cuda[3];                      // 0% CPU  100% CUDA
		candidatos_loadbalance[1] = time_cuda[2] + time_parallel[0];   // 25% CPU  75%CUDA
		candidatos_loadbalance[2] = time_cuda[1] + time_parallel[1];   // 50% CPU 50% CUDA
		candidatos_loadbalance[3] = time_cuda[0] + time_parallel[2];   // 75% CPU 25% CUDA
		candidatos_loadbalance[4] =                time_parallel[3];   // 100% CPU 0% CUDA


		cont=0;
		
		menor_tempo=candidatos_loadbalance[0];
		i_loadbalanced = 0;
		while (cont < 5){
			if (menor_tempo > candidatos_loadbalance[cont]){
				i_loadbalanced = cont;
				menor_tempo = candidatos_loadbalance[cont];
			}

			cont+=1;
		}
		printf("rank 0 sabe qual é o loadbalanced otimo  e vai enviar instrucoes pro rank 1 \n");
		/*ok agora eu sei como eu tenho q fazer na versao load balanced
		agora preciso avisar o outro processo como vamos fazer */



		MPI_Send(&i_loadbalanced, 1, MPI_INT, 1, MPI_ANY_TAG,MPI_COMM_WORLD);


		if (i_loadbalanced != 4){

			cont = 4 - i_loadbalanced;

			funfa = MC_CUDA(  (N_iteracoes/4)*(cont) ,k,M);
			f = funfa[0];
			f2 = funfa[1];
			error_cuda = sqrt((f2 - (f*f))/ (N_iteracoes/4)*(cont) );

		}
		printf("enviei a mensagem e executei minha parte do load balance, n tem mais codigo pra fazer\n");

		/*MPI_Barrier(MPI_COMM_WORLD);*/


		/*escrever um MPI receive pra receber a parte q foi distribuida no outro processo e juntar tudo
		imprimir resultados e acabou*/
    	




    }

    else if(rank==1){
    	printf(" rank 1 comecando  \n");
    	
    	N_iteracoes = N;

    	/*sequencial:
    	cont=1;
    	while(cont<=4){
    		
    		tempo_inicial = omp_get_wtime();
    		seqIntegrate( (N_iteracoes)*(cont/4) , k ,M, &result, &error_seq);
    		time_seq[cont-1]= omp_get_wtime();
    		cont+=1;
    		
    	}*/


    	cont=1;
    	while(cont<=4){
    		tempo_inicial = omp_get_wtime();
    		seqIntegrate_parallel( (N_iteracoes)*(cont/4) ,k,M,&result, &error_par);
    		time_parallel[cont-1]= omp_get_wtime();
    		cont+=1;
    	}
    	printf("rank 1 feito, fiz todas minhas iteracoese e vou avisar o rank 0 os tempos!\n");

    	
    	MPI_Send(&time_parallel, 4, MPI_FLOAT, 0, MPI_ANY_TAG, MPI_COMM_WORLD);
      /*MPI_Send( o q to enviando, tamanho,tipo, pra quem, tag, canal/comunicador);*/

    	/*MPI_Barrier(MPI_COMM_WORLD); // sepa que n precisa disso, da pra se virar so com os send e receive*/
    	printf("rank 1 enviou com sucesso e agora eu vou esperar novas instrucoes \n");
    	MPI_Recv(&i_loadbalanced,1,MPI_INT,0,MPI_ANY_TAG,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    	printf(" rank 1 recebeu novas instrucoes e vai comecar a executar elas \n");

    	if( i_loadbalanced !=0 ){
    		seqIntegrate_parallel( (N_iteracoes)*(i_loadbalanced/4) ,k,M,&result, &error_par); //precisa mudar esse error_par pra algo, acho

    	}
    	printf("rank 1 fez sua parte do load balanced, acabou o codigo\n");
        /* fazer mais um MPI_Send para enviar de volta os resultado e juntar tudo no master, mas isso depende de como vai ser essa parte de juntar*/





    }
    else{
    	printf("  tem mais de 2 processos, pode dar ruim, espero q n \n");
    	MPI_Barrier(MPI_COMM_WORLD);


    }







	 MPI_Finalize();
	
	return 0;
}













