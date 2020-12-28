#include "knnCuda.h" 
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

/*
Funzione kernel per calcolare la distanza euclidea tra gli attributi di un sample di train e uno di test
Parametri:
N: numero sample di train
M: numero sample di test
A: numero di attributi
dev_train: sample di train sul device
dev_test: sample di testsul device
dev_distances: matrice delle distanze sul device
*/
__global__ void euclideanDistance_kernel(int N, int M, int A, const float* __restrict__ dev_train, const float* __restrict__ dev_test, float* __restrict__ dev_distances){//, int* dev_labels){
	
	// indice di inizio della riga
	int idx = threadIdx.x+blockDim.x*blockIdx.x;
  	int idy = threadIdx.y+blockDim.y*blockIdx.y;
	
	// controllo che gli indici del thread siano corretti
	if(idx < N && idy < M){
		float sum = 0.f;
		// rendo parallela l'esecuzione del for, tra i threads del blocco
	    #pragma unroll
	    for (int d = 0; d < A; ++d) {
	    	float x = dev_train[idx*A +d];  
	    	float y = dev_test[idy*A +d];
	        float diff = x - y;
	        sum += diff * diff;
	    }
		dev_distances[(idy * N) + idx] = sqrtf(sum);
	}
}

/*
Funzione kernel per ordinare ordinare i sample di train per il test i-esimo in base alla distanza tra train e test
Questa funzione ordina i vicini di un solo sample di test.
Parametri:
distances: array delle distanze tra il sample di test i-esimo e tutti i sample di train
index: array degli indici di tutti i sample di train per il sample id tst i-esimo
N: numero di sample di train
M: numero sample di test
K: numero di vicini
dev_distances: matrice delle distanze sul device
dev_labels: label del sample di test 
*/
__global__ void sort_kernel(int N, int M, int K, float* __restrict__ dev_distances, int* __restrict__ dev_labels){
	
	// indice di inizio della riga
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	// controllo che l'indici sia corretto
	if(index < M){
		dev_labels[index * K] = 0;
		// rendo parallela l'esecuzione del for, tra i threads del blocco
		#pragma unroll
		for(int i=1; i< N; i++){
			float distanzaCorrente = dev_distances[index*N+i];
        	int indiceCorrente = i;
        	if( i >= K && distanzaCorrente >= dev_distances[index*N+ K-1]){
            	continue;
        	}
			
			int j = i;
        	if (j > K-1)
            	j = K-1;
        
        	while(j > 0 && dev_distances[index*N+ j-1] > distanzaCorrente){
            	dev_distances[index*N +j] = dev_distances[index*N+j-1];
            	dev_labels[index*K+j] = dev_labels[index*K+j-1];
            	--j;
        	}

        	dev_distances[index*N+j] = distanzaCorrente;
        	dev_labels[index*K+j] = indiceCorrente;	
		}
	}
}