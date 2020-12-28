#ifndef LBP_FUNCTIONS 
#define LBP_FUNCTIONS
#include <stdint.h>

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
__global__ void euclideanDistance_kernel(int N, int M, int A, const float* __restrict__ dev_train, const float* __restrict__ dev_test, float* __restrict__ dev_distances);

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
__global__ void sort_kernel(int N, int M, int K, float* __restrict__ dev_distances, int* __restrict__ dev_labels);

#endif

