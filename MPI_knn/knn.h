#ifndef KNN_MPI_H
#define KNN_MPI_H

/*

*/
void splitData(
	int* countsAttr, 
	int* countsClasses, 
	int* adressesAttr, 
	int* adressesClasses, 
	int comm_size, 
	int nAttr, 
	int testSize);

/*
Funzione per ordinare ordinare i sample di train per il test i-esimo in base alla distanza tra train e test
Questa funzione ordina i vicini di un solo sample di test.
Parametri:
distances: array delle distanze tra il sample di test i-esimo e tutti i sample di train
index: array degli indici di tutti i sample di train per il sample id tst i-esimo
K: numero di vicini
N: numero di sample di train
*/
void sort(
	float * distance, 
	int * index, 
	int K, 
	int N);

/*
Funzione per calcolare la distanza euclidea tra gli attributi di un sample di train e uno di test
Parametri:
train: sample di train
test: sample di test
*/
float euclideanDistance(
	float* train, 
	float* test,
	int A);

int localKnn(
	float* trainData, 
	float* testData, 
	uint8_t* trainClass, 
	uint8_t* testClass, 
	int testSize, 
	int* confusionMatrix,
	int N,
	int LABELS,
	int A,
	int K,
	int rank,
	float* k_distances, 
	int* k_labels, 
	int* countsLabel);

/*

*/
void knn(
	const char* trainFile,
	const char* testFile,
	int K, 
	int N,
	int M,
	int A,
	int LABELS,
	int size,
	int rank);
			

#endif 
