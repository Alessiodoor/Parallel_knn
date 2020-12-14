#ifndef KNNOMP_H
#define KNNOMP_H 

/*
Funzione per calcolare la distanza euclidea tra gli attributi di un sample di train e uno di test
Parametri:
train: sample di train
test: sample di test
*/
float euclideanDistance(float* train, float* test);

/*
Funzione principale dell'algoritmo Knn.
Procedimento:
Vengono allocati gli array da utilizzare
Per ogni sample di test, calcolo le distanze euclidea con i sample di train
Per ogni sample di test, ordino i primi k sample di train in base alla distanza calcolata precedentemente
Per ogni sample di test individuo la bestLabel, ovvero quella con il maggior numero di  occorrenze 
tra i vicini
Infine verifico che la bestLabel trovata corrisponda con la label del sample di test e aggiorno la matrice di cofusione

Parametri:
NT: numero di treads
trainingData: samples di train
testingData: samples di test
classesTraining: labels dei sample di train
classesTesting: labels dei sample di test (usati per la verifica)
K: numero di vicini
N: numero sample di trian
M: numero sample di test
*/
void knn(
	int NT,
	float* trainingData, 
	float* testingData,
	uint8_t* classesTraining, 
	uint8_t* classesTesting, 
	int K, 
	int N,
	int M,
	double start);

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

#endif