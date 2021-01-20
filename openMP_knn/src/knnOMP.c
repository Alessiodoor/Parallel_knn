#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "datasetFunctions.h"

// attributi
#define A 30
// labels
#define LABELS 10

/*
Funzione per calcolare la distanza euclidea tra gli attributi di un sample di train e uno di test
Parametri:
train: sample di train
test: sample di test
*/
float euclideanDistance(float* train, float* test){
    float sum = 0.f;

    for (int d = 0; d < A; ++d) {
        const float diff = train[d] - test[d];
        sum += diff * diff;
    }
    return sqrtf(sum);
}

/*
Funzione per ordinare ordinare i sample di train per il test i-esimo in base alla distanza tra train e test
Questa funzione ordina i vicini di un solo sample di test.
Parametri:
distances: array delle distanze tra il sample di test i-esimo e tutti i sample di train
index: array degli indici di tutti i sample di train per il sample id tst i-esimo
K: numero di vicini
N: numero di sample di train
*/
void sort(float * distance, int * index, int K, int N){
    index[0] = 0;
    for(int i =1; i < N; ++i){
        float distanzaCorrente = distance[i];
        int indiceCorrente = index[i];

        if( i >= K && distanzaCorrente >= distance[K-1]){
            continue;
        }

        int j = i;
        if (j > K-1)
            j = K-1;
        
        while(j > 0 && distance[j-1] > distanzaCorrente){
            distance[j] = distance[j-1];
            index[j] = index[j-1];
            --j;
        }

        distance[j] = distanzaCorrente;
        index[j] = indiceCorrente;
    }
}

/*
Funzione principale dell'algoritmo Knn parallelizzato.
Procedimento:
Vengono allocati gli array da utilizzare
Inizio una regione parallela dove utilizzo le variabili precedentemente allocate
Parallelizzo tra i vari treads l'esecuzione di due for innestati, dove per ogni sample di test, calcolo le distanze euclidea con i sample di train
Suvvessivamente, eseguo un altro for in parallelo, dov per ogni sample di test, ordino i primi k sample di train in base alla distanza calcolata precedentemente
A questo punto ho terminato l'esecuzione parallela.
Per ogni sample di test individuo la bestLabel, ovvero quella con il maggior numero di  occorrenze tra i vicini
Infine, verifico che la bestLabel trovata corrisponda con la label del sample di test e aggiorno la matrice di cofusione

Parametri:
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
	double start){

	//distanza da ogni punto del training (dopo ordinamento i primi k saranno i vicini)
   	float* k_distances = (float*) malloc(M * N *sizeof(float)); 
    
   	//indice della label del sample di train (dopo ordinamento i primi k saranno le etichette dei K vicini)
   	int * k_labels = (int*) malloc(M * N * sizeof(int));

   	// matrice di confusione
   	int* confusionMatrix = (int*) malloc(sizeof(int)* LABELS * LABELS);
   
   	//label di ogni sample per majority voting
   	int* countsLabel = (int*) malloc(sizeof(int)* LABELS);

   	//controllo la memoria allocata
   	if ( k_distances == NULL || k_labels == NULL || confusionMatrix == NULL || countsLabel == NULL ){
      	printf("Memoria insufficiente per assegnare le variabili di knn\n");
      	exit(EXIT_FAILURE);
   	}

   	// inizializzo la matrice di confusione con tutti 0
   	for(int i = 0; i < (LABELS * LABELS); i++){
      	confusionMatrix[i] = 0;
   	}

   	//printf("Knn start...\n");
	//numero di errori compessi dall'algoritmo KNN
   	int error = 0;

   	// imposto il numero di tread che eseguiranno la sezione parallela
	omp_set_num_threads(NT);

	// avvio una regione parallela dove essegno l'esecuzione dei cicli for ai treads
	#pragma omp parallel default(none), shared(start, trainingData, testingData, classesTesting, classesTraining, k_distances, k_labels)
    {
		//per ogni sample del test
		#pragma omp for collapse(2) schedule(guided, 1)
		//per ogni sample del test
		for(int i = 0; i < M; i++){
			//calcolo la distanza euclidea con tutti i punti del train
			for(int j = 0; j < N; j++){
				// calcolo distanza euclidea tra singolo train e test
				k_distances[i*M + j] = euclideanDistance(&trainingData[j * A], &testingData[i * A]);
				// salvo la classe del train nella cella i, j
				// i corrisponde all'i-esimo sample di test
				// j corrisponde all'j-esimo sample di train
				k_labels[i*M + j] = classesTraining[j];
			}
		}
		
		
		//ordino ogni distanza in base alle distanze calcolate precedentemente 
		#pragma omp for schedule(guided, 1)
		for(int i = 0; i < N; i++){
			sort(&k_distances[i * M], &k_labels[i * M], K, N);
		}
	}
	// fine della regione parallela

	// eseguo un altro ciclo for per calcolare la migliore label di ogni sample di teste per costruire la matrice di confusione
	for (int i = 0; i < M; ++i)
	{
		//inizializza a zero il vettore contenente la migliore label per ogni sample di test
		for(int j = 0; j < LABELS; j++){
		   	countsLabel[j] = 0;
		}
		int bestLabel = 0;

		// per ogni sample di test individuo la label migliore tra quelle dei k vicini
		//per i primi k vicini
		for(int j = 0; j < K; j++){	
			// salvo la label del train per il sample di test corrente
			int label = k_labels[i*M +j];
			
			// incremento il contatore che identifica il numero di istanze della label
			countsLabel[label] = countsLabel[label] + 1;

			// se la labelc compare più volte della bestlabel, diventa la nuova migliore
			if(countsLabel[label] > countsLabel[bestLabel])
				bestLabel = label;
		}

		int realLabel = classesTesting[i];

		// verifico che la label assegnata al sample di test sia corretta, confrontandola con quella reale
		if (realLabel != bestLabel)
			error = error + 1;
		
		//aggiorno la matrice di confusione
		confusionMatrix[realLabel * LABELS + bestLabel] = confusionMatrix[realLabel * LABELS + bestLabel] + 1;	
	}

	//printConfusionMatrix(confusionMatrix, LABELS);

	// libero la memoria utilizzata 
   	free(confusionMatrix); confusionMatrix = NULL;
   	free(countsLabel); countsLabel = NULL;
   	free(k_distances); k_distances = NULL;
   	free(k_labels); k_labels = NULL;
}
