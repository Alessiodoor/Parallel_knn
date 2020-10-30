#include <stdint.h>
#include <stdio.h> 
#include <stdlib.h>
#include <math.h>

#include "mpi.h"
#include "datasetFunctions.h"

/*

*/
void splitData(
	int* countsAttr, 
	int* countsClasses, 
	int* adressesAttr, 
	int* adressesClasses, 
	int comm_size, 
	int nAttr, 
	int testSize){
    // calcolo il resto della divisione tra il numero di processi e la dimensione del test
    int extraRows = testSize % comm_size;  
    // righe già assegnate, da utilizzare per salvare l'indirizzo della porzione di test successiva
    int sumAttr = 0;   
    int sumClasses = 0;                      

    // per ogni processo:
    for (int i = 0; i < comm_size; i++) {            
    	// assegno il numero di attributi del test per ogni processo, ovvero il numero di righe moltiplicato per il numero di attributi
        countsAttr[i] = (testSize / comm_size) * nAttr;    
        // assegno il numero di classi del test per ogni processo, una classe per ogni riga
        countsClasses[i] = (testSize / comm_size);    

        // se la divisione delle righe di tesst ha il resto
        // assegno le righe extra in ordine così ho le celle di memoria contigue
        if (extraRows > 0) {              
        	// aggiungo al processo gli elementi di una riga in eccesso        
            countsAttr[i]= countsAttr[i] + nAttr;   
            // aggiungo al processo la classe relativa alla riga in eccesso
            countsClasses[i] = countsClasses[i] + 1; 

            // sottraggo la riga che ho aggiunto 
            extraRows--;
        }

        // assegno l'indirizzo di inizio della porzione che dovrà elaborare ogni processo, sia per gli attributi che per le classi
        adressesAttr[i] = sumAttr;
        adressesClasses[i] = sumAttr;
        // mi sposto all'indirizzo successivo, sia per attributi che per classi
        sumAttr += countsAttr[i];
        sumClasses += countsClasses[i];
    }
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
void sort(float * distance, uint8_t * index, int K, int N){
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
        
        while(j > 0 && distance[j - 1] > distanzaCorrente){
            distance[j] = distance[j - 1];
            index[j] = index[j - 1];
            --j;
        }
        
        distance[j] = distanzaCorrente;
        index[j] = indiceCorrente;
    }
}

/*
Funzione per calcolare la distanza euclidea tra gli attributi di un sample di train e uno di test
Parametri:
train: sample di train
test: sample di test
*/
float euclideanDistance(float* train, float* test, int A){
    float sum = 0.f;

    for (int d = 0; d < A; ++d) {
        const float diff = train[d] - test[d];
        sum += diff * diff;
    }
    return sqrtf(sum);
}

/*

*/
int localKnn(float* trainData, float* testData, uint8_t* trainClass, uint8_t* testClass, int testSize, int* confusionMatrix, int N, int LABELS, int A, int K, int rank){
	printf("Start local knn on process %d\n", rank);

	//distanza da ogni punto del training (dopo ordinamento i primi k saranno i vicini) per il sample di test attuale
	// ad ogni iterazione del for pricnipale verrà riassegnato
   	float* k_distances = (float*) malloc(N * sizeof(float)); 
    
   	//indice della label del sample di train (dopo ordinamento i primi k saranno le etichette dei K vicini), per il sample di test attuale
   	// riassegnato ad ogni iterazione del for principale 
   	uint8_t* k_labels = (uint8_t*) malloc(N * sizeof(uint8_t));

   	//label di ogni sample per majority voting
   	int* countsLabel = (int*) malloc(sizeof(int) * LABELS);

   	//controllo la memoria allocata
   	if ( k_distances == NULL || k_labels == NULL || countsLabel == NULL ){
      	printf("Memoria insufficiente\n");
      	exit(EXIT_FAILURE);
   	}

   	// inizializzo la matrice di confusione con tutti 0
   	for(int i = 0; i < (LABELS * LABELS); i++){
      	confusionMatrix[i] = 0;
   	}

	//numero di errori compessi dall'algoritmo KNN
   	int error = 0;

   	//per ogni sample della porzione di test assegnata al processo
	for(int i = 0; i < testSize; i++){
		//calcolo la distanza euclidea con tutti i punti del train
		for(int j = 0; j < N; j++){
			// calcolo distanza euclidea tra singolo train e test
			k_distances[j] = euclideanDistance(&trainData[j * A], &testData[i * A], A);
			// salvo la classe del train nella cella j-esima
			// j corrisponde all'j-esimo sample di train
			k_labels[j] = trainClass[j];
		}

		//ordino i dati in base alle distanze calcolate precedentemente 
		sort(k_distances, k_labels, K, N);

		//inizializza a zero il vettore contenente la migliore label per ogni sample di test
		for(int i = 0; i < LABELS; i++){
		   	countsLabel[i] = 0;
		}
		int bestLabel = 0;

		// per ogni sample di test individuo la label migliore tra quelle dei k vicini
		//per i primi k vicini
		for(int j = 0; j < K; j++){	
			// salvo la label del train per il sample di test corrente
			int label = k_labels[j];
			
			// incremento il contatore che identifica il numero di istanze della label
			countsLabel[label] = countsLabel[label] + 1;

			// se la labelc compare più volte della bestlabel, diventa la nuova migliore
			if(countsLabel[label] > countsLabel[bestLabel])
				bestLabel = label;
		}

		int realLabel = testClass[i];

		// verifico che la label assegnata al sample di test sia corretta, confrontandola con quella reale
		if (realLabel != bestLabel)
			error = error + 1;
		
		//aggiorno la matrice di confusione
		confusionMatrix[realLabel * LABELS + bestLabel] = confusionMatrix[realLabel * LABELS + bestLabel] + 1;
	}

	printf("fine for processo %d\n", rank);

	// libero la memoria utilizzata
	free(countsLabel); countsLabel = NULL;
   	free(k_distances); k_distances = NULL;
   	free(k_labels); k_labels = NULL;
	
	printf("Memoria liberata\n");

   	return error;
}			

void knn(
	const char* trainFile,
	const char* testFile,
	int K, 
	int N,
	int M, 
	int A,
	int LABELS,
	int size, 
	int rank){

   	printf("Knn start on process: %d\n", rank);

	int rank_root = 0;
	// inizializzo le variabili che verranno utilizzate nel metodo scatterv per asseggnare le porzioni di test ad ogni processo

   	//numero di sample di test assegnati ad ogni processo 
    int countsAttr[size]; 
    //indirizzi di inizio della porzione di test assegnata ad ogni processo, relariva agli attributi
    int addressesAttr[size]; 

    //numero classi di test assegnate ad ogni processo, accoppiate con countsAttr
    int countsClasses[size]; 
    //indirizzi di inizio della porzione di test assegnata ad ogni processo, relativa alle classi
    int addressesClasses[size];

    // assegno il numero elementi della porzione di test  per ogni processi e l'indirizzo di inizio di ogni porzione 
   	splitData(countsAttr, countsClasses, addressesAttr, addressesClasses, size, A, M);

   	if(rank == rank_root) {
   		// in questo if sono presenti tutte le operazioni che vengono eseguite dal processo root
   		// inizializzo le matrici del dataset
		float * trainData = (float *) malloc(N * A * sizeof(float));
		float * testData = (float *) malloc(M * A * sizeof(float));

		uint8_t * trainClass = (uint8_t*) malloc(N *sizeof(uint8_t));
		uint8_t * testClass = (uint8_t*)  malloc(M *sizeof(uint8_t));

		if(trainData == NULL || testData == NULL || testClass == NULL || trainClass == NULL){
			printf("Memoria insufficiente!\n");
			exit(EXIT_FAILURE);
		}

		// leggo il dataset e lo salvo nelle variabili appena dichiarate
		readFile(trainFile, N, A, trainData, trainClass);
		readFile(testFile, M, A, testData, testClass);

		// inivio tutti i dati di train a tutti i processi
	    MPI_Bcast(trainData, N * A, MPI_FLOAT, rank_root, MPI_COMM_WORLD);
	    MPI_Bcast(trainClass, N, MPI_UINT8_T, rank_root, MPI_COMM_WORLD);

	    // creo una variabile che conterrà la porzione di test assegnata ad processo root
		float * localTestData = (float *) malloc(countsAttr[rank] * sizeof(float));
		uint8_t * localTestClass = (uint8_t*)  malloc(countsClasses[rank] * sizeof(uint8_t));	

		if(localTestData == NULL || localTestClass == NULL){
			printf("Memoria insufficiente!\n");
			exit(EXIT_FAILURE);
		}

		// Invio la porzione di test ad ogni processo, per quanto riguarda gli attributi e le classi
   		MPI_Scatterv(testData, countsAttr, addressesAttr, MPI_FLOAT, localTestData, countsAttr[rank_root], MPI_FLOAT, rank_root, MPI_COMM_WORLD);
   		MPI_Scatterv(testClass, countsClasses, addressesClasses, MPI_UINT8_T, localTestClass, countsClasses[rank_root], MPI_UINT8_T, rank_root, MPI_COMM_WORLD);

   		// creo la matrice di confusione locale del processo
   		int* confusionMatrixLocal = (int*) malloc(sizeof(int)* LABELS * LABELS);

   		if(confusionMatrixLocal == NULL){
			printf("Memoria insufficiente\n");
			exit(EXIT_FAILURE);
		}

		printf("Allocazione completata sul processo %d\n", rank);

		// calcolo knn locale, ovvero solo per la porzione di test assegnata al processo corrente
		int localError = localKnn(trainData, localTestData, trainClass, localTestClass, countsAttr[rank] / A, confusionMatrixLocal, N, LABELS, A, K, rank);
			
		MPI_Barrier(MPI_COMM_WORLD);

		// alloco la matrice di confusione totale
		int* confusionMatrixTotal = (int*) malloc(sizeof(int)* LABELS * LABELS);
		for(int i = 0; i < (LABELS * LABELS); i++){
	      	confusionMatrixTotal[i] = 0;
	   	}

		// eseguo una riduzione per construire la matrice di confusione totale unendo le matrici ottenute dall'esecuzione di ogni processo
		MPI_Reduce(confusionMatrixLocal, confusionMatrixTotal, LABELS*LABELS, MPI_INT, MPI_SUM, rank_root, MPI_COMM_WORLD);
		
		// ottengo gli errori sommando quelli di ogni processo
		int errors = 0;
		MPI_Reduce(&localError, &errors, 1 , MPI_INT, MPI_SUM, rank_root, MPI_COMM_WORLD);

		//MPI_Barrier(MPI_COMM_WORLD);

		
		printf("Reduce root\n");

		// dealloco memoria utilizzata
		free(trainData); trainData = NULL;
	    free(testData); testData = NULL;
	    free(trainClass); trainClass = NULL;
		free(testClass); testClass = NULL;
		free(localTestData); localTestData = NULL;
		free(localTestClass); localTestClass = NULL;
		free(confusionMatrixLocal); confusionMatrixLocal = NULL;
		free(confusionMatrixTotal); confusionMatrixTotal = NULL;

		writeResultJson(K, N, M, A, 0, "resultMPI.json");

		printf("Fine Root\n");

   	}else {
   		// operazioni per i processi non root

   		// alloco la memoria necessaria a contenere tutto il train set
   		float * trainData = (float *) malloc(N * M * sizeof(float));
 		uint8_t * trainClass = (uint8_t*) malloc(N *sizeof(uint8_t));

		// alloco la memoria necessasia a contenere la porzione di test set utilizzata dal processo
		float * localTestData = (float *) malloc(countsAttr[rank] * sizeof(float));
		uint8_t * localTestClass = (uint8_t*)  malloc(countsClasses[rank] * sizeof(uint8_t));

		if(trainData == NULL || localTestData == NULL || localTestClass == NULL || trainClass == NULL){
			printf("Memoria insufficiente!\n");
			exit(EXIT_FAILURE);
		}	

		// creo la matrice di confusione locale del processo 
   		int* confusionMatrixLocal = (int*) malloc(sizeof(int)* LABELS * LABELS);
   		
   		if(confusionMatrixLocal == NULL){
			printf("Memoria insufficiente\n");
			exit(EXIT_FAILURE);
		}

		printf("Allocazione completata sul processo %d\n", rank);

		// ricevo il train set dal processo root
 		MPI_Bcast(trainData, N * A, MPI_FLOAT, rank_root, MPI_COMM_WORLD);
    	MPI_Bcast(trainClass, N, MPI_UINT8_T, rank_root, MPI_COMM_WORLD);

    	// ricevo la porzione di test set assegnata a questo processo
    	MPI_Scatterv(NULL, NULL, NULL, NULL, localTestData, countsAttr[rank], MPI_FLOAT, rank_root, MPI_COMM_WORLD);
		MPI_Scatterv(NULL, NULL, NULL, NULL, localTestClass, countsClasses[rank], MPI_UINT8_T, rank_root, MPI_COMM_WORLD);	

		// eseguo l'algoritmo knn solo per la porzione di test assegnata al processo
		int localError = localKnn(trainData, localTestData, trainClass, localTestClass, countsAttr[rank] / A, confusionMatrixLocal, N, LABELS, A, K, rank);

		MPI_Barrier(MPI_COMM_WORLD);

		// inivio al proesso root la porzione calcolata localmente della matrice di confusione, che verrà poi unita con le altre
		MPI_Reduce(confusionMatrixLocal, NULL, LABELS*LABELS, MPI_INT, MPI_SUM, rank_root, MPI_COMM_WORLD);

		// invio il valore locale degli errori che verrà sommato a quello calcolato da ogni processo
		MPI_Reduce(&localError, NULL, 1 , MPI_INT, MPI_SUM, rank_root, MPI_COMM_WORLD);

		//MPI_Barrier(MPI_COMM_WORLD);
		printf("Reduce\n");

		// libero la memoria utilizzata
 		free(trainData); trainData = NULL;
 		free(trainClass); trainClass = NULL;
	    free(localTestData); localTestData = NULL;
		free(localTestClass); localTestClass = NULL;
		free(confusionMatrixLocal); confusionMatrixLocal = NULL;

		printf("Fine %d\n", rank);
   	}

   	// termino MPI 
 	MPI_Finalize();

}