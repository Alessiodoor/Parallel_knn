#include <stdint.h> 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

//#include "input.h"
#include "knnCuda.h"
//#include "check.h"
#include "cudaError.h"
#include "datasetFunctions.h"

// attributi
#define A 30
// labels
#define LABELS 10
// numtreads
#define NT 32
//#define BLOCK_SIZE 32

int main(int argc, char *argv[])
{

	// argomenti:
   	// train file name
   	// test file name
   	// N: numero di sample di train
   	// M: numero di sample di test
   	// k: numero di vicini
   	// BLOCK_SIZE: numro di blocchi per grid
   	if(argc != 7){
      	printf(
        	 "Errore non sono stati specificati correttamente i parametri:\n"
         	"1 - Train fileName\n"
         	"2 - Test tileName\n"
         	"3 - Numero sample di train\n"
         	"4 - Numero sample di test\n"
         	"5 - K: numero di vicini\n"
         	"6 - BLOCK_SIZE: numero di blocchi per grid");
      	exit(EXIT_FAILURE);
   	}

   	const char * trainFile = argv[1];
   	const char * testFile = argv[2];

   	int N = atoi(argv[3]);
   	int M = atoi(argv[4]);
   	int K = atoi(argv[5]);
   	int BLOCK_SIZE = atoi(argv[6]);

   	if (K > N){
      	printf("Errore il numero di vicini non può essere superiore al numero di sample!\n");
      	exit(EXIT_FAILURE);
   	}

   	if (K % 2 == 0){
      	printf("Inserire un numero di vicini dispari!\n");
      	exit(EXIT_FAILURE);
   	}

	// device
	int deviceIndex = 0;

	// ottengo il numero di schede presenti
	int count;
	HANDLE_ERROR( cudaGetDeviceCount( &count ) );
    
    // controllo l'esistenza della scheda disponbile
    if(deviceIndex < count)
    {
        HANDLE_ERROR(cudaSetDevice(deviceIndex));
    }
    else
    {
        printf("Device non disponbile!\n");
        exit(EXIT_FAILURE);        
    }

    // proprietà della scheda video
    cudaDeviceProp prop;
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, deviceIndex));

    // misuro il tempo d'esecuzione
	cudaEvent_t start, stop, stopRead, stopSendData, primoStep, secondoStep;
	
	HANDLE_ERROR( cudaEventCreate( &start ) );
	HANDLE_ERROR( cudaEventCreate( &stop ) );
	HANDLE_ERROR( cudaEventCreate( &stopRead ) );
	HANDLE_ERROR( cudaEventCreate( &stopSendData ) );
	HANDLE_ERROR( cudaEventCreate( &primoStep ) );
	HANDLE_ERROR( cudaEventCreate( &secondoStep ) );
	
	HANDLE_ERROR( cudaEventRecord( start, 0 ) );

	// alloco le matrici del dataset e degli altri vettori utili
	float * trainingData= (float *) malloc(N * A * sizeof(float));
	float * testingData= (float *) malloc(M * A * sizeof(float));

	int * classesTraining = (int*) malloc(N *sizeof(int));
	int * classesTesting = (int*)  malloc(M *sizeof(int));

	float * dist = (float *) malloc(M * N * sizeof(float));
	
	// controllo che le variabili siano state allocate correttamente
	if(trainingData == NULL || testingData == NULL || classesTesting == NULL || classesTraining == NULL){
		printf("Not enough memory!\n");
		exit(EXIT_FAILURE);
	}

	// leggo il dataset dal file
	readFile(trainFile, N, A, trainingData, classesTraining);
	readFile(testFile, M, A, testingData, classesTesting);

	HANDLE_ERROR( cudaEventRecord( stopRead, 0 ) );
	HANDLE_ERROR( cudaEventSynchronize( stopRead ) );
	
	// puntattori ai dati sul device
	float* dev_train; 
	float* dev_test;
	float* dev_dist;
	int* dev_label;
	
	// alloco la memoria per il dataset sulla gpu in memoria globale
	HANDLE_ERROR( cudaMalloc((void**)&dev_train, N * A * sizeof(float)));
	
	HANDLE_ERROR( cudaMalloc((void**)&dev_test, M * A * sizeof(float)));

	// allocco la matrice delle distanze e delle label
	HANDLE_ERROR( cudaMalloc((void**)&dev_dist, N * M * sizeof(float)));

	// copio il cotenuto del dataset sulle variabili presenti sul device
	HANDLE_ERROR( cudaMemcpy(dev_train, trainingData, N * A * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR( cudaMemcpy(dev_test, testingData, M * A * sizeof(float), cudaMemcpyHostToDevice));	

	HANDLE_ERROR( cudaEventRecord(stopSendData, 0));
	HANDLE_ERROR( cudaEventSynchronize(stopSendData));
	
	// registre il tempo di lettura
	float elapsedTimeRead;
	HANDLE_ERROR( cudaEventElapsedTime(&elapsedTimeRead, start, stopSendData ));
	
	// creo i blocchi da BLOCK_SIZE * BLOCK_SIZE thread
	dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1); 

	// calcolo il numero di righe e colonne in base a BLOCK_SIZE
	int dim_row = (M + 1 % BLOCK_SIZE == 0) ? M / BLOCK_SIZE : M / BLOCK_SIZE + 1;
	int dim_col = (N + 1 % BLOCK_SIZE == 0) ? N / BLOCK_SIZE : N / BLOCK_SIZE + 1;
	
	// creo la griglia di threads
	dim3 grid(dim_col, dim_row, 1); 

	// calcola distanza euclidea tra i punti train e test attraverso la funzione kernel
	euclideanDistance_kernel<<<grid, block>>>(N, M, A, dev_train, dev_test, dev_dist);

	// alloco le variabili che mi serviranno per calcolare al matrice di confusione
	int * label = (int*) malloc(M * K *sizeof(int));
	int* countsLabel = (int*) malloc(sizeof(int)* LABELS);
	int* confusionMatrix = (int*) malloc(sizeof(int)* LABELS * LABELS);

	if(confusionMatrix ==NULL || countsLabel == NULL || label == NULL){
		printf("Not enough memory!\n");
		exit(EXIT_FAILURE);
	}

	// inizializza a zero la matrice di confusione
	for(int i = 0; i < LABELS * LABELS; i++){
		confusionMatrix[i] = 0;
	}

	// barriera per assicurarsi che tutte le distanze siano state calcolate
	cudaDeviceSynchronize();
	HANDLE_ERROR( cudaEventRecord(  primoStep, 0 ) );
	HANDLE_ERROR( cudaEventSynchronize(  primoStep ) );
	
	// calcolo il tempo d'esecuzione
	HANDLE_ERROR( cudaEventElapsedTime( &elapsedTimeRead, start, primoStep ) );
	
	// elimino il dataset su device, perchè non verrà più usato
	HANDLE_ERROR( cudaFree(dev_train) );
    HANDLE_ERROR( cudaFree(dev_test) );

    // inizializzo l'array delle label sul device
    HANDLE_ERROR( cudaMalloc( (void**)&dev_label, M * K * sizeof(int) ) );

    // creo 
	dim3 blockSort(BLOCK_SIZE, 1, 1);
	dim3 gridSort(dim_row, 1, 1);

	sort_kernel<<<gridSort, blockSort>>>(N, M, K, dev_dist, dev_label);

	// barriera per assicurarsi che tutti i threads abbiamo concluso l'operazione
	cudaDeviceSynchronize();

	// copio l'array delle label dal device alla memoria principale
	HANDLE_ERROR(cudaMemcpy(label , dev_label, M * K * sizeof(int), cudaMemcpyDeviceToHost ) );
	
	HANDLE_ERROR( cudaEventRecord(  secondoStep, 0 ) );
	HANDLE_ERROR( cudaEventSynchronize(  secondoStep ) );
	
	HANDLE_ERROR( cudaEventElapsedTime( &elapsedTimeRead, start, secondoStep ) );
	
	// numero di errori di classificazione commessi dall'algoritmo KNN
	int error = 0;
	
	// calcolo la matrice di confusione, lasciato alla CPU
	// ciclo sui samples di test
	for (int i = 0; i < M; i++){
		// inizializzo l'array utilizzato per contare l'occorrenza delle labels
		for(int l = 0; l < LABELS; l++){
			countsLabel[l] = 0;
		}
		int bestLabel = 0;
		// ciclo sui k sample di train vicini al sample di test i-esimo
		for(int j = 0; j < K; j++){	
			// indice e classe del sample di train j-esimo
			int indice = label[i*K + j];
			int classe = classesTraining[indice]; 
			// incremento il contatore di questa classe
			countsLabel[classe] = countsLabel[classe] + 1;
			// aggiorno la classe migliore se il numero di occorrenze è maggiore
			if(countsLabel[classe] > countsLabel[bestLabel])
				bestLabel = classe;
			}

		// controllo che la label calcolata corrisponda a quella vera
		int realLabel = classesTesting[i];
		if (realLabel != bestLabel){
			error = error + 1;
		}
			
		// aggiorno la matrice di confusione
		confusionMatrix[realLabel * LABELS + bestLabel] = confusionMatrix[realLabel * LABELS + bestLabel] + 1;	
	}

	// libero memoria utilizzata
	free(trainingData); trainingData = NULL;
	free(testingData); testingData = NULL;
	free(dist); dist=NULL;
	
	free(classesTraining); classesTraining = NULL;
	free(classesTesting); classesTesting = NULL;
	
	free(confusionMatrix); confusionMatrix=NULL;
	
	free(label); label=NULL;
	free(countsLabel); countsLabel= NULL;

	// libero memoria sul device
	HANDLE_ERROR( cudaFree(dev_label ) );
    HANDLE_ERROR( cudaFree(dev_dist ) );
    	
	// calcol il tempo totale d'esecuzione
	HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
	HANDLE_ERROR( cudaEventSynchronize( stop ) );
	
	float elapsedTime;
	HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime, start, stop ) );
	
	HANDLE_ERROR( cudaEventDestroy( start ) );
	HANDLE_ERROR( cudaEventDestroy( stop ) );

	// salvo risultati su file
	saveResultsOnFile(K, N, M, A, elapsedTime/1000,BLOCK_SIZE);

	return 0;
}