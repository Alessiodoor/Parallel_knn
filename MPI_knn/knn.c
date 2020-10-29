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

void setupRoot(){

}

void knn(
	char* trainFile,
	char* testFile,
	int K, 
	int N,
	int M, 
	int A,
	int size, 
	int rank){

	int rank_root= 0;
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
   	void splitData(countsAttr, countsClasses, addressesAttr, addressesClasses, size, A, M);

   	if(rank == rank_root) {
   		// in questo if sono presenti tutte le operazioni che vengono eseguite dal processo root
   		// inizializzo le matrici del dataset
		float * trainData = (float *) malloc(N * A * sizeof(float));
		float * testData = (float *) malloc(M* A * sizeof(float));

		uint8_t * classesTraining = (uint8_t*) malloc(N *sizeof(uint8_t));
		uint8_t * classesTesting = (uint8_t*)  malloc(M *sizeof(uint8_t));

		if(trainingData == NULL || testingData == NULL || classesTesting == NULL || classesTraining == NULL){
			printf("Memoria insufficiente!\n");
			exit(EXIT_FAILURE);
		}

		// leggo il dataset e lo salvo nelle variabili appena dichiarate
		readFile(trainFile, N, M, trainData, classesTraining);
		readFile(testFile, P, M, testData, classesTesting);

		// inivio tutti i dati di train a tutti i processi
	    MPI_Bcast(trainData, N * A, MPI_FLOAT, root_rank, MPI_COMM_WORLD);
	    MPI_Bcast(classesTraining, N, MPI_UINT8_T, root_rank, MPI_COMM_WORLD);

	    // creo una variabile che conterrà la porzione di test assegnata ad processo root
		float * localTestingData = (float *) malloc(countsAttr[rank] * sizeof(float));
		uint8_t * localClassesTesting = (uint8_t*)  malloc(countsClasses[rank] * sizeof(uint8_t));	

		if(localTestingData == NULL || localClassesTesting == NULL){
			printf("Memoria insufficiente!\n");
			exit(EXIT_FAILURE);
		}

		// Invio la porzione di test ad ogni processo, per quanto riguarda gli attributi e le classi
   		MPI_Scatterv(testData, countsAttr, addressesAttr, MPI_FLOAT, localTestingData, countsAttr[root_rank], MPI_FLOAT, root_rank, MPI_COMM_WORLD);
   		MPI_Scatterv(classesTesting, countsClasses, adressesClasses, MPI_UINT8_T, localClassesTesting, countsClasses[root_rank], MPI_INT, root_rank, MPI_COMM_WORLD);
   		

   	}else {

   	}
}