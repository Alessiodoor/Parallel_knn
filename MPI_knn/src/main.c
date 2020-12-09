#include <stdint.h>
#include <stdio.h> 
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include <mpi.h>
#include "knnMPI.h"

// Numero di attributi per sample del dataset
#define A 30
// Numero di possibili labels
#define LABELS 10

int main (int argc, char* argv[]){
	// rank del processo corrente e totale numero di processi
	int rank, size;                          

	// inizializzo MPI 
    MPI_Init(NULL, NULL);
    // salvo il numeor di preocessi del communicator 
    MPI_Comm_size(MPI_COMM_WORLD, &size);   
    // salvo il rank di ogni processo
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  

    // variabile che conterrà il tempo di inizio dell'esecuzione
    double time_init;
    // assegno il tempo di inizio
    time_init = MPI_Wtime();
	
	// controllo i parametri da riga di comando
	if(argc != 6){
      printf(
         "Errore non sono stati specificati correttamente i parametri:\n"
         "1 - Train fileName\n"
         "2 - Test tileName\n"
         "3 - Numero sample di train\n"
         "4 - Numero sample di test\n"
         "5 - K: numero di vicini");
      exit(EXIT_FAILURE);
   }

   const char * trainFile = argv[1];
   const char * testFile = argv[2];
   // salvo il numero di sample di train
   int N = atoi(argv[3]);
   // salvo il numero di sample di test
   int M = atoi(argv[4]);
   // salvo il numero di vicini K
   int K = atoi(argv[5]);
    
    //check consistenza valore K
	if (K > N){
		printf("Errore il numero di vicini non può essere superiore al numero di sample!\n");
		exit(EXIT_FAILURE);
	}

	if (K % 2 == 0){
		printf("Inserire un numero di vicini dispari!\n");
		exit(EXIT_FAILURE);
	}

	// inizio l'esecuzione del knn parallelo 
	knn(trainFile, testFile, size, rank, time_init, M, N, K, A, LABELS);

	// concludo MPI per ogni processo
 	MPI_Finalize();
	return 0;

}
