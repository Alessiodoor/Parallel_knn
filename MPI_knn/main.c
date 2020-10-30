#include <stdint.h>
#include <stdio.h> 
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "mpi.h"
#include "datasetFunctions.h"
#include "knn.h"

// attributi
#define A 30
// labels
#define LABELS 10

int main(int argc, char *argv[])
{
	// rank de processo, size: numero di processi nel communicator
	int rank, size;    

    // inizializzo il communicator e assegno  ad ogni processo il relativo rank e il numero totale di processi 
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &size);   
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  

	// argomenti:
   // train file name
   // test file name
   // N: numero di sample di train
   // M: numero di sample di test
   // k: numero di vicini
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
   int N = atoi(argv[3]);
   int M = atoi(argv[4]);
   int K = atoi(argv[5]);

   if (K > N){
      printf("Errore il numero di vicini non può essere superiore al numero di sample!\n");
      exit(EXIT_FAILURE);
   }

   if (K % 2 == 0){
      printf("Inserire un numero di vicini dispari!\n");
      exit(EXIT_FAILURE);
   }

   	// knn implentato in MPI
   	knn(trainFile, testFile, K, N, M, A, LABELS, size, rank);

	return 0;
}