#include <stdint.h>
#include <stdio.h> 
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include <mpi.h>
#include "input.h"
#include "knnMPI.h"

int main (int argc, char* argv[]){

	int rank, size;                           //rank del processo root

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &size);   
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  

    double time_init;
    time_init = MPI_Wtime();
	
    const char * trainFile = argv[1];
	const char * testFile = argv[2];
	if((argc -1) != 2){
		printf("Errore non sono stati specificati correttamente i file del dataset!\n");
		exit(EXIT_FAILURE);
	}
    
    //check consistenza valore K
	if (K > N){
		printf("Errore il numero di vicini non può essere superiore al numero di sample!\n");
		exit(EXIT_FAILURE);
	}

	if (K % 2 == 0){
		printf("Inserire un numero di vicini dispari!\n");
		exit(EXIT_FAILURE);
	}

	knn(trainFile, testFile, size, rank, time_init);

	//tutti i processi devono chiudere correttamente MPI
 	MPI_Finalize();
	return 0;

}
