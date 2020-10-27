#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

#include "datasetFunctions.h"
#include "knnOMP.h"

// attributi
#define A 30
// labels
#define LABELS 10
// numtreads
#define NT 32

int main(int argc, char *argv[])
{
	//imposto la variabile che conterrà il tempo di partenza dell'esecuzione
   	double start = omp_get_wtime();
   	double executionTime;

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

   float * trainData = (float *) malloc(N* A * sizeof(float));
   float * testData = (float *) malloc(M * A * sizeof(float));

   // vettore delle classi di train e test 
   uint8_t * classesTraining = (uint8_t*) malloc(N *sizeof(uint8_t));
   uint8_t * classesTesting = (uint8_t*)  malloc(M *sizeof(uint8_t));

	// Controllo di aver allocato correttamente la memoria
   if(trainData == NULL || testData == NULL || classesTesting == NULL || classesTraining == NULL){
      printf("Memoria insufficiente per assegnare il dataset\n");
      exit(EXIT_FAILURE);
   }

   	// leggo i dati di train e test
   	readFile(trainFile, N, A, trainData, classesTraining);
   	readFile(testFile, M, A, testData, classesTesting);

   	knn(NT, trainData, testData, classesTraining, classesTesting, K, N, M, start);

   	// Libero la memoria utilizzata
   	free(trainData); trainData = NULL;
   	free(testData); testData = NULL;
   	free(classesTraining); classesTraining = NULL;
   	free(classesTesting); classesTesting = NULL;

   	executionTime = omp_get_wtime() - start;
   	writeResultJson(K, N, M, A, executionTime, "resultsOpenMP.json");

	return 0;
}