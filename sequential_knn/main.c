#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "datasetFunctions.h"
#include "knn.h"
// attributi
#define A 30
// labels
#define LABELS 10

int main(int argc, char const *argv[])
{
   //imposto la variabile che conterrà il tempo di partenza dell'esecuzione
   clock_t start; 
   start = clock();

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
   int * classesTraining = (int*) malloc(N *sizeof(int));
   int * classesTesting = (int*)  malloc(M *sizeof(int));


   // Controllo di aver allocato correttamente la memoria
   if(trainData == NULL || testData == NULL || classesTesting == NULL || classesTraining == NULL){
      printf("Memoria insufficiente\n");
      exit(EXIT_FAILURE);
   }

   // leggo i dati di train e test
   readFile(trainFile, N, A, trainData, classesTraining);
   readFile(testFile, M, A, testData, classesTesting);

   // eseguo knn
   knn(trainData, testData, classesTraining, classesTesting, K, N, M);

   // Libero la memoria utilizzata
   free(trainData); trainData = NULL;
   free(testData); testData = NULL;
   free(classesTraining); classesTraining = NULL;
   free(classesTesting); classesTesting = NULL;

   //calcolo tempo d'esecuzione totale
   float totaltime = (float)(clock() - start)/ (float) CLOCKS_PER_SEC;
   //printf("total time: %f \n", totaltime);

   writeResult(K, N, M, A, totaltime, "result.json");

	return 0;
}