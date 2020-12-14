#include <stdint.h> 
#include "datasetFunctions.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <limits.h>
//#include <cjson/cJSON.h>

/*
Funzione per leggere i dati di train e test da file
Parametri:
path: percorso del file
lines: numero di righe/sample 
Nfeatures: numero di attributi di ogni sample
data: array in cui verranno salvati i samples, deve essere allocato precedentemente
labels: array dove verranno salvate le labels del sample, deve essere allocato precedentemente
*/
void read_file(const char *filename, int lines, int Nfeatures, float* data, uint8_t * labels) {
  	//char path[100] = "../dataset/"; 
  	//strcat( path, filename );
  	FILE *file = fopen(filename, "r");

  	if (file == NULL){
		printf("Impossibile leggere il file!");
		exit(EXIT_FAILURE);
	}
	else{
	  	for (int i = 0; i < lines; i++) {
	    	for (int j = 0; j < Nfeatures; j++) {
	      		fscanf(file, "%f", &data[i * Nfeatures + j]);
	      		//printf("i %d j %d data %f\n", i, j, data[i * Nfeatures + j]);
	    	}
		    float label;
		    fscanf(file, "%f", &label);
		    labels[i] = (uint8_t )label;
		}
	}
}

/*
Salvo su file i parametri dell'esecuzione e il tempo totale, compreso di tempo di lettura e esecuzioe
Parametri:
K: numero di vicini
trainSize: numero di sample di train
testSize: numero di sample di test
attributes: numero di attributi per sample
totalTime: tempo d'esecuzione che si vuole salvare
fileName: nome del file di destinazione
*/
int saveResultsOnFile(int k, int trainSize, int testSize, int attributes, float totalTime, int size, char *fileName){
	FILE *fp;

	int i, j;
	char * wheretoprint = "resultsKNN_mpi.out";
	fp = fopen(wheretoprint,"a");

	if (fp == NULL) {
	    printf("\nCannot write on %s\n", wheretoprint);
	    return -1;
	}

	//fprintf(fp, "Test with %d process K = %d trainingData %d and testingData: %d , time: %f\n\n",size, K, N, M, time);
	fprintf(fp, 
		"K %d\n trainSize %d\n trainSize %d\n attributes %d\n totalTime %f\n NP %d\n", 
		k, trainSize, testSize, attributes, totalTime, size
	);
	fclose(fp);

	return 0;
}

/*
Analoga alla funzione precedente ma salva i risultati su un file json
*/
void writeResultJson(int k, int trainSize, int testSize, int attributes, float totalTime, int size, char *fileName){
	/*
	cJSON *result = cJSON_CreateObject();

	cJSON_AddNumberToObject(result, "K", k);
    cJSON_AddNumberToObject(result, "trainSize", trainSize);
    cJSON_AddNumberToObject(result, "testSize", testSize);
    cJSON_AddNumberToObject(result, "attributes", attributes);
    cJSON_AddNumberToObject(result, "totalTime", totalTime);
    cJSON_AddNumberToObject(result, "NP", size);

    const char* const stringResult = cJSON_Print(result);

	FILE *fptr;
	fptr = fopen(fileName, "a");

	if(fptr == NULL)  {
      	printf("Errore scrittuta file");   
      	exit(1);             
   	}

   	fprintf(fptr, "%s,", stringResult);

	fclose(fptr);

	cJSON_Delete(result);
	*/

	printf("K %d\n", k);
	printf("trainSize %d\n", trainSize);
	printf("testSize %d\n", testSize);
	printf("attributes %d\n", attributes);
	printf("totalTime %f\n", totalTime);
	printf("NP %d\n", size);
}

/*
Funzione per stampare a video la matrice di confuzione ottenuta dall'esecuzione dell'algoritmo Knn
Parametri:
confusionMatrix: matrice di confuzione ottenuta dall'esecuzione
labels: numero di labels contenute della matrice 
*/
void printConfusionMatrix(int* confusionMatrix, int LABELS){
	printf ("\tReale X Risultato\n");
	for(int i=0; i <LABELS; i++){
		for(int j=0; j < LABELS; j++)
			printf("%d ", confusionMatrix[i* LABELS + j]);
		printf("\n");
	}
}
