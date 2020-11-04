#include <stdint.h> 
#include "datasetFunctions.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <limits.h>

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
  char path[100] = "../dataset/"; 
  strcat( path, filename );
  FILE *file = fopen(path, "r");
	if (file == NULL){
		printf("Impossibile leggere il file!");
		exit(EXIT_FAILURE);
	}
	else{
	  for (int i = 0; i < lines; i++) {
	    for (int j = 0; j < Nfeatures; j++) {
	      fscanf(file, "%f", &data[i * Nfeatures + j]);
	      //printf("i %d j %d data[i*line +j] %f\n",i, j, data[i * Nfeatures + j]);
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
int saveResultsOnFile(float time, int size, int K, int N, int M){
	FILE *fp;

	int i, j;
	char * wheretoprint = "resultsKNN_mpi.out";
	fp = fopen(wheretoprint,"a");

	if (fp == NULL) {
	    printf("\nCannot write on %s\n", wheretoprint);
	    return -1;
	}

	fprintf(fp, "Test with %d process K = %d trainingData %d and testingData: %d , time: %f\n\n",size, K, N, M, time);
	  
	fclose(fp);

	return 0;
}


void printData(float * data, uint8_t* labels, int size, int M){
	for(int i=0; i< size; i++){
		for(int j=0; j <M; j++)
			printf(" %f ", data[i*M +j]);
		printf("Classe %d\n", labels[i] );
	}
	printf("\n");
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
