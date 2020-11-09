#ifndef UTILITY_FUNCTIONS_H 
#define UTILITY_FUNCTIONS_H
#include <stdint.h>

/*
Funzione per leggere i dati di train e test da file
Parametri:
path: percorso del file
lines: numero di righe/sample 
Nfeatures: numero di attributi di ogni sample
data: array in cui verranno salvati i samples, deve essere allocato precedentemente
labels: array dove verranno salvate le labels del sample, deve essere allocato precedentemente
*/
void read_file(
	const char *filename, 
	int lines, 
	int Nfeatures, 
	float* data, 
	uint8_t * labels);

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
int writeResult(
	float time, 
	int size, 
	int K, 
	int N, 
	int M);

/*
Analoga alla funzione precedente ma salva i risultati vengono salvati su un filejson
*/
void writeResultJson(
	int k, 
	int trainSize, 
	int testSize, 
	int attributes, 
	float totalTime,
	int size,
	char *fileName);

/*
Funzione per stampare a video la matrice di confuzione ottenuta dall'esecuzione dell'algoritmo Knn
Parametri:
confusionMatrix: matrice di confuzione ottenuta dall'esecuzione
labels: numero di labels contenute della matrice 
*/
void printConfusionMatrix(int* confusionMatrix, int LABELS);

#endif