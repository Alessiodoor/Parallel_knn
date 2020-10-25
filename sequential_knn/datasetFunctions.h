#ifndef DATASETFUNCTIONS_H
#define DATASETFUNCTIONS_H 

/*
Funzione per leggere i dati di train e test da file
Parametri:
path: percorso del file
lines: numero di righe/sample 
Nfeatures: numero di attributi di ogni sample
data: array in cui verranno salvati i samples, deve essere allocato precedentemente
labels: array dove verranno salvate le labels del sample, deve essere allocato precedentemente
*/
void readFile(
	const char *path, 
	int lines, 
	int Nfeatures, 
	float* data, 
	int * labels);

/*
Funzione per stampare a video la matrice di confuzione ottenuta dall'esecuzione dell'algoritmo Knn
Parametri:
confusionMatrix: matrice di confuzione ottenuta dall'esecuzione
labels: numero di labels contenute della matrice 
*/
void printConfusionMatrix(
	int* confusionMatrix, 
	int labels);

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
void writeResult(
	int k, 
	int trainSize, 
	int testSize, 
	int attributes, 
	float totalTime,
	char *fileName);
/*
void writeResultJson(
	int k, 
	int trainSize, 
	int testSize, 
	int attributes, 
	float totalTime,
	char *fileName);
*/
#endif