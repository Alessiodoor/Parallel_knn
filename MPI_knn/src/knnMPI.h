#ifndef LBP_FUNCTIONS 
#define LBP_FUNCTIONS
#include <stdint.h>

/*
Funzione per calcolare la distanza euclidea tra gli attributi di un sample di train e uno di test
Parametri:
train: sample di train
test: sample di test
*/
float euclideanDist(float* train, float* test, int A);

/*
Funzione per ordinare ordinare i sample di train per il test i-esimo in base alla distanza tra train e test
Questa funzione ordina i vicini di un solo sample di test.
Parametri:
distances: array delle distanze tra il sample di test i-esimo e tutti i sample di train
index: array degli indici di tutti i sample di train per il sample id tst i-esimo
K: numero di vicini
N: numero di sample di train
*/
void sort(float * distance, int * index, int N, int K);

/*
Funzione che calcola le porzioni di test che verranno assegnate ad ogni processo,
in particolare vengono calcolati il numero di sample e la posizione da cui leggere i sample 
Se il numero di sample non è divisibile per il numero di processi gli elementi in eccesso vengono assegnati uno alla volta per ogni proceso,
così da lasciare i sample contigui
Parametri:
countsAttr: array che contiene il numero di sample di test per ogni processo, i-esimo elemento è il numero di elementi dell'i-esimo processo
countsClass: array che contiene il numero di classi di test per ogni processo, i-esimo elemento è il numero di elementi dell'i-esimo processo
addressAttr: array che contiene , per ogni processo, l'indirizzo di inizio della porzione locale di test nel dataset di test principale, per quanto riguarda i sample
addressClasses: array che contiene , per ogni processo, l'indirizzo di inizio della porzione locale di test nel dataset di test principale, per quanto riguarda le classi
comm_size: numero di processi
nAttr: numero di Attrbuti per sample
testSize: numero di sample di test
*/
void splitData(
    int* countsAttr, 
    int* countsClasses, 
    int* adressesAttr, 
    int* adressesClasses, 
    int comm_size, 
    int nAttr, 
    int testSize);

/*

*/
void knn(	
	const char* trainFile,
	const char* testFile,
	int size,
	int rank,
	double time_init,
	int M,
	int N,
	int K,
	int A,
	int LABELS);

/*
Funzione che esegue l'lagoritmo Knn con la porzione locale di test e l'intero train
La funzione calcola la porzione locale della matrice di confusione e delg errori di classificazione
Parametri:
trainData: samples di train
localTestData: samples di test
trainClass: labels dei sample di train
localtestClass: labels dei sample di test (usati per la verifica)
N: numero sample di trian
M: numero sample di test
LABELS: classi del dataset
A: numero di attributi per sample
K: numero di vicini
*/
int localKnn(
	float* trainData, 
	float* localTestData, 
	uint8_t* trainClass, 
	uint8_t* localTestClass, 
	int NrecordTesting, 
	int* confusionMatrix,
	int N,
    int M,
    int LABELS,
    int A,
    int K);

#endif



