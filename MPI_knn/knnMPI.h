#ifndef LBP_FUNCTIONS 
#define LBP_FUNCTIONS
#include <stdint.h>
//calcola della distanza euclidea tra un sample del train e uno del test
float computeDist(float* train, float* test);

// ordina e aggiorna le distanze e gli indici
void sort(float * distance, int * index);

void splitData(
    int* countsAttr, 
    int* countsClasses, 
    int* adressesAttr, 
    int* adressesClasses, 
    int comm_size, 
    int nAttr, 
    int testSize);

void knn(	
	const char* trainFile,
	const char* testFile,
	int size,
	int rank,
	double time_init);

int localKnn(
	float* trainData, 
	float* localTestData, 
	uint8_t* trainClass, 
	uint8_t* localTestClass, 
	int NrecordTesting, 
	int* confusionMatrix);

#endif



