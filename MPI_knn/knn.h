#ifndef KNN_MPI_H
#define KNN_MPI_H

/*

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
	char* trainFile,
	char* testFile,
	int K, 
	int N,
	int M,
	int A,
	int size,
	int rank);

#endif 
