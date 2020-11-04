#include <stdint.h> 
#include "knnMPI.h"
#include "datasetFunctions.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

/*
Funzione per calcolare la distanza euclidea tra gli attributi di un sample di train e uno di test
Parametri:
train: sample di train
test: sample di test
*/
float euclideanDist(float* train, float* test, int M){
    float sum = 0.f;
    for (int d = 0; d < M; ++d) {
        const float diff = train[d] - test[d];
        sum += diff * diff;
    }
    return sqrtf(sum);
}

/*
Funzione per ordinare ordinare i sample di train per il test i-esimo in base alla distanza tra train e test
Questa funzione ordina i vicini di un solo sample di test.
Parametri:
distances: array delle distanze tra il sample di test i-esimo e tutti i sample di train
index: array degli indici di tutti i sample di train per il sample id tst i-esimo
K: numero di vicini
N: numero di sample di train
*/
void sort(float * distance, int * index, int N, int K){
    index[0] = 0;
    for(int i = 1; i < N; ++i){
        float distanzaCorrente = distance[i];
        int indiceCorrente = i;

        if( i >= K && distanzaCorrente >= distance[K-1]){
            continue;
        }

        int j = i;
        if (j > K-1)
            j = K-1;
        
        while(j > 0 && distance[j-1] > distanzaCorrente){
            distance[j] = distance[j-1];
            index[j] = index[j-1];
            --j;
        }

        distance[j] = distanzaCorrente;
        index[j] = indiceCorrente;
    }
}

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
    int testSize){

    // calcolo il resto della divisione tra il numero di processi e la dimensione del test
    int extraRows = testSize % comm_size;  
    // righe già assegnate, da utilizzare per salvare l'indirizzo della porzione di test successiva
    int sumAttr = 0;   
    int sumClasses = 0;                      

    // per ogni processo:
    for (int i = 0; i < comm_size; i++) {            
        // assegno il numero di attributi del test per ogni processo, ovvero il numero di righe moltiplicato per il numero di attributi
        countsAttr[i] = (testSize / comm_size) * nAttr;    
        // assegno il numero di classi del test per ogni processo, una classe per ogni riga
        countsClasses[i] = (testSize / comm_size);    

        // se la divisione delle righe di tesst ha il resto
        // assegno le righe extra in ordine così ho le celle di memoria contigue
        if (extraRows > 0) {              
            // aggiungo al processo gli elementi di una riga in eccesso        
            countsAttr[i]= countsAttr[i] + nAttr;   
            // aggiungo al processo la classe relativa alla riga in eccesso
            countsClasses[i] = countsClasses[i] + 1; 

            // sottraggo la riga che ho aggiunto 
            extraRows--;
        }

        // assegno l'indirizzo di inizio della porzione che dovrà elaborare ogni processo, sia per gli attributi che per le classi
        adressesAttr[i] = sumAttr;
        adressesClasses[i] = sumClasses;
        // mi sposto all'indirizzo successivo, sia per attributi che per classi
        sumAttr += countsAttr[i];
        sumClasses += countsClasses[i];
    }
}

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
    int * confusionMatrix,
    int N,
    int M,
    int LABELS,
    int A,
    int K){  

        float* k_distances = (float*) malloc(sizeof(float) * N); 
        
        //indice della label del sample di train (dopo ordinamento i primi k saranno le etichette dei K vicini)
        int * k_labels = (int*) malloc(sizeof(int) * N);

        //label di ogni sample per majority voting
        int* countsLabel = (int*) malloc(sizeof(int) * LABELS);

        //check memory
        if (countsLabel == NULL || k_distances == NULL || k_labels == NULL){
            printf("Not enough memory!\n");
            exit(EXIT_FAILURE);
        }
        
        //inizializza a zero la matrice di confusione
        for(int i = 0; i < LABELS * LABELS; i++){
            confusionMatrix[i] = 0;
        }
        
        //numero di errori compessi dall'algoritmo KNN
        int error = 0;
        
        //per ogni sample del testing da trattare
        for(int i = 0; i < NrecordTesting; i++){
            //calcolo distanze con tutti i punti del training
            for(int j = 0; j < N; j++){
                k_distances[j] = euclideanDist(&trainData[j * A], &localTestData[i * A], M);
                k_labels[j] = j;
            }
            
            //ordino i dati
            sort(k_distances, k_labels, N, K);
            
            //inizializza a zero il vettore
            for(int i = 0; i < LABELS; i++){
                countsLabel[i] = 0;
            }

            int bestLabel = 0;
            
            //per i primi k vicini
            for(int j=0; j<K; j++){ 
                int indice = k_labels[j];
                int label = trainClass[indice]; 
                countsLabel[label] = countsLabel[label] + 1;
                if(countsLabel[label] > countsLabel[bestLabel])
                    bestLabel = label;
            }

            int realLabel = localTestClass[i];
            if (realLabel != bestLabel)
                error = error + 1;
            
            //update confusion matrix
            confusionMatrix[realLabel * LABELS + bestLabel] = confusionMatrix[realLabel * LABELS + bestLabel] +1;

        }

        free(countsLabel); countsLabel = NULL; 
        free(k_distances); k_distances = NULL; 
        free(k_labels); k_labels = NULL;
        return error;
}

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
    int LABELS){

    int root_rank = 0; 

    //numero di elementi del testing che ogni processo deve gestire per il calcolo LBP
    //serve per scatterv
    int* countsRow = (int *) malloc(size * sizeof(int));

    //indirizzi di inizio della porzione che ogni processo deve gestire 
    int* displsRow = (int *) malloc(size * sizeof(int)); 

    //numero di elementi che ogni processo deve gestire
    int* countsClasses = (int *) malloc(size * sizeof(int)); 

    //indirizzi di inizio della porzione che ogni processo deve gestire 
    int* displsClasses = (int *) malloc(size * sizeof(int));

    if(countsRow == NULL || displsRow == NULL || countsClasses == NULL || displsClasses == NULL){
        printf("Not enough memory!\n");
        exit(EXIT_FAILURE);
    }

    //funzione che permette di suddividere i record di testing da gestire
    splitData(countsRow, countsClasses, displsRow, displsClasses, size, A, M); 

    if(rank ==0){

        float * trainData = (float *) malloc(N * A * sizeof(float));
        float * testData = (float *) malloc(M * A * sizeof(float));

        uint8_t * trainClass = (uint8_t*) malloc(N * sizeof(uint8_t));
        uint8_t * testClass = (uint8_t*)  malloc(M * sizeof(uint8_t));

        if(trainData == NULL || testData == NULL || testClass == NULL || trainClass == NULL){
            printf("Not enough memory!\n");
            exit(EXIT_FAILURE);
        }

        //read data file
        read_file(trainFile, N, A, trainData, trainClass);
        read_file(testFile, M, A, testData, testClass);

        printf("nome file train %s \n", trainFile);
        printf("nome file test %s \n", testFile);

        double pre_time = MPI_Wtime();
        double tempo_passato = pre_time - time_init;
        printf("Inizializzazione e lettura dati effetuata in %f  \n", tempo_passato);
        
        //invio dati di training
        MPI_Bcast(trainData, N * A, MPI_FLOAT, root_rank, MPI_COMM_WORLD);
        MPI_Bcast(trainClass, N, MPI_UINT8_T, root_rank, MPI_COMM_WORLD);

        //dati locali che il processo root deve gestire
        float * localtestData = (float *) malloc(countsRow[rank] * sizeof(float));
        uint8_t * localtestClass = (uint8_t*)  malloc(countsRow[rank] / A * sizeof(uint8_t));   

        if(localtestData == NULL || localtestClass == NULL){
            printf("Memoria insufficiente!\n");
            exit(EXIT_FAILURE);
        }

        for (int i = 0; i < size; ++i)
        {
            printf("processo %d\n", i);
            printf("%d\n", countsRow[i]);
            printf("%d\n", displsRow[i]);
            printf("%d\n", countsClasses[i]);
            printf("%d\n", displsClasses[i]);
            printf("%d\n", testClass[displsClasses[i]]);
        }

        //Invio ad ogni processo la porzione di immagine a colori
        MPI_Scatterv(testData, countsRow, displsRow, MPI_FLOAT, localtestData, countsRow[root_rank], MPI_FLOAT, root_rank, MPI_COMM_WORLD);
        MPI_Scatterv(testClass, countsClasses, displsClasses, MPI_UINT8_T, localtestClass, countsClasses[root_rank], MPI_INT, root_rank, MPI_COMM_WORLD);

        //risultato delle classificazioni
        int* confusionMatrixLocal = (int*) malloc(sizeof(int) * LABELS * LABELS);
        if(confusionMatrixLocal == NULL){
            printf("Memoria insufficiente!\n");
            exit(EXIT_FAILURE);
        }

        //calcolo della matrice di confusione per la porzione che il processo deve
        int localError = 0;
        localError = localKnn(
            trainData, 
            localtestData, 
            trainClass, 
            localtestClass, 
            countsRow[rank] / A, 
            confusionMatrixLocal,
            N,
            M,
            LABELS,
            A,
            K);
            
        int* confusionMatrix = (int*) malloc(sizeof(int) * LABELS * LABELS);
        MPI_Reduce(confusionMatrixLocal, confusionMatrix, LABELS*LABELS, MPI_INT, MPI_SUM, root_rank, MPI_COMM_WORLD);

        //recupero errori locali ogni processo
        int errors = 0;
        MPI_Reduce(&localError, &errors, 1 , MPI_INT, MPI_SUM, root_rank, MPI_COMM_WORLD);

        //MPI_Barrier(MPI_COMM_WORLD);
        printConfusionMatrix(confusionMatrix, LABELS);

        //dealloco memoria 
        free(trainData); trainData = NULL;
        free(testData); testData = NULL;
        free(trainClass); trainClass = NULL;
        free(testClass); testClass = NULL;
        free(localtestData); testData = NULL;
        free(localtestClass); testClass = NULL;
        free(confusionMatrixLocal); confusionMatrixLocal = NULL;
        free(confusionMatrix); confusionMatrix = NULL;
    
        double final_time = MPI_Wtime();
        double elapsed_time = final_time - time_init;
        printf("Tempo totale esecuzione %f  \n", elapsed_time);
         
        //save on file 
        saveResultsOnFile(elapsed_time, size, K, N, M);
    }
        
    //gestione processo non root
    else
        {   
            float * trainData = (float *) malloc(N * A * sizeof(float));
            uint8_t * trainClass = (uint8_t*) malloc(N *sizeof(uint8_t));
            
            float * localtestData = (float *) malloc(countsRow[rank] * sizeof(float));
            uint8_t * localtestClass = (uint8_t*)  malloc(countsClasses[rank] *sizeof(uint8_t));

            if(trainData == NULL || localtestData == NULL || localtestClass == NULL || trainClass == NULL){
                printf("Not enough memory!\n");
                exit(EXIT_FAILURE);
            }   
            
            //Ricevo dati di training
            MPI_Bcast(trainData, N * A, MPI_FLOAT, root_rank, MPI_COMM_WORLD);
            MPI_Bcast(trainClass, N, MPI_UINT8_T, root_rank, MPI_COMM_WORLD);

            //ricezione porzione di dati di testing da trattare
            MPI_Scatterv(NULL, NULL, NULL, NULL, localtestData, countsRow[rank], MPI_FLOAT, root_rank, MPI_COMM_WORLD);
            MPI_Scatterv(NULL, NULL, NULL, NULL, localtestClass, countsClasses[rank], MPI_UINT8_T, root_rank, MPI_COMM_WORLD);

            int* confusionMatrixLocal = (int*) malloc(sizeof(int) * LABELS * LABELS);
            if(confusionMatrixLocal == NULL){
                printf("Not enough memory!\n");
                exit(EXIT_FAILURE);
            }

            int localError = 0;
            localError = localKnn(
                trainData, 
                localtestData, 
                trainClass, 
                localtestClass, 
                countsRow[rank] / A, 
                confusionMatrixLocal,
                N,
                M,
                LABELS,
                A,
                K);
            
            MPI_Reduce(confusionMatrixLocal, NULL, LABELS * LABELS, MPI_INT, MPI_SUM, root_rank, MPI_COMM_WORLD);

            MPI_Reduce(&localError, NULL, 1 , MPI_INT, MPI_SUM, root_rank, MPI_COMM_WORLD);

            //MPI_Barrier(MPI_COMM_WORLD);
            //dealloco memoria
            free(trainData); trainData = NULL;
            free(trainClass); trainClass = NULL;
            free(localtestData); localtestData = NULL;
            free(localtestClass); localtestClass = NULL;
            free(confusionMatrixLocal); confusionMatrixLocal = NULL;
        }

}