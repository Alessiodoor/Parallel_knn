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
Return: Numero di errori commessi dalla classificazione
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

        //distanza da ogni punto del training (dopo ordinamento i primi k saranno i vicini)
        float* k_distances = (float*) malloc(sizeof(float) * N); 
        
        //indice della label del sample di train (dopo ordinamento i primi k saranno le etichette dei K vicini)
        int * k_labels = (int*) malloc(sizeof(int) * N);

        //label di ogni sample per majority voting   
        int* countsLabel = (int*) malloc(sizeof(int) * LABELS);

        // controllo di aver allocato corretamente gli array
        if (countsLabel == NULL || k_distances == NULL || k_labels == NULL){
            printf("Not enough memory!\n");
            exit(EXIT_FAILURE);
        }
        
        // inizializzo la matrice di confusione locale con tutti 0
        for(int i = 0; i < LABELS * LABELS; i++){
            confusionMatrix[i] = 0;
        }
        
        //numero di errori compessi dall'algoritmo KNN
        int error = 0;
        
        // ciclo su i sample della porzione di test corrente
        for(int i = 0; i < NrecordTesting; i++){
            //calcolo la distanze euclidea con i sample di train
            for(int j = 0; j < N; j++){
                // distanza euclidea tra il sample di test i-esimo e quello di train j-esimo
                k_distances[j] = euclideanDist(&trainData[j * A], &localTestData[i * A], M);
                // salvo l'indice del sample di train 
                k_labels[j] = j;
            }
            
            //ordino le label in base alla distanza dai vari sample di train
            sort(k_distances, k_labels, N, K);
            
            //inizializza a zero l'array che conterrà la migliore label di ogni sample di test
            for(int i = 0; i < LABELS; i++){
                countsLabel[i] = 0;
            }

            int bestLabel = 0;
            
            // scrorro i k sample di train vicini al sample di test i-esimo
            for(int j = 0; j < K; j++){ 
                // salvo la classe del sample di train k-esimo
                int indice = k_labels[j];
                int label = trainClass[indice]; 

                // incremento il contatore della classe corrente
                countsLabel[label] = countsLabel[label] + 1;

                // se il numero di volte che occorre la classe "label" è maggiore di quelle della "bestlabel", 
                // salvo la nuova label migliore, ovvero quella con più occorrenze tra i vicini
                if(countsLabel[label] > countsLabel[bestLabel])
                    bestLabel = label;
            }

            // salvo la vera classe del sample di test i-esimo per fare la verifica
            int realLabel = localTestClass[i];

            // se la classe calcolata non corrisponde con quella vera incremento gli errori commessi dalla classificazione
            if (realLabel != bestLabel)
                error = error + 1;
            
            // aggiorno la matrice di confusione 
            confusionMatrix[realLabel * LABELS + bestLabel] = confusionMatrix[realLabel * LABELS + bestLabel] +1;

        }

        // libero la memoria utilizzata
        free(countsLabel); countsLabel = NULL; 
        free(k_distances); k_distances = NULL; 
        free(k_labels); k_labels = NULL;

        return error;
}


/*
Funzione che contiene l'esecuzione parallela dell'algoritmo knn,
In particolare, calcola e assegna le porzioni di test ai vari processi e condivide tutto il train a tutti i processi
La funzione è divisa in due gruppi di operazioni, quelle del processo root e quelle degli altri processi.
Il processo root inviera tramite un broadcast tutto il train a tutti i processi
Inoltre, invierà la porzione di test assegnata ad ogni processo tramite la funzione scatterv
Successivamente, ogni processo esegue l'algoritmo knn con i propri dati e calcola la matrice di confusione locale.
Infine, tramite la funzione Reduce verranno unificate le matrici di confusione e il numero di errori commessi durante la classificazione
Parametri:
const char* trainFile: percorso al file di train
const char* testFile: percorso al file di test
int size: numero di processi
int rank: rank del processo corrente
double time_init: tempo di inizio dell'esecuzione
int M: numero di sample di test
int N: numero di sample di train
int K: numero di vicini
int A: numero di attributi per sample
int LABELS: numero di possibili labels
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
    int LABELS){

    // 0 sarà il rank del processo root
    int root_rank = 0; 

    // i successivi quattro array servono per condividere la porzione di test tramite la funzione Scatterv

    //numero di elementi del test che ogni processo assegnati ad ogni processo
    int* countsRow = (int *) malloc(size * sizeof(int));

    //indirizzo di inizio della porzione di test assegnata ad ogni processo
    int* displsRow = (int *) malloc(size * sizeof(int)); 

    //numero di classi, relative ai sample di test, assegnati ad ogni processo
    int* countsClasses = (int *) malloc(size * sizeof(int)); 

    //indirizzi di inizio della porzione di classi assegnata ad ogni processo
    int* displsClasses = (int *) malloc(size * sizeof(int));

    // controllo di aver allocato corretamente gli array
    if(countsRow == NULL || displsRow == NULL || countsClasses == NULL || displsClasses == NULL){
        printf("Not enough memory!\n");
        exit(EXIT_FAILURE);
    }

    // calcolo le porzioni di test che verranno assegnate ad ogni processo e salvo i valori negli array inizializzati precedentemente
    splitData(countsRow, countsClasses, displsRow, displsClasses, size, A, M); 

    // se il processo corrente è il root 
    if(rank ==0){

        // alloco la memoria per l'intero dataset
        float * trainData = (float *) malloc(N * A * sizeof(float));
        float * testData = (float *) malloc(M * A * sizeof(float));

        uint8_t * trainClass = (uint8_t*) malloc(N * sizeof(uint8_t));
        uint8_t * testClass = (uint8_t*)  malloc(M * sizeof(uint8_t));

        // controllo di aver allocato correttamente gli array del dataset
        if(trainData == NULL || testData == NULL || testClass == NULL || trainClass == NULL){
            printf("Not enough memory!\n");
            exit(EXIT_FAILURE);
        }

        // leggo i file del dataset e li salvo nelle relative variabili
        read_file(trainFile, N, A, trainData, trainClass);
        read_file(testFile, M, A, testData, testClass);

        // ottengo il tempo attuale
        double pre_time = MPI_Wtime();

        // salvo il tempo impiegato per leggere i dati e inizializzare le matrici
        double tempo_passato = pre_time - time_init;
        
        // invio tutto il train ai processi tramite la funzione Broadcast, i riceventi la chiameranno la loro volta
        MPI_Bcast(trainData, N * A, MPI_FLOAT, root_rank, MPI_COMM_WORLD);
        MPI_Bcast(trainClass, N, MPI_UINT8_T, root_rank, MPI_COMM_WORLD);

        // creo le variabili che conterranno la porzione di test assegnata al processo corrente
        float * localtestData = (float *) malloc(countsRow[rank] * sizeof(float));
        uint8_t * localtestClass = (uint8_t*)  malloc(countsRow[rank] / A * sizeof(uint8_t));   

        // controllo di aver allocato correttamente le matrici
        if(localtestData == NULL || localtestClass == NULL){
            printf("Memoria insufficiente!\n");
            exit(EXIT_FAILURE);
        }

        // invio ad ogni processo la porzione di test 
        MPI_Scatterv(testData, countsRow, displsRow, MPI_FLOAT, localtestData, countsRow[root_rank], MPI_FLOAT, root_rank, MPI_COMM_WORLD);
        MPI_Scatterv(testClass, countsClasses, displsClasses, MPI_UINT8_T, localtestClass, countsClasses[root_rank], MPI_INT, root_rank, MPI_COMM_WORLD);

        // creo la matrice che conterrà la porzione locale della matrice di confusione e controllo la corretta allocazione
        int* confusionMatrixLocal = (int*) malloc(sizeof(int) * LABELS * LABELS);
        if(confusionMatrixLocal == NULL){
            printf("Memoria insufficiente!\n");
            exit(EXIT_FAILURE);
        }

        // eseguo l'algoritmo knn per il processo corrente e salvo la matrice di confusione 
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
            
        // alloco la matrice di confusione totale
        int* confusionMatrix = (int*) malloc(sizeof(int) * LABELS * LABELS);
        
        // ricostruisco la matrice di confusione totale
        // per ogni cella delle matrici locali sommo i valori ottenuti da tutti i processi
        MPI_Reduce(confusionMatrixLocal, confusionMatrix, LABELS*LABELS, MPI_INT, MPI_SUM, root_rank, MPI_COMM_WORLD);

        //rcalcolo gli errori commessi dalla classificazione sommando quelli calcolati da ogni processo
        int errors = 0;
        MPI_Reduce(&localError, &errors, 1 , MPI_INT, MPI_SUM, root_rank, MPI_COMM_WORLD);

        //MPI_Barrier(MPI_COMM_WORLD);
        printConfusionMatrix(confusionMatrix, LABELS);

        // libero la memoria 
        free(trainData); trainData = NULL;
        free(testData); testData = NULL;
        free(trainClass); trainClass = NULL;
        free(testClass); testClass = NULL;
        free(localtestData); testData = NULL;
        free(localtestClass); testClass = NULL;
        free(confusionMatrixLocal); confusionMatrixLocal = NULL;
        free(confusionMatrix); confusionMatrix = NULL;
    
        // ottengo il tempo finale
        double final_time = MPI_Wtime();

        // calcolo il tempo dell'esecuzione totale
        double elapsed_time = final_time - time_init;
         
        // salvo i risultati su un file json
        //writeResultJson(K, N, M, A, elapsed_time, size, "result_MPI.json");
        saveResultsOnFile(K, N, M, A, elapsed_time, size, "result_MPI.json");
    }
        
    // operazioni del processo non root
    else
        {   
            // alloco la memoria dedicata al train
            float * trainData = (float *) malloc(N * A * sizeof(float));
            uint8_t * trainClass = (uint8_t*) malloc(N *sizeof(uint8_t));
            
            // alloco la memoria dedicata alla porzione locale di test
            float * localtestData = (float *) malloc(countsRow[rank] * sizeof(float));
            uint8_t * localtestClass = (uint8_t*)  malloc(countsClasses[rank] *sizeof(uint8_t));

            // controllo di aver allocato correttamente le matrici
            if(trainData == NULL || localtestData == NULL || localtestClass == NULL || trainClass == NULL){
                printf("Not enough memory!\n");
                exit(EXIT_FAILURE);
            }   
            
            // ricevo il train dal processo root 
            MPI_Bcast(trainData, N * A, MPI_FLOAT, root_rank, MPI_COMM_WORLD);
            MPI_Bcast(trainClass, N, MPI_UINT8_T, root_rank, MPI_COMM_WORLD);

            // riveco la porzione locale di test dal processo root
            MPI_Scatterv(NULL, NULL, NULL, NULL, localtestData, countsRow[rank], MPI_FLOAT, root_rank, MPI_COMM_WORLD);
            MPI_Scatterv(NULL, NULL, NULL, NULL, localtestClass, countsClasses[rank], MPI_UINT8_T, root_rank, MPI_COMM_WORLD);

            // alloco la matrice di confusione locale e controllo la corretta allocazione
            int* confusionMatrixLocal = (int*) malloc(sizeof(int) * LABELS * LABELS);
            if(confusionMatrixLocal == NULL){
                printf("Not enough memory!\n");
                exit(EXIT_FAILURE);
            }

            // eseguo l'algoritmo knn con i dati del dataset dedicati al processo corrente
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
            
            // invio la porzione locale della matrice di confusione al root
            MPI_Reduce(confusionMatrixLocal, NULL, LABELS * LABELS, MPI_INT, MPI_SUM, root_rank, MPI_COMM_WORLD);

            // invio gli errori locale di classificazione al processo root
            MPI_Reduce(&localError, NULL, 1 , MPI_INT, MPI_SUM, root_rank, MPI_COMM_WORLD);

            //MPI_Barrier(MPI_COMM_WORLD);
            // libero la memoria
            free(trainData); trainData = NULL;
            free(trainClass); trainClass = NULL;
            free(localtestData); localtestData = NULL;
            free(localtestClass); localtestClass = NULL;
            free(confusionMatrixLocal); confusionMatrixLocal = NULL;
        }

}