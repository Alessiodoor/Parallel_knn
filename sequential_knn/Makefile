main: main.o datasetFunctions.o knn.o
	gcc -std=c99 -o main.exe main.o datasetFunctions.o knn.o -l cJSON

main.o:
	gcc -std=c99 -c main.c -o main.o 

datasetFunctions.o: datasetFunctions.c datasetFunctions.h 
	gcc -std=c99 -c datasetFunctions.c -o datasetFunctions.o

knn.o:
	gcc -std=c99 -c knn.c -o knn.o

clean:
	rm -f *.o
	rm main

.PHONY: main

rebuild:
	$(MAKE) clean
	$(MAKE) main