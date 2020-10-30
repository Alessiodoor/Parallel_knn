main: main.o datasetFunctions.o knn.o
	mpicc -std=c99 -o main.exe main.o datasetFunctions.o knn.o -lm -I/usr/local/include -L/usr/local/lib -Wl,-rpath=/usr/local/lib -l cjson

main.o:
	mpicc -std=c99 -c main.c -o main.o -lm

datasetFunctions.o: datasetFunctions.c datasetFunctions.h 
	mpicc -std=c99 -c datasetFunctions.c -o datasetFunctions.o -lm

knn.o:
	mpicc -std=c99 -c knn.c -o knn.o -lm

clean:
	rm -f *.o
	rm main.exe

.PHONY: main

rebuild:
	$(MAKE) clean
	$(MAKE) main