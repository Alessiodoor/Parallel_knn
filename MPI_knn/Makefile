main.exe: main.o knnMPI.o datasetFunctions.o 
	mpicc -std=c99 -o main.exe knnMPI.o main.o datasetFunctions.o -lm -I/usr/local/include -L/usr/local/lib -Wl,-rpath=/usr/local/lib -l cjson

main.o: main.c input.h 
	mpicc -std=c99 -c main.c -o main.o -lm

knnMPI.o: knnMPI.c knnMPI.h input.h datasetFunctions.o
	mpicc -std=c99 -c knnMPI.c -o knnMPI.o -lm

datasetFunctions.o: datasetFunctions.c datasetFunctions.h input.h 
	mpicc -std=c99 -c datasetFunctions.c -o datasetFunctions.o -lm

.PHONY: clean

clean:
	rm -f *.o main.exe

