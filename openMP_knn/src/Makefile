main.exe: main.o datasetFunctions.o knnOMP.o
	gcc -fopenmp -std=c99 -o main.exe main.o datasetFunctions.o knnOMP.o -lm 
datasetFunctions.o: datasetFunctions.c datasetFunctions.h 
	gcc -std=c99 -c datasetFunctions.c -o datasetFunctions.o -lm

knnOMP.o:
	gcc -std=c99 -c knnOMP.c -o knnOMP.o -lm

clean:
	rm -f *.o
	rm main.exe

#-I/usr/local/include -L/usr/local/lib -Wl,-rpath=/usr/local/lib -l cjson