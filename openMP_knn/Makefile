
main.exe: main.o datasetFunctions.o
	gcc -fopenmp -std=c99 -o main.exe main.o datasetFunctions.o -lm -I/usr/local/include -L/usr/local/lib -Wl,-rpath=/usr/local/lib -l cjson

datasetFunctions.o: datasetFunctions.c datasetFunctions.h 
	gcc -std=c99 -c datasetFunctions.c -o datasetFunctions.o -lm

clean:
	rm -f *.o
	rm main.exe