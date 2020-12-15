main.exe: main.o datasetFunctions.o knnCuda.o 
	nvcc -o main.exe main.o datasetFunctions.o knnCuda.o 

main.o : main.cu
	nvcc -c main.cu -o main.o

datasetFunctions.o: datasetFunctions.cu datasetFunctions.h
	nvcc -c datasetFunctions.cu -o datasetFunctions.o

knnCuda.o: knnCuda.cu knnCuda.h 
	nvcc -c knnCuda.cu -o knnCuda.o

.PHONY: clean

clean:
	rm -f *.o main.exe