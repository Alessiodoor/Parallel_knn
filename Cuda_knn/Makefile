main.exe: main.o datasetFunctions.o knnCuda.o 

	nvcc -Xptxas= --use_fast_math -gencode arch=compute_35,code=sm_35 -o main.exe main.o datasetFunctions.o knnCuda.o 

datasetFunctions.o: datasetFunctions.cu datasetFunctions.h

	nvcc -Xptxas= --use_fast_math -gencode arch=compute_35,code=sm_35 -c datasetFunctions.cu -o datasetFunctions.o

knnCuda.o: knnCuda.cu knnCuda.h 

	nvcc -Xptxas= --use_fast_math -gencode arch=compute_35,code=sm_35 -c knnCuda.cu -o knnCuda.o

.PHONY: clean

clean:

	rm -f *.o main.exe