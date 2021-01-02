# Parallel_knn
Parallel implementation of Knn in c

This repository in an Universitary project, three different implementation as been used: MPI, OpenMP and Cuda.

# Structure

The repository contains a folder for each technoligy, every folder contains the Dockerfile used to create the docker image of the project and the src folder with the code.
The src folder contains a test.py script uset to automate the testing process and a MakeFile used to compile the code, besides to the knn's files.
The dataset folder contains all datasets osed during the tests and a python script used to create the samples.

.
├── sequential_knn
    ├── Makefile
    └── test.py
├── Cuda_knn
│   ├── Dockerfile
│   └── src
│       ├── Makefile
│       └── test.py
├── MPI_knn
│   ├── Dockerfile
│   └── src
│       ├── Makefile
│       └── test.py
├── openMP_knn
│   ├── Dockerfile
│   └── src
│       ├── Makefile
│       └── test.py
├── README.md
├── Risultati
├── dataset
│   ├── randomGenerator.py
├── plots
│   ├── plots.py
└── pythonKnn
