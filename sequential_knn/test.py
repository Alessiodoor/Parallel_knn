import subprocess 
from tqdm import tqdm

N = [700, 3500, 7000, 10500]
M = [300, 1500, 3000, 4500]

K = 5

#subprocess.call("rm resultsKNN_mpi.out", shell= True)
#lines = ["#ifndef INPUT\n", "#define INPUT\n", "#include <stdlib.h>\n", "#include <stdio.h>\n", "", "", "","", "#define LABELS 10\n", "typedef enum {true, false} bool;\n", "#endif\n"]
for i in tqdm(range(len(N))):
	trainFile = "../dataset/train_" + str(N[i])
	testFile= "../dataset/test_" + str(M[i])
	'''
	lines[4] = "#define M {}\n".format(M)
	lines[5] = "#define N {}\n".format(N[i])
	lines[6] = "#define P {}\n".format(P[i])
	lines[7] = "#define K {}\n".format(K)
	with open("input.h", "w") as f:
		for x in lines:
			f.write(x)'''
	#print("Test con ", N[i],)
	#print("con NP ", NP[j])
	#for x in range(10):
		#subprocess.check_output(["make", "clean"])
	#subprocess.check_output(["make"])
	command = "./main.exe {} {} {} {} {}".format(trainFile, testFile, N[i], M[i], K)
	print(command)
	#subprocess.check_output(["make clean"])
	subprocess.call(command, shell= True)
