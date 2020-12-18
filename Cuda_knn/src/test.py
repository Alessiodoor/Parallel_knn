import subprocess

N = [700, 3500, 7000, 10500]
M = [300, 1500, 3000, 4500]
K = 5
LABELS = 10
BLOCK_SIZE = [2, 4, 8, 16, 32]

#lines = ["#ifndef INPUT\n", "#define INPUT\n", "#include <stdlib.h>\n", "#include <stdio.h>\n", "", "", "","", "", "#define LABELS 10\n", "#endif\n"]
for j in range(len(BLOCK_SIZE)):
	for i in range(len(N)):
		'''
		lines[4] = "#define N {}\n".format(N[i])
		lines[5] = "#define P {}\n".format(P[i])
		lines[6] = "#define K {}\n".format(K)
		lines[7] = "#define BLOCK_SIZE {}\n".format(BLOCK_SIZE[j])
		lines[8] = "#define M {}\n".format(M)
		with open("input.h", "w") as f:
			for x in lines:
				f.write(x)
		'''
		trainFile = "../../dataset/train_" + str(N[i])
		testFile= "../../dataset/test_" + str(M[i])

		print("TEST WITH = ", N[i])
		print("BLOCK_SIZE = ", BLOCK_SIZE[j])

		#subprocess.check_output(["make", "clean"])
		#subprocess.check_output(["make"])
		command = "./main.exe {} {} {} {} {} {}".format(trainFile, testFile, N[i], M[i], K, BLOCK_SIZE[j])
		print(command)
		
		subprocess.call(command, shell= True)

