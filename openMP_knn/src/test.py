import subprocess 
from tqdm import tqdm

N = [700, 3500, 7000, 10500]
M = [300, 1500, 3000, 4500]

K = 5
NP = [2, 4, 8, 16, 32]

for j in range(len(NP)):
	for i in range(len(N)):
		trainFile = "../../dataset/train_" + str(N[i])
		testFile= "../../dataset/test_" + str(M[i])
		command = "./main.exe {} {} {} {} {} {}".format(trainFile, testFile, N[i], M[i], K, NP[j])
		print(command)
		subprocess.call(command, shell= True)
