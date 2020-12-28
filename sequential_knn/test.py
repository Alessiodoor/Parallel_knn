import subprocess 
from tqdm import tqdm

N = [700, 3500, 7000, 10500]
M = [300, 1500, 3000, 4500]

K = 5

for i in tqdm(range(len(N))):
	trainFile = "../dataset/train_" + str(N[i])
	testFile= "../dataset/test_" + str(M[i])
	command = "./main.exe {} {} {} {} {}".format(trainFile, testFile, N[i], M[i], K)
	print(command)
	subprocess.call(command, shell= True)
