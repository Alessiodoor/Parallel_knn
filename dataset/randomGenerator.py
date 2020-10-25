#necessario aver installato numpy e pandas per utilizzare questo generatore
from random import seed
from random import random
from random import randint 
import pandas as pd
import numpy as np
import sys

''' 
Tramite questo script vengono generati casualmente i sample di train e test 
che verranno utilizzati durante l'esecuzione dell'algoritmo Knn
'''

# argomenti da riga di comando:
# numero di smaple di train
# numero di sample di test
if len(sys.argv) == 3:
	train_size = int(sys.argv[1]);
	test_size = int(sys.argv[2]);
else:
	print("Inserisci gli argomenti corretti")
	sys.exit()

# funzione per generare i sample casuali di train e test
# parametri:
# fileName: nome del file di destinazione
# size: numero di sample da generare
# seedNum: valore da assegnare alla generazione del seed
def generateSamples(fileName, size, seedNum):
	# seed 
	if(seedNum == -1):
		seed()
	else:
		seed(seedNum);
	df = pd.DataFrame(columns = range(0, 31))
	print("Inizio generazione " + fileName + ", samples: " + str(size))
	for i in range (0, size):
		tmp_list = []
		for j in range(0, 30):
			value = random()
			roundValue = np.around(value, decimals = 6) 
			tmp_list.append(roundValue)
		label = randint(0, 9)
		tmp_list.append(label)
		df.loc[i] = tmp_list

	df.to_csv("./" + fileName + "_" + str(size), index = False, header = False, sep = " ")

# chiamo la funzione generateSample per creare train e test
generateSamples("train", train_size, -1)
generateSamples("test", test_size, 42)
