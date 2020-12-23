import matplotlib.pyplot as plt
import json
import pprint as pp
from matplotlib.lines import Line2D

# plot del singolo metodo con diverso numero di processi e un solo trianSize
def plotSingleProgram(np_list, dataSize, data, title, label, fileName):
	x_list = [0, 0.1, 0.2, 0.3, 0.4]
	x_ticks_pos = [0.25, 1.25, 2.25, 3.25, 4.25]
	colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:red']

	plt.figure()
	plt.title(title)

	for i in range(len(data)):
		height = []
		for np in np_list:
			height.append(data[np][i])

		pos = []
		for x in x_list:
			pos.append(x + i)


		plt.bar(pos, height, width = 0.1, color = colors)

	# custom legend
	custom_lines = []
	for color in colors:
		custom_lines.append(Line2D([0], [0], color=color, lw=4))
	plt.legend(custom_lines, [label + '2', label + '4', label + '8', label + '16', label + '32'])
	
	plt.xticks(x_ticks_pos, dataSize)
	plt.xlabel("Dimensione del dataset")
	plt.ylabel("Tempo d'esecuzione")
	plt.yscale('log')
	#plt.show()
	plt.savefig(fileName + ".pdf")
	plt.savefig(fileName + ".png")

# restitusice il tempo migliore per ogni dimensiose del train
def getBestResult(data, trainSizes = [64]):
	best = {}
	for size in trainSizes:
		best[size] = {}
		bestTime = 9999999
		for result in data:
			if result['trainSize'] == size:
				if result['totalTime'] < bestTime:
					bestTime = result['totalTime']
					best[size] = result

	return best

# plot del tempo migliore degli algoritmi per ogni dimemsione di train
# ATTENZIONE: il numero di processo è diverso ognuno di essi
def plotPrograms(methods, labels, dataSize, title, fileName):
	'''
	timesForSizes = {}
	for k in mpi:
		timesForSizes[k] = []
		timesForSizes[k].append(mpi[k]['totalTime'])
		timesForSizes[k].append(openMp[k]['totalTime'])
		timesForSizes[k].append(sequential[k]['totalTime'])

	print(timesForSizes)
	'''
	plt.figure()
	plt.title(title)
	for k in methods:
		times = methods[k]
		line, = plt.plot(range(len(times)), times, label = k)

	plt.xticks(range(len(dataSize)), dataSize)
	plt.xlabel("Dimensione del dataset")
	plt.ylabel("Tempo d'esecuzione")
	plt.yscale("log")
	plt.legend()
	#plt.show()
	plt.savefig(fileName + ".pdf")
	plt.savefig(fileName + ".png")

# per ogni algoritmo calcolo lo speedUp di ogni dimensione di train con il numero ottimale di processi
def getSpeedUp(parallel_times, sequential_times):
	speedUp = []

	for i in range(len(parallel_times)):
		speedUp.append(sequential_times[i] / parallel_times[i])

	return speedUp

# plot lo speedUp di ogni algoritmo per ogni dimensione di train
def plotSpeedUp(speedUpMPI, speedUpOpenMP, dataSize, fileName):
	labels = ['MPI', 'OpenMp']
	x_ticks_pos = [0.15, 0.75]
	colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:red']

	plt.figure()

	x_list = [0, 0.1, 0.2, 0.3, 0.4]
	plt.bar(x_list, speedUpMPI, width = 0.1, color = colors)
	x_list = [0.6, 0.7, 0.8, 0.9, 1]
	plt.bar(x_list, speedUpOpenMP, width = 0.1, color = colors)

	plt.xticks(x_ticks_pos, labels)
	plt.title("SpeedUp rispetto alla versione sequenziale")

	# custom legend
	custom_lines = []
	for color in colors:
		custom_lines.append(Line2D([0], [0], color=color, lw=4))

	labels = []
	for size in dataSize:
		labels.append("dataSize:" + str(size))
	plt.legend(custom_lines, labels)
	plt.yscale("log")
	#plt.ylim(-1, 3.5)
	
	plt.ylabel("SpeedUp")

	#plt.show()
	plt.savefig(fileName + ".pdf")
	plt.savefig(fileName + ".png")

'''
with open('data/resultMPI.json') as f:
  mpi = json.load(f)

with open('data/resultOpenMP.json') as f:
  openMp = json.load(f)

with open('data/resultSequential.json') as f:
  sequential = json.load(f)
'''

#N = [700, 3500, 7000, 10500]
#M = [300, 1500, 3000, 4500]

np_list = ['2', '4', '8', '16', '32']
dataSize = [1000, 5000, 10000, 15000, 40000]

mpi = {}
mpi['2'] = [0.03, 0.41, 2.05, 4.05, 23.48]
mpi['4'] = [0.04, 0.42, 1.89, 3.22, 21.85]
mpi['8'] = [0.11, 0.64, 2.7, 4.61, 22.9]
mpi['16'] = [0.27, 1.28, 3.52, 5.02, 25.93]
mpi['32'] = [0.38, 1.71, 4.54, 7.14, 30.55]
#best np=8

openMp = {}
openMp['2'] = [0.049, 0.76, 3.13, 6.55, 49.03]
openMp['4'] = [0.052, 0.76, 3.06, 6.51, 44.46]
openMp['8'] = [0.06, 0.75, 3.03, 6.54, 43.71] 
openMp['16'] = [0.044, 0.83, 3.08, 6.66, 43.53] 
openMp['32'] = [0.046, 0.75, 3.04, 6.65, 43.61] 
#tutti simili

mpiPy = {}
mpiPy['2'] = [1.29, 32.33, 133.98, 294.32]
mpiPy['4'] = [1.34, 31.96, 125.91, 291.39]
mpiPy['8'] = [1.6, 32.63, 134.15, 296.9]
mpiPy['16'] = [2.18, 51.68, 200.77, 392.2]
mpiPy['32'] = [5.4, 66.3, 176.18, 331.97]
#best np = 16

sequential = {}
sequential['sequential'] = [0.05, 0.89, 3.3, 7.16, 51.54]
# plotting single program 
# ogni linea np diverso
#plotSingleProgram(np_list, dataSize, mpi, "Confronto Mpi variando il numero di processi", "#process:", "confrontoMpi")
#plotSingleProgram(np_list, dataSize, openMp, "Confronto OpenMp variando il numero di threads", "#thread:", "confrontoOpenMp")
#plotSingleProgram(np_list, dataSize, mpiPy, "Confronto Mpi in python variando il numero di processi", "#process:", "confrontoMpiPy")
#plotSingleProgram(["sequential"], dataSize, sequential, "Confronto Sequenziale", "")

bestMPI = mpi['4']
bestOpenMP = openMp['8']
bestMPIPy = mpiPy['16']


# per ogni metodo np più efficiente
# ogni linea un metodo
labels = ['Mpi', 'OpenMp', 'Sequential']

best_list = {}
best_list['mpi'] = bestMPI
best_list['openMp'] = bestOpenMP
best_list['sequential'] = sequential['sequential']

#plotPrograms(best_list, labels, dataSize, "Confronto con sequenziale", "confrontoAll")

'''
# confronto con versione in python
labels = ['Mpi', 'OpenMp', 'Python']

best_list = {}
best_list['mpi'] = bestMPI
best_list['openMp'] = bestOpenMP
best_list['mpiPy'] = bestMPIPy

plotPrograms(best_list, labels, dataSize, "Confronto con python")
'''

# plt speedup
speedUpMPI = getSpeedUp(bestMPI, sequential['sequential'])
speedUpOpenMP = getSpeedUp(bestOpenMP, sequential['sequential'])

print(speedUpMPI)
print(speedUpOpenMP)
#plotSpeedUp(speedUpMPI, speedUpOpenMP, dataSize, "speedup")