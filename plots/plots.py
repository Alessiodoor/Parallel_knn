import matplotlib.pyplot as plt
import json
import pprint as pp
from matplotlib.lines import Line2D

# plot del singolo metodo con diverso numero di processi e un solo trianSize
def plotSingleProgram(np_list, dataSize, data, title, label):
	x_list = [0, 0.1, 0.2, 0.3, 0.4]
	x_ticks_pos = [0.25, 1.25, 2.25, 3.25]
	colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:red']

	plt.figure()
	plt.title(title)
	for i in range(4):
		height = []
		for np in np_list:
			height.append(data[np][i])

		pos = []
		for x in x_list:
			pos.append(x + i)

		plt.bar(
			pos, 
			height, 
			width = 0.1, 
			color = colors)

	# custom legend
	custom_lines = []
	for color in colors:
		custom_lines.append(Line2D([0], [0], color=color, lw=4))
	plt.legend(custom_lines, [label + '2', label + '4', label + '8', label + '16', label + '32'])
	
	plt.xticks(x_ticks_pos, dataSize)
	plt.xlabel("Dimensione del dataset")
	plt.ylabel("Tempo d'esecuzione")
	plt.yscale('log')
	plt.show()

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
def plotPrograms(methods, labels, dataSize, title):
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
	plt.show()

# per ogni algoritmo calcolo lo speedUp di ogni dimensione di train con il numero ottimale di processi
def getSpeedUp(parallel_times, sequential_times):
	speedUp = []

	for i in range(len(parallel_times)):
		speedUp.append(sequential_times[i] / parallel_times[i])

	return speedUp

# plot lo speedUp di ogni algoritmo per ogni dimensione di train
def plotSpeedUp(speedUpMPI, speedUpOpenMP, dataSize):
	labels = ['MPI', 'OpenMp']
	x_ticks_pos = [0.15, 0.75]
	colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple']

	plt.figure()

	x_list = [0, 0.1, 0.2, 0.3]
	plt.bar(x_list, speedUpMPI, width = 0.1, color = colors)
	x_list = [0.6, 0.7, 0.8, 0.9]
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
	
	plt.ylabel("SpeedUp")

	plt.show()

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
dataSize = [1000, 5000, 10000, 15000]

mpi = {}
mpi['2'] = [0.14, 18.34, 129.95, 705.85]
mpi['4'] = [0.15, 15.3, 150.86, 550.98]
mpi['8'] = [0.45, 20.51, 157.85, 427.13]
mpi['16'] = [0.55, 21.63, 137.48, 578.34]
mpi['32'] = [0.69, 17.29, 118.94, 390.15]
#best np=32

openMp = {}
openMp['2'] = [0.049, 0.76, 3.13, 6.55]
openMp['4'] = [0.052, 0.76, 3.06, 6.51]
openMp['8'] = [0.06, 0.75, 3.03, 6.54]
openMp['16'] = [0.044, 0.83, 3.08, 6.66]
openMp['32'] = [0.046, 0.75, 3.04, 6.65]
#tutti simili

mpiPy = {}
mpiPy['2'] = [1.29, 32.33, 133.98, 294.32]
mpiPy['4'] = [1.34, 31.96, 125.91, 291.39]
mpiPy['8'] = [1.6, 32.63, 134.15, 296.9]
mpiPy['16'] = [2.18, 51.68, 200.77, 392.2]
mpiPy['32'] = [5.4, 66.3, 176.18, 331.97]
#best np = 16

sequential = {}
sequential['sequential'] = [0.05, 0.89, 3.3, 7.16]
# plotting single program 
# ogni linea np diverso
#plotSingleProgram(np_list, dataSize, mpi, "Confronto Mpi variando il numero di processi", "#process:")
#plotSingleProgram(np_list, dataSize, openMp, "Confronto OpenMp variando il numero di threads", "#thread:")
#plotSingleProgram(np_list, dataSize, mpiPy, "Confronto Mpi in python variando il numero di processi", "#process:")
#plotSingleProgram(["sequential"], dataSize, sequential, "Confronto Sequenziale", "")

# per ogni metodo np più efficiente
# ogni linea un metodo
bestMPI = mpi['32']
bestOpenMP = openMp['32']
bestMPIPy = mpiPy['16']
'''
labels = ['Mpi', 'OpenMp', 'Sequential']

best_list = {}
best_list['mpi'] = bestMPI
best_list['openMp'] = bestOpenMP
best_list['sequential'] = sequential['sequential']

plotPrograms(best_list, labels, dataSize, "Confronto con sequenziale")

# confronto con versione in python
labels = ['Mpi', 'OpenMp', 'Python']

best_list = {}
best_list['mpi'] = bestMPI
best_list['openMp'] = bestOpenMP
best_list['mpiPy'] = bestMPIPy

plotPrograms(best_list, labels, dataSize, "Confronto con python")
'''
speedUpMPI = getSpeedUp(bestMPI, sequential['sequential'])
speedUpOpenMP = getSpeedUp(bestOpenMP, sequential['sequential'])

plotSpeedUp(speedUpMPI, speedUpOpenMP, dataSize)