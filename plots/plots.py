import matplotlib.pyplot as plt
import json
import pprint as pp

# plot del singolo metodo con diverso numero di processi e un solo trianSize
def plotSingleProgram(data, trainSize = 64):
	np = []
	times = []
	title = 'K:' + str(data[0]['K']) + ', trainSize:' + str(data[0]['trainSize'])  + ', testSize:' + str(data[0]['testSize']) 
	for result in data:
		if result['trainSize'] == trainSize:
			if 'NP' in result: 
				np.append(result['NP'])
			elif 'numberTreads' in result: 
				np.append(result['numberTreads'])
			times.append(result['totalTime'])

	plt.figure()
	plt.title(title)
	plt.plot(range(len(np)), times)
	plt.xticks(range(len(np)), np)
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
def plotPrograms(mpi, openMp, sequential):
	labels = ['Mpi', 'OpenMp', 'Sequential']

	timesForSizes = {}
	for k in mpi:
		timesForSizes[k] = []
		timesForSizes[k].append(mpi[k]['totalTime'])
		timesForSizes[k].append(openMp[k]['totalTime'])
		timesForSizes[k].append(sequential[k]['totalTime'])

	print(timesForSizes)

	plt.figure()
	for k in timesForSizes:
		plt.bar(range(len(timesForSizes[k])), timesForSizes[k])

	plt.xticks(range(len(timesForSizes[64])), labels)
	plt.title("Tempo migliore per ogni dimensione di train")

	plt.show()

# per ogni algoritmo calcolo lo speedUp di ogni dimensione di train con il numero ottimale di processi
def getSpeedUp(parallel, sequential):
	speedUp = {}

	for size in parallel:
		speedUp[size] = (sequential[size]['totalTime'] / parallel[size]['totalTime'])

	return speedUp

# plot lo speedUp di ogni algoritmo per ogni dimensione di train
def plotSpeedUp(speedUpMPI, speedUpOpenMP):
	labels = ['MPI', 'OpenMp']

	speedUpForSizes = {}
	for k in speedUpMPI:
		speedUpForSizes[k] = []
		speedUpForSizes[k].append(speedUpMPI[k])
		speedUpForSizes[k].append(speedUpOpenMP[k])

	print(speedUpForSizes)

	plt.figure()
	for k in speedUpForSizes:
		plt.bar(range(len(speedUpForSizes[k])), speedUpForSizes[k])

	plt.xticks(range(len(speedUpForSizes[64])), labels)
	plt.title("Tempo migliore per ogni dimensione di train")

	plt.show()

with open('data/resultMPI.json') as f:
  mpi = json.load(f)

with open('data/resultOpenMP.json') as f:
  openMp = json.load(f)

with open('data/resultSequential.json') as f:
  sequential = json.load(f)

# plotting single program 
#plotSingleProgram(mpi)
#plotSingleProgram(openMp)

bestMPI = getBestResult(mpi)
bestOpenMP = getBestResult(openMp)
bestSequential = getBestResult(sequential)

#plotPrograms(bestMPI, bestOpenMP, bestSequential)

speedUpMPI = getSpeedUp(bestMPI, bestSequential)
speedUpOpenMP = getSpeedUp(bestOpenMP, bestSequential)

plotSpeedUp(speedUpMPI, speedUpOpenMP)