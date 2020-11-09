import matplotlib.pyplot as plt
import json
import pprint as pp

def plotSingleProgram(data):
	np = []
	times = []
	title = 'K:' + str(data[0]['K']) + ', trainSize:' + str(data[0]['trainSize'])  + ', testSize:' + str(data[0]['testSize']) 
	for result in data:
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

def getBestResult(data):
	best = data[0]
	bestTime = data[0]['totalTime']
	for i in range(1, len(data)):
		if data[i]['totalTime'] < bestTime:
			bestTime = data[i]['totalTime']
			best = data[i]

	return best

def plotPrograms(mpi, openMp, sequential):
	times = []
	times.append(mpi['totalTime'])
	times.append(openMp['totalTime'])
	times.append(sequential['totalTime'])

	print(times)

	plt.figure()
	plt.bar(range(len(times)), times)
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

plotPrograms(bestMPI, bestOpenMP, sequential)