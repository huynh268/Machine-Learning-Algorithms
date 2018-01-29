import numpy 

def sigmoid(scores):
	return 1/(1 + numpy.exp(-scores));