import numpy

def log_likelihood(feartures, target, weights):
	scores = numpy.dot(feartures, weights)
	logLikelihood = numpy.sum(target*scores - numpy.log(1 + numpy.exp(scores)))
	return logLikelihood