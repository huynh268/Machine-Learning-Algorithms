import numpy
from sigmoid import sigmoid

def log_likelihood_gradient(features, target, weights):
	scores = numpy.dot(features, weights)
	predictions = sigmoid(scores)
	error = target - predictions
	return numpy.dot(features.T, error)
