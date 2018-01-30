from log_likelihood import log_likelihood
from gradient import log_likelihood_gradient
import numpy

def logistic_regression(features, target, steps, learning_rate, add_intercept = False):
	if add_intercept:
		intercept = numpy.ones((features.shape[0],1))
		features = numpy.hstack((intercept,features))

	weights = numpy.zeros(features.shape[1])

	for step in range(steps):
		gradient = log_likelihood_gradient(features, target, weights)
		weights += learning_rate * gradient 

		if step % 1000 == 0:
			print(log_likelihood(features, target, weights))

	return weights

