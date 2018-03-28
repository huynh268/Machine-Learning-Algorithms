import numpy

#Logistic Regression function
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

#Calculate the gradient of the log-likelihood
def log_likelihood_gradient(features, target, weights):
	scores = numpy.dot(features, weights)
	predictions = sigmoid(scores)
	error = target - predictions
	return numpy.dot(features.T, error)

#Calculate the log-likelihood
def log_likelihood(feartures, target, weights):
	scores = numpy.dot(feartures, weights)
	logLikelihood = numpy.sum(target*scores - numpy.log(1 + numpy.exp(scores)))
	return logLikelihood

#Sigmoid function
def sigmoid(scores):
	return 1/(1 + numpy.exp(-scores));