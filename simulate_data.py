import numpy

def simulate()
	numpy.random.seed(12)
	num_observations = 5000
	
	x = numpy.random.multivariate_normal([0, 0], [[1, 0.75], [0.75, 1]], num_observations)
	y = numpy.random.multivariate_normal([1, 4], [[1, 0.75], [0.75, 1]], num_observations)

	simulated_separableish_features = numpy.vstack((x, y)).astype(numpy.float32)
	simulated_label = numpy.hstack((numpy.zero(num_observations), numpy.one(num_observations)))

	return [simulated_separableish_features, simulated_label]