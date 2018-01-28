import matplotlib.pyplot as pyplot

def plot(simulated_separableish_features, simulated_labels):
	pyplot.figure(figsize = (12, 8))
	pyplot.scatter(simulated_separableish_features[:, 0], simulated_separableish_features[:, 1], 
					c = simulated_labels, alpha = 0.4)
	pyplot.show()
