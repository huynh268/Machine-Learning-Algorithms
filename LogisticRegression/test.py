import sys
sys.path.append('E:/Github/Machine-Learning-Algorithms/SimulateData')
import simulate_data
import plot_data
from logistic_regression import logistic_regression

data = simulate_data.simulate()

weights = logistic_regression(features = data[0], target = data[1], steps = 300000, learning_rate = 5e-5, add_intercept=True)

print(weights)

plot_data.plot(data[0], data[1])