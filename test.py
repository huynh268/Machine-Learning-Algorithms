from simulate_data import simulate
from plot_data import plot

data = simulate();
plot(data[0], data[1])
plt.show()