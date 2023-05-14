import numpy as np
import matplotlib.pyplot as plt

# Generate time data
time = np.arange(0, 10, 0.01)

# Generate Gaussian wave
amplitude = np.exp(-(time - 5) ** 2 / 4)

# Plotting the Gaussian wave
plt.plot(time, amplitude)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Gaussian Wave')
plt.grid(True)
plt.show()
