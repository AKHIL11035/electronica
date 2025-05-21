import numpy as np
import matplotlib.pyplot as plt

# Parameters
mu = 0.01
filter_order = 32

# Load input files
s = np.loadtxt('input_signal.txt')  # Clean signal
h = np.loadtxt('unknown_imp_res.txt')  # Impulse response
s = s / np.max(np.abs(s))
h = h / np.max(np.abs(h))

# Generate distorted signal x(n)
x_full = np.convolve(s, h)
N = len(s)
x = x_full[:N]

# Preallocate
y = np.zeros(N)
e = np.zeros(N)
W = np.zeros(filter_order)

# LMS Adaptive filtering
for n in range(filter_order, N):
    x_vec = x[n - filter_order + 1 : n + 1][::-1]
    y[n] = np.dot(W, x_vec)
    e[n] = s[n] - y[n]
    W = W + mu * x_vec * e[n]


# Performance metric
mse = np.mean(e[filter_order:] ** 2)
print(f"Final MSE = {mse:.4e}")

# Plotting
plt.figure(figsize=(10, 8))

plt.subplot(3, 1, 1)
plt.plot(s)
plt.title('Clean signal s(n)')
plt.xlabel('Sample index'); plt.ylabel('Amplitude')

plt.subplot(3, 1, 2)
plt.plot(x)
plt.title('Distorted signal x(n)')
plt.xlabel('Sample index'); plt.ylabel('Amplitude')

plt.subplot(3, 1, 3)
plt.plot(y)
plt.title('Recovered signal y(n) via LMS')
plt.xlabel('Sample index'); plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()

plt.figure()
plt.plot(e[filter_order:])
plt.title('Error signal e(n)')
plt.xlabel('Sample index'); plt.ylabel('Error')
plt.show()

# Save results
np.savetxt('recovered_signal.txt', y)
np.savetxt('error_signal.txt', e)

