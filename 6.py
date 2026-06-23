import numpy as np
import matplotlib.pyplot as plt

def lw(x, X, y, tau):
    w = np.exp(-np.sum((X - x)**2, axis=1) / (2 * tau**2))
    W = np.diag(w)
    theta = np.linalg.pinv(X.T @ W @ X) @ X.T @ W @ y
    return x @ theta

np.random.seed(42)
X = np.linspace(0, 2*np.pi, 100)
y = np.sin(X) + 0.1*np.random.randn(100)

Xb = np.c_[np.ones(len(X)), X]
xt = np.linspace(0, 2*np.pi, 200)
xtb = np.c_[np.ones(len(xt)), xt]

y_pred = [lw(xi, Xb, y, 0.5) for xi in xtb]

plt.scatter(X, y, c='r')
plt.plot(xt, y_pred, 'b')
plt.title("Locally Weighted Regression")
plt.show()