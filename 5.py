import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

# Generate 100 random values
np.random.seed(42)
x = np.random.rand(100)

# Training and Testing data
train = x[:50]
test = x[50:]

# Labels
labels = ["Class1" if i <= 0.5 else "Class2" for i in train]

# KNN Function
def knn(point, k):
    d = sorted([(abs(point - train[i]), labels[i]) for i in range(50)])
    return Counter([label for _, label in d[:k]]).most_common(1)[0][0]

# Test for different k values
for k in [1, 2, 3, 4, 5, 20, 30]:
    print(f"\nk = {k}")
    for p in test[:5]:
        print(f"{p:.3f} -> {knn(p, k)}")

# Graph (using k = 5)
k = 5
pred = [knn(p, k) for p in test]

plt.scatter(test, [1 if p == "Class2" else 0 for p in pred])
plt.axvline(0.5, color='r', linestyle='--')
plt.yticks([0, 1], ['Class1', 'Class2'])
plt.xlabel("x value")
plt.ylabel("Predicted Class")
plt.title(f"KNN Classification (k={k})")
plt.grid(True)
plt.show()