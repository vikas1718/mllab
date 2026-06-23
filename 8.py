import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
X, y = data.data, data.target

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier().fit(Xtr, ytr)
pred = clf.predict(Xte)

print("Accuracy:", accuracy_score(yte, pred))

print("Prediction:", "Benign" if clf.predict([Xte[0]])[0] == 1 else "Malignant")

plt.figure(figsize=(10,6))
plot_tree(clf, feature_names=data.feature_names, class_names=data.target_names, filled=True)
plt.show()