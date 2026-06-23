import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report

X, y = load_breast_cancer(return_X_y=True)

X = StandardScaler().fit_transform(X)
kmeans = KMeans(n_clusters=2, random_state=42).fit_predict(X)

print(confusion_matrix(y, kmeans))
print(classification_report(y, kmeans))

X_pca = PCA(2).fit_transform(X)

plt.scatter(X_pca[:,0], X_pca[:,1], c=kmeans, cmap='Set1')
plt.title("K-Means Clustering")
plt.show()

plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap='coolwarm')
plt.title("True Labels")
plt.show()