import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

X, y = fetch_olivetti_faces(shuffle=True, random_state=42, return_X_y=True)

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)

model = GaussianNB().fit(Xtr, ytr)
pred = model.predict(Xte)

print("Accuracy:", accuracy_score(yte, pred))
print(confusion_matrix(yte,pred))
print(classification_report(yte,pred,zero_division=0))

fig, ax = plt.subplots(2, 5)
for a, img, t, p in zip(ax.ravel(), Xte, yte, pred):
    a.imshow(img.reshape(64, 64), cmap="gray")
    a.set_title(f"T:{t} P:{p}")
    a.axis("off")

plt.show()