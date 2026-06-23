import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = fetch_california_housing(as_frame=True).frame

# Histograms
df.hist(figsize=(12,8))
plt.show()

# Boxplots
for col in df.columns:
    sns.boxplot(x=df[col])
    plt.title(col)
    plt.show()

# Outlier Detection using IQR
for col in df.columns:
    Q1, Q3 = df[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
    print(col, ":", len(outliers), "outliers")

# Summary Statistics
print(df.describe())