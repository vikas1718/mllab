import pandas as pd

data = pd.read_csv("training_data.csv")
print(data.columns)
h = None

for _, row in data.iterrows():
    if row["PlayTennis"] == "Yes":
        if h is None:
            h = list(row[:-1])
        else:
            for i in range(len(h)):
                if h[i] != row.iloc[i]:
                    h[i] = "?"

print("Final Hypothesis:", h)