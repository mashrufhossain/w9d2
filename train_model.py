from sklearn.linear_model import LogisticRegression
import numpy as np
import joblib

# Simple dummy training data
X = np.array([
    [1, 2],
    [2, 1],
    [3, 4],
    [4, 3],
    [10, 10]
])

y = np.array([0, 0, 0, 1, 1])

model = LogisticRegression()
model.fit(X, y)

joblib.dump(model, "models/baseline.joblib")
print("Saved baseline.joblib")
