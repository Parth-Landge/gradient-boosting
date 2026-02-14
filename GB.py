from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from dd import GradientBoostingRegressorScratch
import numpy as np
# Generate data
X, y = make_regression(n_samples=500, n_features=5, noise=0.1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = GradientBoostingRegressorScratch(n_estimators=50, learning_rate=0.1)
model.fit(X_train, y_train)

# Predict
preds = model.predict(X_test)
print(f"MSE: {np.mean((y_test - preds)**2)}")

