import numpy as np
from sklearn.tree import DecisionTreeRegressor

class GradientBoostingRegressorScratch:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.initial_prediction = None

    def fit(self, X, y):
        self.initial_prediction = np.mean(y)
        f_m = np.full(len(y), self.initial_prediction)

        for s in range(self.n_estimators):
            residuals = y - f_m
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            predictions_of_residuals = tree.predict(X)
            f_m += self.learning_rate * predictions_of_residuals
            self.trees.append(tree)

    def predict(self, X):
        y_pred = np.full(X.shape[0], self.initial_prediction)
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred

