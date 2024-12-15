import numpy as np
from collections import Counter
from sklearn.neighbors import BallTree

class KNNScratch:
    def __init__(self, neighbors=5, metric='euclidean', p=2):
        self.neighbors = neighbors
        self.metric = metric
        self.p = p

    def fit(self, X_train, y_train):
        self.X_train = X_train.to_numpy() if hasattr(X_train, 'to_numpy') else X_train
        self.y_train = y_train.to_numpy() if hasattr(y_train, 'to_numpy') else y_train

        self.tree = BallTree(self.X_train, metric=self.metric)

    def predict(self, X_test):
        X_test = X_test.to_numpy() if hasattr(X_test, 'to_numpy') else X_test

        predictions = []
        for x_test in X_test:
            distances, indices = self.tree.query([x_test], k=self.neighbors)
            k_nearest_labels = self.y_train[indices[0]] 
            most_common = Counter(k_nearest_labels).most_common(1)
            predictions.append(most_common[0][0])

        return np.array(predictions)