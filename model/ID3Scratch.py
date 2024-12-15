import numpy as np

class DecisionTreeClassifierScratch:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def gini(self, y):
        classes, counts = np.unique(y, return_counts=True)
        p = counts / len(y)
        return 1 - np.sum(p ** 2)

    def split(self, X, y, feature_index, threshold):
        left_idxs = X[:, feature_index] <= threshold
        right_idxs = X[:, feature_index] > threshold
        return X[left_idxs], X[right_idxs], y[left_idxs], y[right_idxs]

    def best_split(self, X, y):
        best_gini = float('inf')
        best_index, best_threshold = None, None

        for feature_index in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                X_left, X_right, y_left, y_right = self.split(X, y, feature_index, threshold)

                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                gini_left = self.gini(y_left)
                gini_right = self.gini(y_right)
                weighted_gini = (len(y_left) * gini_left + len(y_right) * gini_right) / len(y)

                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_index = feature_index
                    best_threshold = threshold

        return best_index, best_threshold

    def build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if n_labels == 1 or n_samples == 0 or (self.max_depth is not None and depth >= self.max_depth):
            return np.bincount(y).argmax()

        feature_index, threshold = self.best_split(X, y)
        if feature_index is None:
            return np.bincount(y).argmax()

        left_idxs = X[:, feature_index] <= threshold
        right_idxs = X[:, feature_index] > threshold
        left = self.build_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self.build_tree(X[right_idxs], y[right_idxs], depth + 1)

        return {'feature_index': feature_index, 'threshold': threshold, 'left': left, 'right': right}

    def fit(self, X, y):
        self.tree = self.build_tree(X, y)

    def predict_one(self, x, tree):
        if not isinstance(tree, dict):
            return tree
        if x[tree['feature_index']] <= tree['threshold']:
            return self.predict_one(x, tree['left'])
        else:
            return self.predict_one(x, tree['right'])

    def predict(self, X):
        return np.array([self.predict_one(x, self.tree) for x in X])