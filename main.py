import numpy as np
from collections import Counter
from sklearn.neighbors import BallTree
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns


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

class GaussianNaiveBayesScratch:
    def gauss_dist(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
    
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean = np.zeros((len(self.classes), X.shape[1]))
        self.var = np.zeros((len(self.classes), X.shape[1]))
        self.priors = np.zeros(len(self.classes))

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0)
            self.priors[idx] = X_c.shape[0] / X.shape[0]

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        likelihoods = []
        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            likelihood = np.sum(np.log(self.gauss_dist(idx, x)))
            likelihood += prior
            likelihoods.append(likelihood)
        return self.classes[np.argmax(likelihoods)]
    
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

# iris = datasets.load_iris()
# X = iris.data 
# y = iris.target

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

# print("kNN: ")
# print("Scratch: ")
# knn_scratch = KNNScratch(neighbors=3, metric='euclidean')
# knn_scratch.fit(X_train, y_train)
# knn_scratch_pred = knn_scratch.predict(X_test)
# print("Accuracy: ", accuracy_score(y_test, knn_scratch_pred))
# print("Classification Report: \n", classification_report(y_test, knn_scratch_pred))

# conf_matrix = confusion_matrix(y_test, knn_scratch_pred)

# plt.figure(figsize=(8, 6))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
#             xticklabels=iris.target_names, yticklabels=iris.target_names)
# plt.ylabel('Actual')
# plt.xlabel('Predicted')
# plt.title('Confusion Matrix')
# plt.show()



# print("\nLibrary: ")
# knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
# knn.fit(X_train, y_train)
# knn_pred = knn_scratch.predict(X_test)
# print("Accuracy: ", accuracy_score(y_test, knn_pred))
# print("Classification Report: \n", classification_report(y_test, knn_pred))

# print("\n==================================================================\n")

# print("\n\nGNB: ")
# print("Scratch: ")
# gnb_scratch = GaussianNaiveBayesScratch()
# gnb_scratch.fit(X_train, y_train)
# gnb_scratch_pred = gnb_scratch.predict(X_test)
# print("Accuracy: ", accuracy_score(y_test, gnb_scratch_pred))
# print("Classification Report: \n", classification_report(y_test, gnb_scratch_pred))


# print("\nLibrary: ")
# gnb = GaussianNB()
# gnb.fit(X_train, y_train)
# gnb_pred = gnb.predict(X_test)
# print("Accuracy: ", accuracy_score(y_test, gnb_pred))
# print("Classification Report: \n", classification_report(y_test, gnb_pred))

# print("\n==================================================================\n")

# print("\n\nDT: ")
# print("Scratch: ")
# dt_scratch = DecisionTreeClassifierScratch()
# dt_scratch.fit(X_train, y_train)
# dt_scratch_pred = dt_scratch.predict(X_test)
# print("Accuracy: ", accuracy_score(y_test, dt_scratch_pred))
# print("Classification Report: \n", classification_report(y_test, dt_scratch_pred))

# print("\nLibrary: ")
# dt = DecisionTreeClassifier()
# dt.fit(X_train, y_train)
# dt_pred = dt.predict(X_test)
# print("Accuracy: ", accuracy_score(y_test, dt_pred))
# print("Classification Report: \n", classification_report(y_test, dt_pred))