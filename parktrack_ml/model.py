import numpy as np
import json


class CustomScaler:

    def __init__(self):
        self.mean = None
        self.std  = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std  = np.std(X, axis=0)
        return self

    def transform(self, X):
        return (X - self.mean) / (self.std + 1e-15)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def save(self, filename):
        with open(filename, 'w') as f:
            json.dump({'mean': self.mean.tolist(), 'std': self.std.tolist()}, f)

    def load(self, filename):
        with open(filename, 'r') as f:
            d = json.load(f)
        self.mean = np.array(d['mean'])
        self.std  = np.array(d['std'])


class CustomLogisticRegression:
    """Multinomial logistic regression with L2 regularisation (numpy only)."""

    def __init__(self, learning_rate=0.01, iterations=500, regularization=0.01):
        self.learning_rate  = learning_rate
        self.iterations     = iterations
        self.regularization = regularization
        self.weights        = None
        self.bias           = None
        self.classes        = None
        self.feature_names  = None

    def _softmax(self, z):
        e = np.exp(z - np.max(z, axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def _cross_entropy(self, y_true, y_pred):
        return -np.sum(y_true * np.log(y_pred + 1e-15)) / y_true.shape[0]

    def _one_hot(self, y):
        n = len(self.classes)
        oh = np.zeros((len(y), n))
        for i, v in enumerate(y):
            oh[i, np.where(self.classes == v)[0][0]] = 1
        return oh

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes    = len(self.classes)

        self.weights = np.random.randn(n_features, n_classes) * 0.01
        self.bias    = np.zeros((1, n_classes))
        y_oh         = self._one_hot(y)

        for i in range(self.iterations):
            yp  = self._softmax(X @ self.weights + self.bias)
            err = yp - y_oh
            self.weights -= self.learning_rate * (X.T @ err / n_samples + self.regularization * self.weights)
            self.bias    -= self.learning_rate * err.mean(axis=0, keepdims=True)

            if (i + 1) % 100 == 0:
                loss = self._cross_entropy(y_oh, yp)
                print(f"  iter {i + 1:>4}/{self.iterations}  loss={loss:.4f}")

    def predict_proba(self, X):
        return self._softmax(X @ self.weights + self.bias)

    def predict(self, X):
        return self.classes[np.argmax(self.predict_proba(X), axis=1)]

    def save_weights(self, filename):
        with open(filename, 'w') as f:
            json.dump({
                'weights':       self.weights.tolist(),
                'bias':          self.bias.tolist(),
                'classes':       self.classes.tolist(),
                'feature_names': self.feature_names,
                'learning_rate': self.learning_rate,
                'regularization': self.regularization,
            }, f)

    def load_weights(self, filename):
        with open(filename, 'r') as f:
            d = json.load(f)
        self.weights       = np.array(d['weights'])
        self.bias          = np.array(d['bias'])
        self.classes       = np.array(d['classes'])
        self.feature_names = d['feature_names']
        self.learning_rate = d['learning_rate']
        self.regularization = d['regularization']
