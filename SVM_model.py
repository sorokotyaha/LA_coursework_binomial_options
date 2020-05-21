import numpy as np

class SVM:
    def __init__(self, iterations=200, step=0.1, fine=0.01):
        self._W = None # The noraml vector to plane
        self._interations = iterations # Number of iteratins
        self._step = step # Step in the gradient descent
        self._fine = fine # Fine coeefitient in lose function

    # Add 1 to the and of vector, that represent term without x
    # w1x1+w2x2+w0 = 0, (x1, x2) -> (x1, x2, 1)
    def _add_free_term(self, X):
        return [x + [1] for x in X]

    # Model train y = {-1, 1}
    def fit(self, X, Y):
        X = np.array(self._add_free_term(X))
        Y = np.array(Y)
        self._W = np.random.normal(loc=0, scale=0.05, size=X.shape[1])

        for i in range(self._interations):
            for k in range(X.shape[1]):
                margin = Y[k]*np.dot(self._W, X[k])
                if margin >= 1:
                    delta = self._fine/self._interations * self._W
                else:
                    delta = self._fine/self._interations * self._W - Y[k] * X[k]
                
                self._W = self._W - self._step * delta

    # Predict class for new X
    def predict(self, X):
        X = np.array(self._add_free_term(X))
        Y = []
        for i in range(len(X)):
            Y.append(np.sign(np.dot(self._W, X[i])))
        return np.array(Y)

