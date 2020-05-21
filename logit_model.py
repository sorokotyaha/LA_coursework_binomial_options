import numpy as np
from numpy.linalg import qr, inv
import math
import random

class Logit:
    def __init__(self):
        self.p = None

    def fit(self, A, b):
        A = np.array(A)
        b = np.array([b]).T
        q, r = qr(A)
        self.p = np.matmul(inv(r), np.matmul(q.T, b))
        # self.p = np.dot(q.T, b)

    def predict(self, C):
        return [round(1 / (1 + math.exp(-np.dot(i, self.p)))) for i in C]

