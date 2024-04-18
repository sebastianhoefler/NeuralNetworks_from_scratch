import numpy as np

class L2_Regularizer():
    def __init__(self, alpha) -> None:
        self.alpha = alpha # regularization parameter "lambda"

    def calculate_gradient(self, weights):
        grad_penalty = self.alpha * weights # isn't there a 2 missing?????
        return grad_penalty

    def norm(self, weights):
        L2_penalty = self.alpha * np.sum(weights ** 2)
        return L2_penalty


class L1_Regularizer():
    def __init__(self, alpha) -> None:
        self.alpha = alpha # regularization parameter "lambda"

    def calculate_gradient(self, weights):
        grad_penalty = self.alpha * np.sign(weights)
        return grad_penalty

    def norm(self, weights):
        L1_penalty = self.alpha * np.sum(np.abs(weights))
        return L1_penalty