import numpy as np

class Sgd():
    def __init__(self, learning_rate) -> None:
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        weight_tensor = weight_tensor - self.learning_rate * gradient_tensor
        
        return weight_tensor