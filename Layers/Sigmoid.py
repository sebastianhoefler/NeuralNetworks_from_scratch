import numpy as np

class Sigmoid():
    def __init__(self) -> None:
        self.trainable = False

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.output_tensor = 1 / (1 + np.exp(-self.input_tensor))
        
        return self.output_tensor


    def backward(self, error_tensor):
        d_sigmoid = self.output_tensor * (1 - self.output_tensor)
        out_back = error_tensor * d_sigmoid #chain rule
        
        return out_back