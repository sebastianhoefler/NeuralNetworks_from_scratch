import numpy as np

class TanH():
    def __init__(self) -> None:
        self.trainable = False

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.output_tensor = np.tanh(self.input_tensor)

        return self.output_tensor

    def backward(self, error_tensor):
        d_tanh = 1 - self.output_tensor ** 2
        out_back = d_tanh * error_tensor

        return out_back
    