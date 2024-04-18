import numpy as np

# prevents net from relying on too many or too few neurons. Better for generalization

class Dropout():
    def __init__(self, probability) -> None:
        self.probability = probability
        self.trainable = False
        self.testing_phase = False

    def forward(self, input_tensor):
        self.input_tensor = input_tensor

        if self.testing_phase == False:
        # define a dropout filter / tensor
            self.dropout_filter =  (np.random.rand(*self.input_tensor.shape) >= 1 - self.probability) / (self.probability)
            
            return self.input_tensor * self.dropout_filter

        else:
            return self.input_tensor
    
    def backward(self, error_tensor):
        
        return error_tensor * self.dropout_filter
    