import numpy as np
'''
For fully connected layers:
fan_in: input dimensions of weights
fan_out: output dimension of weights

For CNN:
fan_in : input channels x kernel height x kernel width
fan_out: output channels x kernel height x kernel width
'''
class Constant():
    def __init__(self, constant_value=0.1) -> None:
        self.constant_value = constant_value

    def initialize(self, weights_shape, fan_in, fan_out):
        return np.full(weights_shape,self.constant_value)


class UniformRandom():
    def __init__(self) -> None:
        pass
    def initialize(self, weights_shape, fan_in, fan_out):
        return np.random.uniform(0,1,size=(fan_in,fan_out))


class Xavier():
    # typically used for weights
    def __init__(self) -> None:
        pass
    
    def initialize(self, weights_shape, fan_in, fan_out):
        # Calculate the variance of the distribution
        variance = 2.0 / (fan_in + fan_out)

        # Calculate the standard deviation
        std = np.sqrt(variance)

        # Generate the zero mean Gaussian
        gaussian = np.random.normal(0, std, weights_shape)

        return gaussian


class He():
    def __init__(self) -> None:
        pass
    def initialize(self, weights_shape, fan_in, fan_out):

        gaussian =  np.random.normal(0, (2 / fan_in)**(1/2), weights_shape)

        return gaussian