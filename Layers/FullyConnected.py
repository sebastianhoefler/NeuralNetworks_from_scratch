import numpy as np
from Layers.Base import BaseLayer
from Optimization.Optimizers import Sgd
from Optimization import *
from Layers.Initializers import *


class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.output_size = output_size
        self._optimizer = None
        self.weights = np.random.uniform(0,1,(self.input_size+1, self.output_size))    
        #self._gradient_weights = np.zeros_like(self.weights)

    @property
    def gradient_weights(self):
        return self._gradient_weights


    @gradient_weights.setter
    def gradient_weights(self, grad):
        self._gradient_weights = grad
    
    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, opt):
        self._optimizer = opt

    def forward(self, input_tensor):
        # Save the input tensor for later
        self.input_tensor = input_tensor
    
        # Get the batch dimension of the input tensor
        batch_dim = np.shape(input_tensor)[0]
        
        # Add a row of ones to the input tensor
        added_row = np.ones((batch_dim,1))
        self.new_input_tensor = np.concatenate((input_tensor,added_row), axis = 1)
        
        # Compute the dot product of the input tensor and the weights
        output_tensor = np.matmul(self.new_input_tensor, self.weights)
        
        return output_tensor
    
    def backward(self, error_tensor):
        # Calculate the error tensor to pass on
        error_tensor_prev = np.matmul(error_tensor, self.weights.T)
        error_tensor_prev = error_tensor_prev[:, :-1] # get rid of bias column
        
        # Calculate the gradient weights
        self._gradient_weights = np.matmul(self.new_input_tensor.T,error_tensor)

        # Only update if optimizer is set
        if self._optimizer != None:

            # Calculate the update
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
            self.bias = self.weights[-1:]
        else:
            pass
        
        return error_tensor_prev
    

    # reinitialize the weights and biases
    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize((self.input_size, self.output_size), self.input_size, self.output_size)
        self.bias = bias_initializer.initialize((1, self.output_size), 1, self.output_size)
        self.weights = np.vstack((self.weights, self.bias))

        


