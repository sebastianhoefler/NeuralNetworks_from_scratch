import numpy as np
from Layers.Helpers import compute_bn_gradients # compute batch norm grad w.r.t inputs NOT weights.

class BatchNormalization():

    def __init__(self, channels) -> None:
        self.trainable = True

        self.channels = channels
        self.weights = np.ones((channels)) # gamma
        self.bias = np.zeros((channels)) # beta
        self.weights_reshaped = self.weights.reshape(1, -1, 1, 1)
        self.bias_reshaped = self.bias.reshape(1, -1, 1, 1)

        self._optimizer = None 
        self._bias_optimizer = None
        self._gradient_bias = None
        self._gradient_weights = None

        self.mu = 0
        self.var = 1
        self.mu_test = 0 # for moving average. This basically stores previous value of mean and var
        self.var_test = 1 # for moving average

        self.testing_phase = False # if set to true, I need to use moving avg.
        self.alpha = 0.8


    def forward(self, input_tensor):
        """
        -   Compute the output of the batch normalization layer for the given input.

        -   During training, the layer normalizes the batch data and scales the output using learned 
            parameters. The layer also maintains moving averages of the mean and variance for use 
            during testing.

        -   During testing, the layer normalizes the batch using the previously calculated moving averages.
        """
        X = input_tensor
        self.X = X
        eps = np.finfo(float).eps
        conv_layer = False

        self.mu = np.mean(X, axis=0)
        self.var = np.var(X, axis=0)

        if X.ndim == 4:
            conv_layer = True
            X = self.reformat(X)
            self.mu = np.mean(X, axis=0)
            self.var = np.var(X, axis=0)
        
        if self.testing_phase == False:
            new_mu = np.mean(X, axis=0)
            new_var = np.var(X, axis=0)

            # Calculation of the moving average
            self.mu_test = self.alpha * self.mu + (1 - self.alpha) * new_mu
            self.var_test = self.alpha * self.var + (1 - self.alpha) * new_var
            self.mu = new_mu
            self.var = new_var

            X_hat = (X - new_mu) / np.sqrt(new_var + eps)
            self.X_hat = X_hat

        elif self.testing_phase == True:
            print('self.X', self.X.shape)
            print('self.mu_test', self.mu_test.shape)
            print('self.var_test',self.var_test.shape)

            X_hat = (X - self.mu_test) / (np.sqrt(self.var_test + eps))
            self.X_hat = X_hat
        
        output = self.weights * self.X_hat + self.bias
        if conv_layer == True:
            output = self.reformat(output)

        return output
    
    def backward(self, error_tensor):
        """
        Perform backpropagation for the batch normalization layer and compute the gradient of the loss 
        with respect to the layer's input.

        Parameters:
        error_tensor (np.ndarray): Gradient of the loss with respect to the output of this layer.

        Returns:
        np.ndarray: Gradient of the loss with respect to the input of this layer.
        """

        # --------------------------- gradient input calculation ---------------------------    
        E = error_tensor
        X = self.X

        # print('X', X.shape)
        # print('E', E.shape)
        conv_layer = False

        if E.ndim == 4:
            E = self.reformat(E)
            X = self.reformat(X)
            conv_layer = True

        X_grad = compute_bn_gradients(E, X, self.weights, self.mu, self.var)

        # ----------------- gradient weight and bias calculation and update -----------------       
        self.gradient_weights = np.sum(E * self.X_hat, axis=0)
        self.gradient_bias = np.sum(E, axis=0)

        if self._optimizer is not None:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
        if self._bias_optimizer is not None:
            self.bias = self._bias_optimizer.calculate_update(self.bias, self._gradient_bias)

        if conv_layer == True:
            X_grad = self.reformat(X_grad)
        
        return X_grad


    def reformat(self, tensor):
     
        # Helper function to convert 4D tensor to 2D (flatten spatial dimensions) and vice versa.

        T = tensor
        if tensor.ndim == 2:
            B, H, M, N = self.original_shape
            T = np.reshape(T, (B, M * N, H))
            T = np.transpose(T, (0,2,1))
            T = np.reshape(T, self.original_shape)
            return T

        if tensor.ndim == 4:
            self.original_shape = T.shape
            B, H, M, N = T.shape
            T = np.reshape(T, (B, H, M * N))
            T = np.transpose(T, (0,2,1))
            T = np.reshape(T, (B * M * N, H))
            return T
        
    def initialize(self, weights_initializer, bias_initializer):
        self.weights = np.ones(self.channels)
        self.bias = np.zeros(self.channels)


    @property
    def gradient_weights(self):
        return self._gradient_weights
    
    @gradient_weights.setter
    def gradient_weights(self, grad):
        self._gradient_weights = grad

    @property
    def gradient_bias(self):
        return self._gradient_bias
    @gradient_bias.setter
    def gradient_bias(self, bias):
        self._gradient_bias = bias

    @property
    def optimizer(self):
        return self._optimizer
    @optimizer.setter
    def optimizer(self, opt):
        self._optimizer = opt


    @property
    def bias_optimizer(self):
        return self._bias_optimizer
    @bias_optimizer.setter
    def bias_optimizer(self, bias):
        self._bias_optimizer = bias

