from Layers.FullyConnected import FullyConnected
from Layers.TanH import TanH
from Layers.Sigmoid import Sigmoid
import numpy as np
import copy

class RNN():
    def __init__(self, input_size, hidden_size, output_size):
        self.trainable = True
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc1 = FullyConnected(self.input_size + self.hidden_size, self.hidden_size)
        self.fc2 = FullyConnected(self.hidden_size, self.output_size)
        self.weights_fc1 = None
        self.weights_fc2 = None

        self.weights = self.fc1.weights

        self.tanh = TanH()
        self.sigmoid = Sigmoid()

        # added
        self.weights_out = None
        self.weights_hidden = None
        #####
        

        # Initialize hidden state to None
        self.h = None
        self.h_prev = None

        

        self.optimizer = None
        self._memorize = False
        self.bptt = None




    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.batch, _ = input_tensor.shape

        Y = np.zeros((self.batch, self.output_size))

        self.concat_mem = np.zeros((self.batch+1, self.input_size + self.hidden_size))
        self.fc1_mem = []
        self.fc2_mem = []
        self.tanh_mem = np.zeros((self.batch+1, self.hidden_size))
        self.sigmoid_mem = []

        if self._memorize:
            if self.h is None:
                self.h = np.zeros((self.batch + 1, self.hidden_size)) #changed b+1 to b
            else:
                self.h[0] = self.h_prev # inserts the last hidden state at first entry

        else:
            self.h = np.zeros((self.batch+1, self.hidden_size)) #changed b+1 to b


        for i in range(self.batch):
            x = input_tensor[i][None, :]
            h = self.h[i][None, :]

            # Concatenate and save (This goes into FC1)
            xh = np.concatenate((x,h), axis = 1)
            self.concat_mem[i] = xh

            # FC1 forward and save output (This goes into TanH)
            fc1 = self.fc1.forward(xh)
            self.fc1_mem.append(self.fc1.new_input_tensor)


            # Update hidden state here with the output of tanh.forward (This goes into FC2)
            self.h[i+1] = self.tanh.forward(fc1)
            self.tanh_mem[i] = self.tanh.output_tensor

        
            # FC2 forward to get output
            fc2 = self.fc2.forward(self.h[i+1][None, :])
            self.fc2_mem.append(self.fc2.new_input_tensor)

            # worked until here
            
            # implement sigmoid
            sig_out = self.sigmoid.forward(fc2)
            self.sigmoid_mem.append(self.sigmoid.output_tensor)

            Y[i] = sig_out

        self.h_prev = self.h[-1]

        return Y

    def backward(self, error_tensor):

        # Initialize the gradient w.r.t. input
        self.grad_x = np.zeros((self.batch, self.input_size))
        self.grad_h =np.zeros((self.batch, self.hidden_size))
                          
        # initialize the gradient weights
        self.gradient_weights_fc2 = np.zeros((self.hidden_size + 1, self.output_size))
        self.gradient_weights_fc1 = np.zeros((self.hidden_size + self.input_size + 1, self.hidden_size))


        '''
        1.  We don't use last hidden state so the error of the last hidden state is zero when we backpass.
            Therefore we initialize the tensor of h_error to zero and successively fill it

        2.  Gradient of copy at branch is just the sum of gradients

        3.  We need information about input_tensor when we backpass through a FullyConnected Layer. 

        4.  In sigmoid and tanh we need the activation of forward to compute the backward pass

        5. Store the outputs of tanh and sigmoid because we need them in the backward pass

        6. For the FullyConnected, we need the self.new_input_tensor in the backward pass so save that one!

        Note: Use command: self.fc1.input_tensor or self.tanh.input_tensor to set the according tensor 
      
        '''

        hidden_E = np.zeros((1, self.hidden_size))
        
        for i in reversed(range(self.batch)):

            #Backward for sigmoid
            self.sigmoid.output_tensor = self.sigmoid_mem[i]
            sig_error = self.sigmoid.backward(error_tensor[i])

            # Backward for FC2
            self.fc2.new_input_tensor = self.fc2_mem[i]
            y_error = self.fc2.backward(sig_error)

            # add two errors coming from h and y
            grad_hy = hidden_E + y_error

            # set tanh inp and backpass
            self.tanh.output_tensor = self.tanh_mem[i]
            grad_h = self.tanh.backward(grad_hy)
            
 
            # Backward for FC1 (input to this was concat)
            self.fc1.new_input_tensor = self.fc1_mem[i]
            xh_error = self.fc1.backward(grad_h)
            

            # split concatenated gradient (reverse concat)
            x_recov = xh_error[:, :self.input_size]
            h_recov = xh_error[:, self.input_size:]

            # Collect the grad_x outputs in a tensor and that is the gradient
            self.grad_x[i] = x_recov

            #print('h_recov', h_recov)
            hidden_E = h_recov


            self.gradient_weights_fc1 += self.fc1.gradient_weights

            self.gradient_weights_fc2 += self.fc2.gradient_weights
        


        if self.optimizer is not None:
            self.fc2.weights = self.optimizer.calculate_update(self.fc2.weights, self.gradient_weights_fc2)
            self.fc1.weights = self.optimizer.calculate_update(self.fc1.weights, self.gradient_weights_fc1)
            


        
        return self.grad_x
    


    def initialize(self, weights_initializer, bias_initializer):
        self.fc1.initialize(weights_initializer, bias_initializer)
        self.fc2.initialize(weights_initializer, bias_initializer)



    @property
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = copy.deepcopy(optimizer)

    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, value):
        self._memorize = value

    @property
    def weights(self):
        return self.fc1.weights
    
    @weights.setter
    def weights(self, weights):
        self.fc1.weights = weights

    @property
    def gradient_weights(self):
        return self.gradient_weights_fc1

    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self.fc1.gradient_weights = gradient_weights