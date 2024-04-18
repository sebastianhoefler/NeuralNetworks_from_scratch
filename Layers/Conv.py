import numpy as np
import scipy
from scipy.signal import correlate
from scipy.signal import resample



class Conv():
    def __init__(self, stride_shape, convolution_shape,num_kernels) -> None:
        self.stride_shape = stride_shape #can be single value or tuple
        self.convolution_shape = convolution_shape #1D or 2D conv. layer
        self.num_kernels = num_kernels
        self.trainable = True
        self.bias = np.random.uniform(size = self.num_kernels)
        self._optimizer = None
        self._bias_optimizer = None

        # Generating 1D convolution filter
        if len(self.convolution_shape) == 2:
            self.weights = np.random.rand(self.num_kernels, convolution_shape[0], convolution_shape[1])
        # Generating the nxn convolution filter
        if len(self.convolution_shape) == 3:
            self.weights = np.random.rand(self.num_kernels, convolution_shape[0], convolution_shape[1], convolution_shape[2])
        

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        # Determine the output shape based on the fact that we only use same correlation
        if input_tensor.ndim == 3:
            output_shape = (input_tensor.shape[0], self.num_kernels, input_tensor.shape[2])
        elif input_tensor.ndim == 4:
            output_shape = (input_tensor.shape[0], self.num_kernels, input_tensor.shape[2], input_tensor.shape[3])

        # make an empty output tensor
        output = np.zeros(output_shape)

        # perform correlation and add to output tensor
        for i in range(input_tensor.shape[0]):
            for j in range(self.num_kernels):
                out_sum = np.zeros(output_shape[2:])
                for k in range(input_tensor.shape[1]):
                    out_sum += scipy.signal.correlate(input_tensor[i, k], self.weights[j, k], mode='same', method='auto')
                output[i, j] = out_sum + self.bias[j]
     
        
        # -------------------------------------- SUBSAMPLING -------------------------------------- 
        if self.stride_shape != (1, 1) and isinstance(self.stride_shape, tuple):
            output = output[:, :, ::self.stride_shape[0], ::self.stride_shape[1]]

        # In the 1D case the stride_shape is a list. Therefore select the 0-th element to get an int.
        elif self.stride_shape != 1 and isinstance(self.stride_shape[0], int):
            output = output[:, :, ::self.stride_shape[0]]

        return output

    def backward(self, error_tensor):

        # make a copy of the weights
        weights_copy = np.copy(self.weights)
        
        # -------------------------------------- UPSAMPLING --------------------------------------
        if len(error_tensor.shape) == 3:
            # upsample the 1D error_tensor
            e1, e2, e3 = error_tensor.shape
            upsampled = np.zeros((e1, e2, self.input_tensor.shape[2]))
            upsampled[:, :, ::self.stride_shape[0]] = error_tensor
            print('upsampled,', upsampled.shape)
            error_tensor = upsampled
            
                
        if len(error_tensor.shape) == 4:
            # upsample the 2D error_tensor
            e1, e2, e3, e4 = error_tensor.shape
            upsampled = np.zeros((e1, e2, self.input_tensor.shape[2],self.input_tensor.shape[3]))
            upsampled[:, :, ::self.stride_shape[0], ::self.stride_shape[1]] = error_tensor
            error_tensor = upsampled

        output_back_shape = np.zeros_like(self.input_tensor)

        backward_filter = np.swapaxes(self.weights,0,1)


        for b in range(error_tensor.shape[0]):
            # For every filter
            for out_ch in range(backward_filter.shape[0]):
                # Create a list  so we can store the convolution output to stack and sum later
                conv_list = []
                for f_ch in range(backward_filter.shape[1]):
                    # convolve ever channel of error tensor with channel of filter, append results to list
                    conv_list.append(scipy.signal.convolve(error_tensor[b, f_ch], backward_filter[out_ch, f_ch], mode='same'))
                # stack entries of list on top of each other for summing    
                tmp = np.stack(conv_list, axis=0)
                # sum over the channel dim.
                tmp = tmp.sum(axis=0)
                # input the the out tensor
                output_back_shape[b, out_ch] = tmp


        # ------------------- Calculation of gradient with respect to weights -------------------

        # Pad the input_tensor with half the kernel width / height. Differentiate cases with even and odd kernel size!
        if len(error_tensor.shape) == 3:
            kw = self.weights.shape[2]
            kw_half = int(np.floor(kw / 2))

            pad_width = ((0, 0), (0, 0), (kw_half, kw_half))
            input_pad = np.pad(self.input_tensor, pad_width, mode='constant', constant_values=0)

            if self.weights.shape[2]%2 ==0:
                input_pad = input_pad[:,:,:-1]

        
        if len(error_tensor.shape) == 4:
            kw = self.weights.shape[2]
            kh = self.weights.shape[3]
            kw_half = int(np.floor(kw / 2))
            kh_half = int(np.floor(kh / 2))

            pad_width = ((0, 0), (0, 0), (kw_half, kw_half), (kh_half, kh_half))
            input_pad = np.pad(self.input_tensor, pad_width, mode='constant', constant_values=0)


            if self.weights.shape[2]%2 ==0:
                # do asymmetric padding remove the last row /col
                input_pad = input_pad[:,:,:-1,:]
            if self.weights.shape[3]%2 ==0:
                input_pad = input_pad[:,:,:,:-1]
            

        if len(self.convolution_shape) == 2:
            grad_weights = np.zeros((error_tensor.shape[0],self.num_kernels, self.convolution_shape[0], self.convolution_shape[1]))
        elif len(self.convolution_shape) == 3:
            grad_weights = np.zeros((error_tensor.shape[0],self.num_kernels, self.convolution_shape[0], self.convolution_shape[1], self.convolution_shape[2]))

        for batch in range(error_tensor.shape[0]):
            # loop over different kernels (output channels)
            for i in range(error_tensor.shape[1]):
                # loop over input channels
                for j in range(self.input_tensor.shape[1]):
                    grad_weights[batch, i, j] = scipy.signal.correlate(input_pad[batch, j], error_tensor[batch,i], mode='valid')
        # we have to sum over the batches because we have gradients for every batch.
        self.gradient_weights = grad_weights.sum(axis=0)



        dim = len(self.convolution_shape)

        if dim == 3:
            axes = (0,2,3)
        elif dim == 2:
            axes = (0,2)
        
        self._gradient_bias = np.sum(error_tensor, axis=axes)


        # update the weights if an optimizer is set
        if self._optimizer != None:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
        
        if self._bias_optimizer != None:
            self.bias = self._bias_optimizer.calculate_update(self.bias, self._gradient_bias)


        return output_back_shape
        
        

    def initialize(self, weights_initializer, bias_initializer):
        # basicall passes it into the initializers.py file. 

        # Specifying 1D filter
        if len(self.convolution_shape) == 2:
            shape = (self.num_kernels, self.convolution_shape[0], self.convolution_shape[1])

            # fan_in: (input channels x kernel height x kernel width)
            fan_in = self.convolution_shape[0] * self.convolution_shape[1]
            # dimensions_in: (output channels x kernel height x kernel width)
            fan_out = self.num_kernels * self.convolution_shape[1]

            self.weights = weights_initializer.initialize(shape, fan_in, fan_out)
            
            # we need to do this in order for it to be fed into initializers. Otherwise not enough arguments
            self.bias = bias_initializer.initialize((self.num_kernels), 1, self.num_kernels)[-1]
            
        
        
        # Specifying 2D filter
        if len(self.convolution_shape) == 3:
            shape = (self.num_kernels, self.convolution_shape[0], self.convolution_shape[1], self.convolution_shape[2])
            
            # dimensions_in: (input channels x kernel height x kernel width)
            fan_in = self.convolution_shape[0] * self.convolution_shape[1]* self.convolution_shape[2]
            # dimensions_in: (output channels x kernel height x kernel width)
            fan_out = self.num_kernels * self.convolution_shape[1] * self.convolution_shape[2]
    
            self.weights = weights_initializer.initialize(shape, fan_in, fan_out)

            # we need to do this in order for it to be fed into initializers. Otherwise not enough arguments.
            self.bias = bias_initializer.initialize((1, self.num_kernels), 1, self.num_kernels)[-1]
            

    @property
    def gradient_weights(self):
        return self._gradient_weights
    @gradient_weights.setter
    def gradient_weights(self, grad_weights):
        self._gradient_weights = grad_weights


    @property
    def gradient_bias(self):
        return self._gradient_bias
    @gradient_bias.setter
    def gradient_bias(self, grad_bias):
        self._gradient_bias = grad_bias

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
    def bias_optimizer(self, bias_opt):
        self._bias_optimizer = bias_opt