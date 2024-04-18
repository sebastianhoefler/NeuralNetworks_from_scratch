
# implement only for 2D case

import numpy as np


class Pooling():
    def __init__(self, stride_shape, pooling_shape) -> None:
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.trainable = False

    def forward(self, input_tensor):
        # store input tensor
        self.input_tensor = input_tensor

        # get the shape for the output
        b, c, h, w = self.input_tensor.shape
        # get output height
        h_pool = int((h - self.pooling_shape[0]) / self.stride_shape[0]) + 1
        # get output width
        w_pool = int((w - self.pooling_shape[1]) / self.stride_shape[1]) + 1

        # create an empty output tensor to store values
        self.output = np.zeros((b,c,h_pool, w_pool))

        # create an array that stores the max indices for h and w. Will be a tuple like x,y!
        self.max_indices_h = np.zeros((b, c, h_pool, w_pool), dtype=int)
        self.max_indices_w = np.zeros((b, c, h_pool, w_pool), dtype=int)
        
        for batch in range(b):
            for channel in range(c):
                for k in range(h_pool):
                    for l in range(w_pool):
                        # starting point for window in y direction depends on stride y
                        h_start = k * self.stride_shape[0]
                        # ending point for window in y direc. is h_start and add the pooling shape of y
                        h_end = h_start + self.pooling_shape[0]

                        # same idea here as with h
                        w_start = l * self.stride_shape[1]
                        w_end = w_start + self.pooling_shape[1]


                        pool_region = self.input_tensor[batch, channel, h_start:h_end, w_start:w_end]

                        # place values in empty array
                        self.output[batch, channel, k, l] = np.max(pool_region)

                        # get the index of the maximum value in the pooling region
                        max_index = np.unravel_index(pool_region.argmax(), pool_region.shape)

                        # place value in array for backprop later need to shift max index by the 
                        # h_start and w_start because thats our new point of reference
                        self.max_indices_h[batch, channel, k, l] = h_start + max_index[0]
                        self.max_indices_w[batch, channel, k, l] = w_start + max_index[1]


        return self.output

    def backward(self, error_tensor):
        '''
        The winner takes it all!!! Propagate the gradients to the locations where the maximum values were
        during the foward pass.
        '''

        b, c, h, w = self.input_tensor.shape
        grad_x = np.zeros((b, c, h, w)) # create tensor that has input shape

        _, _, h_pool, w_pool = error_tensor.shape
        for batch in range(b):
            for channel in range(c):
                for k in range(h_pool):
                    for l in range(w_pool):
                        h_max = self.max_indices_h[batch, channel, k, l]
                        w_max = self.max_indices_w[batch, channel, k, l]
                        grad_x[batch, channel, h_max, w_max] += error_tensor[batch, channel, k, l]

        return grad_x