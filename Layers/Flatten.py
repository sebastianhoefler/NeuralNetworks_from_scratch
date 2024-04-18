import numpy as np

class Flatten():
    def __init__(self) -> None:
        self.trainable = False

    def forward(self, input_tensor):
        # The tensor that will be forwarded here will have b,c,y,x dimensions.
        # Batch is the first

        # get batch dimensions
        self.shape = input_tensor.shape

        # flatten all dimensions for every element of batch
        tensor_flatten = input_tensor.reshape(self.shape[0],-1)

        return tensor_flatten

    def backward(self, error_tensor):
        # reshape the error tensor to the stored shape
        reshape_error = error_tensor.reshape(self.shape)

        return reshape_error