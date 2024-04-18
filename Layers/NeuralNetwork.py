import numpy as np
from Layers import *
from Optimization import *
import copy

class NeuralNetwork():
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.layers = []
        self.data_layer = []
        self.loss_layer = []
        self.loss = []

    def forward(self):
        self.input_tensor, self.label_tensor = self.data_layer.next()
        output = self.input_tensor

        for layer in self.layers:
            output = layer.forward(output)

        loss_output = self.loss_layer.forward(output, self.label_tensor)
        return loss_output

    def backward(self):
        loss_backward = self.loss_layer.backward(self.label_tensor)

        for layer in reversed(self.layers):
            loss_backward = layer.backward(loss_backward)
        
        return loss_backward

    def append_layer(self, layer):
        if layer.trainable == True:
            layer.optimizer = copy.deepcopy(self.optimizer)

        self.layers.append(layer)

    def train(self, iterations):
        for _ in range(iterations):
            out = self.forward()
            self.loss.append(out)
            self.backward()
        

    def test(self, input_tensor):
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)

        return input_tensor

