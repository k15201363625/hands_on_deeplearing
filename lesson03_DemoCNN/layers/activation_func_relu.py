import numpy as np

class Relu(object):
    def __init__(self, shape):
        self.input_shape = shape
        self.output_shape = shape

    def forward_propagate(self, x):
        self.x = x
        return np.maximum(x, 0)

    def backward_propagate(self,delta):
        self.delta = delta
        self.delta[self.x<0]=0
        return self.delta
