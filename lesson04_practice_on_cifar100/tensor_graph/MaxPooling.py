import numpy as np
import math
from tensor_graph.Operator import Operator
from tensor_graph.Variable import Variable,GLOBAL_VARIABLE_SCOPE

class MaxPooling(Operator):
    def __init__(self, input_variable=Variable, name=str, scope='',ksize=2, stride=2):

        if not isinstance(input_variable, Variable):
            raise Exception("Operator Maxpooling name: %s's input_variable is not instance of Variable" % name)


        self.ksize = ksize
        self.stride = stride
        self.batch_size = input_variable.shape[0]
        self.output_channels = input_variable.shape[-1]
        self.index = np.zeros(input_variable.shape)

        self.input_variables = input_variable
        # default stride == ksize
        _output_shape = [self.batch_size, input_variable.shape[1]//stride, input_variable.shape[2]//stride,
                         self.output_channels]
        self.output_variables = Variable(_output_shape, name='out', scope=name)
        Operator.__init__(self,[self.input_variables], [self.output_variables], name, scope)

    def forward(self):
        if self.wait_forward:
            for parent in self.parent:
                GLOBAL_VARIABLE_SCOPE[parent].eval()
            self._pool()
            self.wait_forward = False

    def backward(self):
        if self.wait_forward:
            pass
        else:
            for child in self.child:
                GLOBAL_VARIABLE_SCOPE[child].diff_eval()
            self.input_variables.diff = np.repeat(np.repeat(self.output_variables.diff, self.stride, axis=1),
                                                  self.stride, axis=2) * self.index
            self.wait_forward = True

    def _pool(self):
        out = np.zeros(self.output_variables.shape)
        for b in range(self.input_variables.shape[0]):
            for c in range(self.output_channels):
                for i in range(0, self.input_variables.shape[1], self.stride):
                    for j in range(0, self.input_variables.shape[2], self.stride):
                        out[b, i//self.stride, j//self.stride, c] = np.max(
                            self.input_variables.data[b, i:i + self.ksize, j:j + self.ksize, c])
                        index = np.argmax(self.input_variables.data[b, i:i + self.ksize, j:j + self.ksize, c])
                        self.index[b, i+index//self.ksize, j + index%self.ksize, c] = 1
        self.output_variables.data = out
