import numpy as np
import math
from tensor_graph.Operator import Operator
from tensor_graph.Variable import Variable,GLOBAL_VARIABLE_SCOPE
from functools import reduce

class FullyConnect(Operator):
    def __init__(self, input_variable=Variable, output_num=int, name=str, scope=''):
        if not isinstance(input_variable, Variable):
            raise Exception("Operator fullyconnected name: %s's input_variable is not instance of Variable" % name)

        self.batch_size = input_variable.shape[0]
        input_len = reduce(lambda x, y: x * y, input_variable.shape[1:])
        self.output_num = output_num
        self.weights = Variable([input_len, self.output_num], name='weights', scope=name, init='std_const', trainable=True)
        self.bias = Variable([self.output_num], name='bias', scope=name, init='zeros', trainable=True)

        self.output_variables = Variable([self.batch_size, self.output_num], name='out', scope=name)
        self.input_variables = input_variable
        Operator.__init__(self, [self.input_variables], [self.output_variables], name, scope)

    def forward(self):
        if self.wait_forward:
            for parent in self.parent:
                GLOBAL_VARIABLE_SCOPE[parent].eval()
            self.flatten_x = self.input_variables.data.reshape([self.batch_size, -1])
            self.output_variables.data = np.dot(self.flatten_x, self.weights.data)+self.bias.data
            self.wait_forward = False


    def backward(self):
        if self.wait_forward:
            pass
        else:
            for child in self.child:
                GLOBAL_VARIABLE_SCOPE[child].diff_eval()
            '''
            for i in range(self.batch_size):
                col_x = self.flatten_x[i][:, np.newaxis]
                diff_i = self.output_variables.diff[i][:, np.newaxis].T
                self.weights.diff += np.dot(col_x, diff_i)
                self.bias.diff += diff_i.reshape(self.bias.shape)
            '''
            # matrix op to complete
            self.weights.diff = np.dot(self.flatten_x.T,self.output_variables.diff)
            self.bias.diff = np.sum(self.output_variables.diff,axis=0)

            input_diff = np.dot(self.output_variables.diff, self.weights.data.T)
            self.input_variables.diff = np.reshape(input_diff, self.input_variables.shape)
            self.wait_forward = True