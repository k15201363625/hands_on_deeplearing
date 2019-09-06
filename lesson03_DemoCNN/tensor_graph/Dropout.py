import numpy as np
import math
from tensor_graph.Operator import Operator
from tensor_graph.Variable import Variable,GLOBAL_VARIABLE_SCOPE

class DropOut(Operator):
    def __init__(self, input_variable=Variable, state='test', name=str, scope='', prob=0.7):
        self.input_variables = input_variable
        self.output_variables = Variable(shape=input_variable.shape, scope=name, name='out')
        self.prob = prob
        self.state = state
        self.index = np.ones(input_variable.shape)

        Operator.__init__(self, [self.input_variables], [self.output_variables],name,scope)

    def forward(self):
        if self.wait_forward:
            for parent in self.parent:
                GLOBAL_VARIABLE_SCOPE[parent].eval()
            if self.state == 'train':
                self.index = np.random.random(self.input_variables.shape) < self.prob
                self.output_variables.data = self.input_variables.data * self.index
                # 保证均值不变 但是方差会发生改变 怎么解决??
                self.output_variables.data /= self.prob
            elif self.state == 'test':
                self.output_variables.data = self.input_variables.data
            else:
                raise Exception('Operator %s phase is not in test or train'% self.name)

            self.wait_forward=False


    def backward(self):
        if self.wait_forward:
            pass
        else:
            for child in self.child:
                GLOBAL_VARIABLE_SCOPE[child].diff_eval()
            if self.state == 'train':
                self.input_variables.diff = self.output_variables.diff * self.index * self.prob
            elif self.state == 'test':
                self.output_variables.diff = self.input_variables.diff
            else:
                raise Exception('Operator %s phase is not in test or train'% self.name)

            self.wait_forward = True

