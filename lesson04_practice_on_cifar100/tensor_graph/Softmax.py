import numpy as np
import math
from tensor_graph.Operator import Operator
from tensor_graph.Variable import Variable,GLOBAL_VARIABLE_SCOPE

class Softmax(Operator):
    def __init__(self, predict = Variable, label=Variable, name=str, scope=''):
        """multi input and multi output"""
        self.batch_size = predict.shape[0]
        self.input_variables = [predict, label]
        self.loss = Variable([1], name='loss', scope=name, init='zeros')
        self.prediction = Variable(predict.shape, name='prediction', scope=name)

        self.output_variables = [self.loss, self.prediction]

        Operator.__init__(self, self.input_variables, self.output_variables, name, scope)

    def forward(self):
        if self.wait_forward:
            for parent in self.parent:
                GLOBAL_VARIABLE_SCOPE[parent].eval()

            predict = self.input_variables[0].data
            label = self.input_variables[1].data

            self.prediction.data = self.predict(predict)

            self.loss.data = 0
            for i in range(self.batch_size):

                self.loss.data += np.log(np.sum(np.exp(predict[i]))) - predict[i, label[i]]

            self.wait_forward = False


    def backward(self):
        if self.wait_forward:
            pass
        else:
            for child in self.child:
                GLOBAL_VARIABLE_SCOPE[child].diff_eval()
            self.input_variables[0].diff = self.prediction.data.copy()
            for i in range(self.batch_size):
                self.input_variables[0].diff[i, self.input_variables[1].data[i]] -= 1
            self.wait_forward = True


    def predict(self, prediction):
        exp_prediction = np.zeros(prediction.shape)
        softmax = np.zeros(prediction.shape)
        for i in range(self.batch_size):
            prediction[i, :] -= np.max(prediction[i, :])
            exp_prediction[i] = np.exp(prediction[i])
            softmax[i] = exp_prediction[i]/np.sum(exp_prediction[i])
        return softmax
