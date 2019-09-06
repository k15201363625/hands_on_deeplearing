import numpy as np
import math
from tensor_graph.Variable import Variable,GLOBAL_VARIABLE_SCOPE
from abc import abstractmethod

class Operator(object):
    """
    obtain register graph function
    obtain forward and backward propagate abstract func
    之后实现的层次结构都属于Operator 而输入输出data属于Variable
    """
    def __init__(self, input_variables, output_variables,name=str,scope=''):
        if scope != '':
            self.scope = scope if scope[-1] == '/' else scope + '/'
            self.name = self.scope + name
        else:
            self.name = name
            self.scope = scope

        # init input check
        if name in GLOBAL_VARIABLE_SCOPE.keys():
            raise Exception("Operator %s has exists !" % name)

        if not isinstance(input_variables[0], Variable):
            raise Exception("Operator %s 's input_variables is not instance(or list) of Variable!")

        if not isinstance(output_variables[0], Variable):
            raise Exception("Operator %s 's output_variables is not instance(or list) of Variable!")

        # register in GLOBAL_OP_SCOPE
        self.name = name
        GLOBAL_VARIABLE_SCOPE[self.name] = self

        # save input/output variable array
        self.child = []
        self.parent = []

        # register for input Variable's child and output Variable's parents
        self._register_graph(input_variables, output_variables)

        # self.wait_backward = not self.wait_forward
        self.wait_forward = True

    def _register_graph(self,input_variable, output_variable):
        """
        实现为一个操作关联所操作的变量 就是根据一个操作register一个图
        私有函数 不支持从外部进行注册 只支持通过operator进行注册
        传入变量要求是list/tuple形式
        """
        for _input in input_variable:
            _input.child.append(self.name)
            self.parent.append(_input.name)
        for _output in output_variable:
            _output.parent.append(self.name)
            self.child.append(_output.name)

    @abstractmethod
    def forward(self):
        """
        Template
        if self.wait_forward == True:
            1.check_parent_eval()
                for variable in self.parent:
                    variable.eval()
            2.do forward_cal()
            3.set wait_forward()
                self.wait_forward = False
        else:
            pass
        """
        pass

    @abstractmethod
    def backward(self):
        """
        Template:
        if self.wait_forward == True:
            pass
        else:
            1.check_child_diffeval()
                for variable in self.child:
                    variable.diff_eval()
            2.do backward_cal()
            3.set wait forward()
                self.wait_forward=True
        """
        pass

