import numpy as np
import math
from tensor_graph.utils import initializer

# simulate tensorflow API
if 'GLOBAL_VARIABLE_SCOPE' not in globals():
    GLOBAL_VARIABLE_SCOPE = {}

class Variable(object):
    """
    Variable node class

    通过Variable 以及 Operator 的定义
    构建tensor graph 通过递归 实现自动多层次网络的反向传播争相传播
    同时这种基于图结构的组织方式有利于对于网络结构的组织更改 以及变量的管理

    as input/output data or operation weights data

    as input/output data:
        - input data : as palceholder
        - output data: provide shape as placeholder
        - None initialize
        - trainable = False
        - grad = True

    as weights data:
        - initialize by special method
        - Trainable = True
        - grad = True
        - can change optimizer method (default SGD)
    """
    def __init__(self, shape=list, name=str, scope='', grad=True, trainable=False, init='MSRA',optimizer_method='SGD'):
        """使用限定类型参数方法"""
        if scope != '':
            self.scope = scope if scope[-1] == '/' else scope + '/'
            self.name = self.scope + name
        else:
            self.name = name
            self.scope = scope

        if self.name in GLOBAL_VARIABLE_SCOPE:
            raise Exception('Variable name: %s exists!' % self.name)
        else:
            GLOBAL_VARIABLE_SCOPE[self.name] = self
        print(shape)
        for i in shape:
            if not isinstance(i, int):
                raise Exception("Variable name: %s shape is not list of int" % self.name)

        self.shape = shape
        self.initial = init
        self.data = initializer(shape, self.initial)
        self.optimizer_method = optimizer_method
        # 维护父子节点关系 父子节点都是operator
        self.child = []
        self.parent = []
        # 如果可以求导 在考虑是否trainable
        # 并且通过设置wait_bp 表示是否可以反向传播
        if grad:
            self.diff = np.zeros(self.shape)
            self.wait_bp = True
            self.trainable = trainable

    def eval(self):
        """
        正向求值
        :return:
        """
        for operator in self.parent:
            GLOBAL_VARIABLE_SCOPE[operator].forward()
        self.wait_bp = True
        return self.data

    def diff_eval(self):
        """
        只要经过了forward传播之后 反向求导才有意义
        :return:
        """
        if self.wait_bp:
            for operator in self.child:
                GLOBAL_VARIABLE_SCOPE[operator].backward()
            self.wait_bp = False
            return self.diff

    def apply_gradient(self, learning_rate=float, decay_rate=float, batch_size=1):
        """
        默认使用SGD 并且默认情况下使用L2 regularization
        并且支持通过函数接口 改变学习方法 以及 学习速率
        """
        self.data *= (1 - decay_rate)
        self.learning_rate = learning_rate

        if self.optimizer_method == 'SGD':
            self.data -= (learning_rate * self.diff / batch_size)
            self.diff *= 0

        elif self.optimizer_method == 'Momentum':
            self.mtmp = self.momentum * self.mtmp + self.diff / batch_size
            self.data -= learning_rate * self.mtmp
            self.diff *= 0

        elif self.optimizer_method == 'NGA':
            self.mtmp = self.momentum * self.mtmp + self.diff / batch_size + self.momentum * (
                        self.diff - self.lastdiff) / batch_size
            self.data -= learning_rate * self.mtmp
            self.lastdiff = self.diff
            self.diff *= 0

        elif self.optimizer_method == 'Adam':
            self.t += 1
            learning_rate_t = learning_rate * math.sqrt(1 - pow(self.beta2, self.t)) / (1 - pow(self.beta1, self.t))
            self.m_t = self.beta1 * self.m_t + (1 - self.beta1) * self.diff / batch_size
            self.v_t = self.beta2 * self.v_t + (1 - self.beta2) * ((self.diff / batch_size) ** 2)
            self.data -= learning_rate_t * self.m_t / (self.v_t + self.epsilon) ** 0.5
            self.diff *= 0

        else:
            raise Exception('No apply_gradient method: %s' % self.method)

    def set_method_sgd(self):
        self.optimizer_method = 'SGD'

    def set_method_momentum(self, momentum=0.9):
        self.optimizer_method = 'Momentum'
        self.momentum = momentum
        self.mtmp = np.zeros(self.diff.shape)

    def set_method_nga(self, momentum=0.9):
        self.optimizer_method = 'NGA'
        self.lastdiff = np.zeros(self.diff.shape)
        self.momentum = momentum
        self.mtmp = np.zeros(self.diff.shape)

    def set_method_adam(self, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.optimizer_method = 'Adam'
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_t = np.zeros(self.diff.shape)
        self.v_t = np.zeros(self.diff.shape)
        self.t = 0



def get_by_name(name):
    """
    通过名字获取变量 本质是构成hash dict
    """
    if 'GLOBAL_VARIABLE_SCOPE' in globals():
        try:
            return GLOBAL_VARIABLE_SCOPE[name]
        except:
            raise Exception('GLOBAL_VARIABLE_SCOPE not include name: %s' % name)
    else:
        raise Exception('No Variable')
