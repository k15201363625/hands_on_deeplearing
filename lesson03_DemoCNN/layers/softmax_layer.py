import numpy as np

class Softmax(object):
    def __init__(self,shape):
        self.input_shape = np.zeros(shape)
        self.batch_size = shape[0]
        # loss 对于 z求偏导
        self.delta = np.zeros(shape)
    def cal_loss(self,pred,target):
        self.target = target
        self.pred = pred
        self.loss = 0
        # 需要前向传播
        self.forward_propagate(pred)

        for i in range(self.batch_size):
            # print(target[i])
            # print(pred[i,target[i]])
            self.loss += np.log(np.sum(np.exp(pred[i])))
            self.loss -= pred[i,int(target[i])]
        return self.loss

    def forward_propagate(self,pred):
        self.prob = np.zeros(pred.shape)
        for i in range(self.batch_size):
            # 防止运算溢出
            pred[i,:] -= np.max(pred[i,:])
            self.prob[i] = np.exp(pred[i]) / np.sum(np.exp(pred[i]))
        return self.prob

    def predict(self):
        """must be called after forward/cal_loss"""
        return np.argmax(self.prob,axis=1)

    def backward_propagate(self):
        self.delta = self.prob.copy()
        for i in range(self.batch_size):
            self.delta[i,int(self.target[i])] -= 1
        return self.delta


