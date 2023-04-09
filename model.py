from config import *
import numpy as np
class TwoLayerNet:
    def __init__(self,input_size,hidden_size,output_size,reg_lambda):
        self.reg_lambda=reg_lambda
        self.params={}
        self.params["w1"]=np.random.randn(input_size,hidden_size)
        self.params["b1"]=np.zeros(hidden_size)
        self.params["w2"]=np.random.randn(hidden_size,output_size)
        self.params["b2"]=np.zeros(output_size)



    def softmax(self,x):
        e_x=np.exp(x)
        return e_x/np.sum(e_x,axis=1,keepdims=True)


    def forward(self,x):#前向传播
        caches=[]
        w1,b1,w2,b2=self.params["w1"],self.params["b1"],self.params["w2"],self.params["b2"]
        z1=np.dot(x,w1)+b1
        a1=np.tanh(z1)
        cache=(z1,a1)
        caches.append(cache)

        z2=np.dot(a1,w2)+b2
        y_hat=self.softmax(z2)
        cache=(z2,y_hat)
        caches.append(cache)
        return caches,y_hat

    def backward(self,x,y,learning_rate,batch_size):
        w1,b1,w2,b2=self.params["w1"],self.params["b1"],self.params["w2"],self.params["b2"]
        caches,_=self.forward(x)
        z1,a1=caches[0]
        z2,a2=caches[1]
        delta3=a2

        delta3[range(x.shape[0]),y.astype("int64")]-=1

        #下面计算隐藏层误差
        delta2=np.dot(delta3,w2.T)*(1-np.power(a1,2))

        ##计算权重和偏差的梯度
        dw2=np.dot(a1.T,delta3)
        db2=np.sum(delta3,axis=0)
        dw1=np.dot(x.T,delta2)
        db1=np.sum(delta2,axis=0)

        ##加入L2正则化
        dw2+=self.reg_lambda*w2
        dw1+=self.reg_lambda*w1

        self.params["w1"]+=-learning_rate*dw1/batch_size
        #assert(self.params["b1"].shape==db1.shape)
        self.params["b1"] += -learning_rate * db1/batch_size
        self.params["w2"] += -learning_rate * dw2/batch_size
        #assert (self.params["b2"].shape==db2.shape)
        self.params["b2"] += -learning_rate * db2/batch_size
    def compute_loss(self, y_hat, y):
        """
        计算交叉熵损失函数值，并加上L2正则化项
        """
        # 交叉熵损失函数
        data_loss = np.sum(-np.log(y_hat[range(y_hat.shape[0]), y.astype("int64")]))
        # 加上L2正则化项
        reg_loss = 0.5 * self.reg_lambda * (np.sum(np.square(self.params["w1"])) + np.sum(np.square(self.params["w2"])))
        # 总损失函数值
        loss = data_loss + reg_loss
        return loss/y.shape[0]
    def predict(self,x):
        _,y_hat=self.forward(x)
        return np.argmax(y_hat,axis=1)






