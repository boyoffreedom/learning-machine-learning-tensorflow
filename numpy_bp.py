import numpy as np

class Layer(object):
    def __init__(self,input_nodes,output_nodes,active_function = 'relu'):        #生成一层BP神经网络
        self.w = np.random.normal(0.05,1.0,(input_nodes,output_nodes))
        self.b = 0.1*np.ones((output_nodes,1))
        self.active_function = active_function
        
    def acfunc(self,Z):
        if(self.active_function == 'sigmoid'):      #激活函数为sigmoid
            return (1/(1 + np.exp(-Z)))
        elif(self.active_function == 'relu'):       #激活函数为relu
            return np.maximum(Z,0)
        else:                                       #否则当线性处理
            return Z
    
    def forward(self, x):                            #计算前向传播 Z=W*X+B
        self.Z = np.dot(self.w.T,x)+self.b
        self.A = self.acfunc(self.Z)
    
    def backward(self,inputs_list,dZ,learning_rate):    #计算反向传播
        m = inputs_list.shape[0]                        #特征个数
        dw = np.dot(dZ,inputs_list.T)/m                 #计算dw
        db = np.sum(dZ,axis=1,keepdims=True)/m          #计算db
        self.w -= learning_rate * dw.T                    #更新权值
        self.b -= learning_rate * db                    #更新阈值
        
    def acfunc_prime(self,Z):                           #定义激活函数导数
        if(self.active_function == 'relu'):             #relu导数
            data = Z.copy()
            data[data>0] = 1
            data[data<0] = 0
            return data 
        elif(self.active_function == 'sigmoid'):        #sigmoid导数
            return self.acfunc(Z)

#生成训练样本
X = np.array([ [0,0,0],
               [0,0,1],
               [0,1,0],
               [0,1,1],
               [1,0,0],
               [1,0,1],
               [1,1,0],
               [1,1,1]]).T

y = np.array([[0,1,1,0,1,0,0,1]])

#定义神经网络结构
bp_l1 = Layer(3,10,'relu')
bp_l2 = Layer(10,1,'sigmoid')
#训练神经网络
for i in range(80000):
    bp_l1.forward(X)        #l1前向传播
    bp_l2.forward(bp_l1.A)  #l2前向传播

    dZ2 = bp_l2.A - y       #计算dZ2
    bp_l2.backward(bp_l1.A,dZ2,0.05) #l2反向传播
    dZ1 = np.multiply(np.dot(bp_l2.w,dZ2),bp_l1.acfunc_prime(bp_l1.Z))  #计算dZ1
    bp_l1.backward(X,dZ1,0.05)       #l1反向传播
    if(i%20000 == 0):
        print("error ",np.sum(abs(dZ2)))    #定期输出误差

#loss = -np.dot(np.log(A2),y.T) + np.dot(np.log(1-A2),(1-y).T)；0
