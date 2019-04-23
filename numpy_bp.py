#本段代码使用numpy科学计算库定义了神经网络矩阵计算模型，推荐大家仅用于研究，想提高效率还是建议使用tensorflow，如果要研究原理性的东西，建议配合吴恩达
#的深度学习教学视频并使用numpy进行编写，能够提高您对整个神经网络的抽象结构的理解
import numpy as np

class Layer(object):
    def __init__(self,input_nodes,output_nodes,active_function = 'relu'):        #生成一层BP神经网络
        self.in_n = input_nodes
        self.out_n = output_nodes
        self.wlist = []
        self.w = np.random.normal(0,1,(input_nodes,output_nodes))
        self.b = 0.01*np.ones((output_nodes,1))
        self.active_function = active_function

    def acfunc(self,Z):
        if(self.active_function == 'sigmoid'):      #激活函数为sigmoid
            return (1/(1 + np.exp(-Z)))
        elif(self.active_function == 'relu'):       #激活函数为relu
            return np.maximum(Z,0)
        else:                                       #否则当线性处理
            return Z
    
    def forward(self, x):                            #计算前向传播 Z=W*X+B
        self.Z = np.dot(self.w.T, x)+self.b
        self.A = self.acfunc(self.Z)
    
    def backward(self,inputs_list,dZ,learning_rate):    #计算反向传播
        m = inputs_list.shape[1]                        #样本个数
        dw = np.dot(dZ,inputs_list.T)/m                 #计算dw
        db = np.sum(dZ,axis=1,keepdims=True)/m          #计算db
        self.w -= learning_rate * dw.T                    #更新权值
        self.b -= learning_rate*db

    def acfunc_prime(self,Z):                           #定义激活函数导数
        if(self.active_function == 'relu'):             #relu导数
            data = Z.copy()
            data[data>0] = 1
            data[data<0] = 0
            return data 
        elif(self.active_function == 'sigmoid'):        #sigmoid导数
            return self.acfunc(Z)
        else:
            return 1

def train(x_,y_,alpha):
    l1.forward(x_)  # l1前向传播
    l2.forward(l1.A)  # l2前向传播

    dZ2 = l2.A - y_  # 计算dZ2
    l2.backward(l1.A, dZ2, alpha)  # l2反向传播
    dZ1 = np.multiply(np.dot(l2.w, dZ2), l1.acfunc_prime(l1.Z))  # 计算dZ1
    l1.backward(x_, dZ1, alpha)  # l1反向传播

def lose(x_,y_):
    l1.forward(x_)  # l1前向传播
    l2.forward(l1.A)  # l2前向传播

    dZ2 = l2.A - y_  # 计算dZ2
    return np.sum(abs(dZ2))/dZ2.shape[1]

def predict(x_):
    l1.forward(x_)  # l1前向传播
    l2.forward(l1.A)  # l2前向传播
    prd = l2.A
    return prd

#生成训练样本
x_train = np.array([])
y_train = np.array([])

x_test = ds.test_x
y_test = ds.test_y

#定义神经网络结构
l1 = Layer(72,36,'relu')
l2 = Layer(36,10,'sigmoid')

#训练神经网络
for i in range(5001):
    train(x_train,y_train,0.05)
    if(i%1000 == 0):
        print("train_error ",lose(x_train,y_train))    #定期输出误差

print("test_set error ",lose(x_test,y_test))    #定期输出误差



