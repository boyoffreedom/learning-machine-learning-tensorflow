import tensorflow as tf
import numpy as np
from numpy.random import RandomState
import matplotlib.pyplot as plt

def add_layer(inputs, in_size, out_size, activation_function=None):  #定义添加神经网络层函数
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]),name='w')    #定义权重变量
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1,out_size]) + 0.1, name='b')              #定义偏置变量
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases                 #定义运算模型 输出=输入*权重+偏置
        if activation_function is None:                                 #选择激活函数
            outputs = Wx_plus_b                                         #线性输出
        else:
            outputs = activation_function(Wx_plus_b)                    #其他激活函数
        return outputs
    
#generate sample
rdm = RandomState(1)
dataset_size = 200
# 模拟输入是一个二维数组  
x_data = rdm.rand(dataset_size,2)  
#定义输出值，将x1+x2 < 1的输入数据定义为正样本
#y_data = [[int((x1-0.5)**2+(x2-0.5)**2 < 0.1)] for (x1,x2) in x_data]
y_data = [[int(2*x1+x2 < 1)] for (x1,x2) in x_data]
fig,ax=plt.subplots(1,1) 
idx_1 = [i for i in range(len(y_data)) if y_data[i] == [0]]
p1 = ax.scatter(x_data[idx_1,0],x_data[idx_1,1],marker='*',color='m',label='1',s=30)
idx_2 = [i for i in range(len(y_data)) if y_data[i] == [1]]
p2 = ax.scatter(x_data[idx_2,0],x_data[idx_2,1],marker='+',color='c',label='2',s=50)
ax.set_title('Data Set')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
#plt.show()

with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32,[None,2],name = 'x_input')           #定义两个placeholder用于训练
    ys = tf.placeholder(tf.float32,[None,1],name = 'y_input')

#generate layer
l1 = add_layer(xs,2,4,activation_function=tf.nn.relu)              #定义第一层神经元
prediction = add_layer(l1,4,1,activation_function=None)            #定义输出层神经元
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                         reduction_indices=[1]))   #开根求和求平均
    tf.summary.scalar('loss',loss)
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss)#训练为了减小误差

init = tf.global_variables_initializer()        #初始化所有变量
sess = tf.Session()                             #创建会话并初始化变量
sess.run(init)

merged = tf.summary.merge_all()                 #将图形、训练过程等数据合并在一起
writer = tf.summary.FileWriter("logs/",sess.graph)

for i in range(10001):                           #训练1000次，并输出结果
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if(i%20 == 0):
        result=sess.run(merged,feed_dict={xs:x_data,ys:y_data})
        writer.add_summary(result,i)
        #print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
plt.show()
