import tensorflow as tf         #导入tensorflow
import numpy as np              

# create data
x_data = np.random.rand(100).astype(np.float32)         #生成样本数据
y_data = x_data*0.1 + 0.3

Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))    #初始化神经网络权值，使用随机生成能提高效率
biases = tf.Variable(tf.zeros([1]))                         #初始化神经网络偏置

y = Weights*x_data + biases                                 #定义神经网络模型

loss = tf.reduce_mean(tf.square(y-y_data))                  #定义误差，计算输出值与样本Y值的方差

optimizer = tf.train.GradientDescentOptimizer(0.5)          #选择梯度下降法训练神经网络
train = optimizer.minimize(loss)                            #训练方法为减小误差

init = tf.global_variables_initializer()                    #初始化变量

sess = tf.Session()                                         #创建会话
sess.run(init)                                              #运行变量初始化

for step in range(201):
    sess.run(train)                                        #运行训练
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))    #输出步数与训练的拟合值
