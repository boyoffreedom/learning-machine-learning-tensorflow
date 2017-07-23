import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs, in_size, out_size, activation_function=None):  #定义添加神经网络层函数
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))    #定义权重变量
    biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)              #定义偏置变量
    Wx_plus_b = tf.matmul(inputs, Weights) + biases                 #定义运算模型 输出=输入*权重+偏置
    if activation_function is None:                                 #选择激活函数
        outputs = Wx_plus_b                                         #线性输出
    else:
        outputs = activation_function(Wx_plus_b)                    #其他激活函数
    return outputs

#generate sample
x_data = np.linspace(-1,1,300)[:,np.newaxis]                        #x_data为300列1行的矩阵
noise = np.random.normal(0,0.05,x_data.shape)                       #noise为0-0.05的随机数，shape=x_data.shape
y_data = np.square(x_data) - 0.5 + noise                            #定义y_data

xs = tf.placeholder(tf.float32,[None,1])                            #定义两个placeholder用于训练
ys = tf.placeholder(tf.float32,[None,1])                            

#generate layer
l1 = add_layer(xs,1,10,activation_function=tf.nn.relu)              #定义第一层神经元
prediction = add_layer(l1,10,1,activation_function=None)            #定义输出层神经元

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                     reduction_indices=[1]))   #开根求和求平均

train_step = tf.train.GradientDescentOptimizer(0.3).minimize(loss)#训练为了减小误差


init = tf.global_variables_initializer()        #初始化所有变量
with tf.device('gpu:0'):
    sess = tf.Session()                             #创建会话并初始化变量
    sess.run(init)
    fig = plt.figure()                              #创建一个图像窗口
    ax = fig.add_subplot(1,1,1)                     #创建图像子窗口
    ax.scatter(x_data,y_data)                       #先画原始数据
    plt.ion()                                       #不暂停
    plt.show()                                      #显示图像
    for i in range(1000):                           #训练1000次，并输出结果
        sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
        if i % 50:
            #print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
            try:                                   #try语句删除生成的线条
                ax.lines.remove(lines[0])
            except Exception:
                pass
            prediction_value = sess.run(prediction,feed_dict={xs:x_data,ys:y_data})   #运行prediction向量图，获得300个输入的输出值
            lines = ax.plot(x_data, prediction_value,'r-',lw=5)                       #更新线条
            plt.pause(0.01)                                                           #暂停
            

