#lesson 06 tensorboard graph
import tensorflow as tf

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

with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32,[None,1],name = 'x_input')           #定义两个placeholder用于训练
    ys = tf.placeholder(tf.float32,[None,1],name = 'y_input')

#generate layer
l1 = add_layer(xs,1,10,activation_function=tf.nn.relu)              #定义第一层神经元
prediction = add_layer(l1,10,1,activation_function=None)            #定义输出层神经元

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                     reduction_indices=[1]))   #开根求和求平均
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)#训练为了减小误差

sess = tf.Session()
sess.run(tf.global_variables_initializer())

writer = tf.summary.FileWriter("logs/",sess.graph)
#Now you can use cmd open tensorboard by input 'tensorboard --logdir=logs' to check out the graph
