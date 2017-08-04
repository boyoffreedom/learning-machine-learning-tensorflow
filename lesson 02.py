import tensorflow as tf

with tf.device('cpu:0'):                        #通过这句代码将程序运行在cpu上
    state = tf.Variable(0,name='counter')

    #print(state.name)
    one = tf.constant(1)

    new_value = tf.add(state, one)
    update = tf.assign(state,new_value)
    init = tf.global_variables_initializer()  #如果有变量，一定要写这一句，初始化所有变量

    with tf.Session() as sess:
        sess.run(init)
        for _ in range(3):
            sess.run(update)
            print(sess.run(state))
