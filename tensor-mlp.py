import tensorflow as tf
import numpy as np


def dense_layer(input, out_dim, name, func=tf.nn.relu, trainalbe=True):
    in_dim = input.get_shape().as_list()[-1]
    d = 1.0 / np.sqrt(in_dim)
    with tf.variable_scope(name):
        w_init = tf.random_uniform_initializer(-d, d)
        b_init = tf.random_uniform_initializer(-d, d)
        w = tf.get_variable('w', dtype=tf.float32, shape=[in_dim, out_dim], initializer=w_init, trainable=trainalbe)
        b = tf.get_variable('b', shape=[out_dim], initializer=b_init, trainable=trainalbe)

        output = tf.matmul(input, w) + b
        if func is not None:
            output = func(output)

    return output


def conv2d_layer(input, filter_size, out_dim, name, strides, func=tf.nn.relu, trainalbe=True):
    in_dim = input.get_shape().as_list()[-1]
    d = 1.0 / np.sqrt(filter_size * filter_size * in_dim)
    with tf.variable_scope(name):
        w_init = tf.random_uniform_initializer(-d, d)
        b_init = tf.random_uniform_initializer(-d, d)
        w = tf.get_variable('w',
                            shape=[filter_size, filter_size, in_dim, out_dim],
                            dtype=tf.float32,
                            initializer=w_init, trainable=trainalbe)
        b = tf.get_variable('b', shape=[out_dim], initializer=b_init, trainable=trainalbe)

        output = tf.nn.conv2d(input, w, strides=strides, padding='SAME') + b
        if func is not None:
            output = func(output)

    return output


from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./')

trainalbe = True

g = tf.Graph()
with g.as_default() as g:
    with tf.device('/gpu:0'):
        global_step = tf.Variable(0, trainable=False)
        global_step_inc = tf.assign_add(global_step, 1)
        starter_learning_rate = 0.9
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 1000, 0.96, staircase=True)

        x = tf.placeholder(tf.float32, [None, 784], name='X')
        y = tf.placeholder(tf.int64, [None], name='Y')
        y_one_hot = tf.one_hot(y, 10)

        n1 = dense_layer(x, 400, 'n1', func=tf.nn.relu, trainalbe=True)
        n2 = dense_layer(n1, 200, 'n2', func=tf.nn.relu, trainalbe=True)

        p = dense_layer(n2, 10, 'logits_p', func=tf.nn.softmax, trainalbe=True)

        loss = tf.losses.softmax_cross_entropy(y_one_hot, p)

        acc = tf.reduce_mean(tf.cast(tf.equal(y, tf.argmax(p, 1)), tf.float32))

        opt = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(loss)

        sess = tf.Session(
            graph=g,
            config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False,
                gpu_options=tf.GPUOptions(allow_growth=True)))

        sess.run(tf.global_variables_initializer())

        # variables_to_restore = tf.contrib.slim.get_variables_to_restore(exclude=["conv11/w:0", "conv11/b:0"])
        variables_to_restore = tf.global_variables()
        # for v in variables_to_restore :
        #     print(v.name)
        saver = tf.train.Saver({var.name: var for var in variables_to_restore}, max_to_keep=0)

        # saver.restore(sess, 'checkpoints/check-mlp')

        for i in range(2000):
            batch = mnist.train.next_batch(50)
            o, train_loss, train_acc, inc = sess.run([opt, loss, acc, global_step_inc],
                                                     feed_dict={x: batch[0], y: batch[1]})
            test_loss, test_acc = sess.run([loss, acc], feed_dict={x: mnist.test.images, y: mnist.test.labels})
            print('Epoch : %d Train loss : %.3f Test loss : %.3f Test Acc : %.3f Train acc : %.3f' % (
                i, train_loss, test_loss, test_acc, train_acc))
        saver.save(sess, 'checkpoints/check-mlp')
