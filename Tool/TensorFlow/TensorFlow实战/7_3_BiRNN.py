import tensorflow as tf 
import numpy as np 
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

learning_rate = 0.01
max_samples = 400000
batch_size = 128
display_step = 10

n_input = 28
n_steps = 28   # LSTM的展开步数
n_hidden = 256   # LSTM的隐藏节点数
n_classes = 10

# 二维结构的样本，n_steps表示时间点，n_input是每个时间点的数据
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

# 因为是双向LSTM，有forward和backward两个LSTM的cell,所以weights的参数量也翻倍
weights = tf.Variable(tf.random_normal([2*n_hidden, n_classes]))
biases = tf.Variable(tf.random_normal([n_classes]))

# 定义Bidirectional LSTM网络的生成函数
def BiRNN(x, weights, biases):

	# 把形状为(batch_size, n_steps, n_input)的输入变成长度为n_steps的列表，而其中元素形状为(batch_size, n_input)
	x = tf.transpose(x, [1, 0, 2])    # 将第一个维度和第二个维度进行交换
	x = tf.reshape(x, [-1, n_input])   # 变为(n_steps*batch_size, n_input)
	x = tf.split(x, n_steps)      # 将x拆成长度为n_steps的列表

	lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
	lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

	outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)
	return tf.matmul(outputs[-1], weights) + biases

# 生成Bidirectional LSTM网络
pred = BiRNN(x, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	step = 1
	while step * batch_size < max_samples:
		batch_x, batch_y = mnist.train.next_batch(batch_size)
		batch_x = batch_x.reshape((batch_size, n_steps, n_input))
		sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
		if step % display_step == 0:
			acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
			loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
			print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
				"{:.6f}".format(loss) + ", Training Accuracy= " + \
				"{:.5f}".format(acc))
		step += 1
	print("Optimization Finished!")

	test_len = 10000
	test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
	test_label = mnist.test.labels[:test_len]
	print("Test Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label}))