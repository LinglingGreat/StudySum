from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# read data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)

# model
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, 784])    # None代表不限制条数
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# train
tf.global_variables_initializer().run()

for i in range(1000):
	# 使用一小部分样本进行训练：随机梯度下降
	batch_xs, batch_ys = mnist.train.next_batch(100)
	train_step.run({x: batch_xs, y_: batch_ys})

# test
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# 将bool值转换为float32
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))
