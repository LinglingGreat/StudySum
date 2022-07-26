# CIFAR-10数据集上详细的Benchmark和排名在classification datasets results上
# http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html
# 本节实现的卷积神经网络根据Alex描述的cuda-convnet模型修改而得
# 新的技巧有：1.对weights进行了L2正则化，2.对图片进行了翻转、随机剪切等数据增强，制造了更多样本
# 3.在每个卷积-最大池化层后面使用了LRN层，增强了模型的泛化能力

# git clone https://github.com/tensorflow/models.git
# cd models/tutorials/image/cifar10

import cifar10, cifar10_input
import tensorflow as tf
import numpy as np
import time


max_steps = 3000
batch_size = 128
data_dir = '/tmp/cifar10_data/cifar-10-batches-bin'

def variable_with_weight_loss(shape, stddev, w1):
	var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
	if w1 is not None:
		weight_loss = tf.multiply(tf.nn.l2_loss(var), w1, name="weight_loss")
		tf.add_to_collection('losses', weight_loss)
	return var


def loss(logits, labels):
#      """Add L2Loss to all the trainable variables.
#      Add summary for "Loss" and "Loss/avg".
#      Args:
#        logits: Logits from inference().
#        labels: Labels from distorted_inputs or inputs(). 1-D tensor
#                of shape [batch_size]
#      Returns:
#        Loss tensor of type float.
#      """
#      # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


cifar10.maybe_download_and_extract()
# 对数据进行了数据增强，包括随机的水平翻转、随机剪切、设置随机的亮度和对比度，以及对数据进行标准化
images_train, labels_train = cifar10_input.distorted_inputs(data_dir=data_dir, batch_size=batch_size)
# 需要裁剪图片正中间的24x24大小的区块，并进行数据标准化操作
images_test, labels_test = cifar10_input.inputs(eval_data=True, data_dir=data_dir, batch_size=batch_size)

image_holder = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
label_holder = tf.placeholder(tf.int32, [batch_size])

# 第一个卷积层
weight1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=5e-2, w1=0.0)    # 不做L2正则化
kernel1 = tf.nn.conv2d(image_holder, weight1, [1, 1, 1, 1], padding="SAME")
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))
pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
# LRN对ReLu这种没有上限边界的激活函数会比较有用，因为它会从附近的多个卷积核的响应中挑选比较大的反馈,但不适合Sigmoid这种有固定边界并且能抑制过大值的激活函数
norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

# 第二个卷积层
weight2 = variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=5e-2, w1=0.0)
kernel2 = tf.nn.conv2d(norm1, weight2, [1, 1, 1, 1], padding="SAME")
bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias2))
norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

# 第一个全连接层
reshape = tf.reshape(pool2, [batch_size, -1])
dim = reshape.get_shape()[1].value
weight3 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, w1=0.004)
bias3 = tf.Variable(tf.constant(0.1, shape=[384]))
local3 = tf.nn.relu(tf.matmul(reshape, weight3) + bias3)

# 第二个全连接层
weight4 = variable_with_weight_loss(shape=[384, 192], stddev=0.04, w1=0.004)
bias4 = tf.Variable(tf.constant(0.1, shape=[192]))
local4 = tf.nn.relu(tf.matmul(local3, weight4) + bias4)

# 最后一层
weight5 = variable_with_weight_loss(shape=[192, 10], stddev=1/192.0, w1=0.0)
bias5 = tf.Variable(tf.constant(0.0, shape=[10]))
logits = tf.nn.relu(tf.matmul(local4, weight5) + bias5)

loss = loss(logits, label_holder)
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
# 求top k的准确率，默认使用top 1,即输出分数最高的那一类的准确率
top_k_op = tf.nn.in_top_k(logits, label_holder, 1)
sess = tf.InteractiveSession()
tf.global_variables.initializer().run()

# 启动前面的图片数据增强的线程队列(16个线程)来加速任务
tf.train.start_queue_runners()

for step in range(max_steps):
	start_time = time.time()
	image_batch, label_batch = sess.run([image_train, labels_train])
	_, loss_value = sess.run([train_op, loss], feed_dict={image_holder: image_batch, label_holder: label_batch})
	duration = time.time() - start_time
	if step % 10 == 0:
		examples_per_sec = batch_size / duration
		sec_per_batch = float(duration)
		format_str = ('step %d, loss=%.2f (%.1f examples/sec; %.3f sec/batch)')
		print(format_str % (step, loss_value, examples_per_sec, sec_per_batch))

num_examples = 10000
import math
num_iter = int(math.ceil(num_examples / batch_size))
true_count = 0
total_sample_count = num_iter * batch_size
step = 0
while step < num_iter:
	image_batch, label_batch = sess.run([images_batch, labels_batch])
	predictions = sess.run([top_k_op], feed_dict={image_holder:image_batch, label_holder:label_batch})
	true_count += np.sum(predictions)
	step += 1

precision = true_count / total_sample_count
print('precision @ 1 = %.3f' % precision)
