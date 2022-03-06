#%%
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# 执行本文件，生成日志数据后
# 切换到Linux命令行下，执行TensorBoard程序，并通过--logdir指定TensorFlow日志路径，然后TensorBoard就可以生成所有汇总数据可视化的结果了
# tensorboard --logdir=/temp/tensorflow/mnist/logs/mnist_with_summaries
# tensorboard --logdir=MNIST_data/logs/mnist_with_summaries
# 执行上面的命令后，出现一条提示信息，复制其中的网址到浏览器，就可以看到可视化的图表了

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


max_steps=1000
learning_rate=0.001
dropout=0.9
# log_dir存放所有汇总数据供TensorBoard展示
# data_dir='/tmp/tensorflow/mnist/input_data'
# log_dir='/tmp/tensorflow/mnist/logs/mnist_with_summaries'
data_dir='MNIST_data/'
log_dir='MNIST_data/logs/mnist_with_summaries'


  # Import data
mnist = input_data.read_data_sets(data_dir,one_hot=True)

sess = tf.InteractiveSession()
  # Create a multilayer model.

  # Input placeholders
with tf.name_scope('input'):
  x = tf.placeholder(tf.float32, [None, 784], name='x-input')
  y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

with tf.name_scope('input_reshape'):
  image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
  # 将图片数据汇总给TensorBoard展示
  tf.summary.image('input', image_shaped_input, 10)

  # We can't initialize these variables to 0 - the network will get stuck.
def weight_variable(shape):
  """Create a weight variable with appropriate initialization."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  """Create a bias variable with appropriate initialization."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    # 进行记录和汇总
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    # 直接记录变量var的直方图数据
    tf.summary.histogram('histogram', var)

# 创建一层神经网络并进行数据汇总的函数
def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
  """Reusable code for making a simple neural net layer.
  It does a matrix multiply, bias add, and then uses relu to nonlinearize.
  It also sets up name scoping so that the resultant graph is easy to read,
  and adds a number of summary ops.
  """
  # Adding a name scope ensures logical grouping of the layers in the graph.
  with tf.name_scope(layer_name):
    # This Variable will hold the state of the weights for the layer
    with tf.name_scope('weights'):
      weights = weight_variable([input_dim, output_dim])
      variable_summaries(weights)
    with tf.name_scope('biases'):
      biases = bias_variable([output_dim])
      variable_summaries(biases)
    with tf.name_scope('Wx_plus_b'):
      preactivate = tf.matmul(input_tensor, weights) + biases
      tf.summary.histogram('pre_activations', preactivate)
    activations = act(preactivate, name='activation')
    tf.summary.histogram('activations', activations)
    return activations

hidden1 = nn_layer(x, 784, 500, 'layer1')

with tf.name_scope('dropout'):
  keep_prob = tf.placeholder(tf.float32)
  tf.summary.scalar('dropout_keep_probability', keep_prob)
  dropped = tf.nn.dropout(hidden1, keep_prob)

  # Do not apply softmax activation yet, see below.全等映射identity
y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)

with tf.name_scope('cross_entropy'):
    # The raw formulation of cross-entropy,
    #
    # tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
    #                               reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.nn.softmax_cross_entropy_with_logits on the
    # raw outputs of the nn_layer above, and then average across
    # the batch.
  diff = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)
  with tf.name_scope('total'):
    cross_entropy = tf.reduce_mean(diff)
tf.summary.scalar('cross_entropy', cross_entropy)

with tf.name_scope('train'):
  train_step = tf.train.AdamOptimizer(learning_rate).minimize(
      cross_entropy)

with tf.name_scope('accuracy'):
  with tf.name_scope('correct_prediction'):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

  # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
merged = tf.summary.merge_all()    # 获取所有汇总操作，以便后面执行
# 定义文件记录器，将Session的计算图sess.graph加入训练过程的记录器，这样在TensorBoard的GRAPHS窗口中就能展示整个计算图的可视化效果
train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
test_writer = tf.summary.FileWriter(log_dir + '/test')
tf.global_variables_initializer().run()

  # Train the model, and also write summaries.
  # Every 10th step, measure test-set accuracy, and write test summaries
  # All other steps, run train_step on training data, & add training summaries

def feed_dict(train):
  """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
  if train:
    xs, ys = mnist.train.next_batch(100)
    k = dropout
  else:
    xs, ys = mnist.test.images, mnist.test.labels
    k = 1.0
  return {x: xs, y_: ys, keep_prob: k}

# 实际执行具体的训练、测试及日志记录的操作  
saver = tf.train.Saver()     # 创建模型的保存器
for i in range(max_steps):
  if i % 10 == 0:  # Record summaries and test-set accuracy
    summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
    # 将汇总结果和循环步数写入日志文件
    test_writer.add_summary(summary, i)
    print('Accuracy at step %s: %s' % (i, acc))
  else:  # Record train set summaries, and train
    if i % 100 == 99:  # Record execution stats
      # 定义TensorFlow运行选项，其中trace_level为FULL_TRACE，并使用tf.RunMetadata()定义TensorFlow运行的元信息,这样可以记录训练时运算时间和内存占用等方面的信息
      run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
      run_metadata = tf.RunMetadata()
      summary, _ = sess.run([merged, train_step],
                            feed_dict=feed_dict(True),
                            options=run_options,
                            run_metadata=run_metadata)
      train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
      train_writer.add_summary(summary, i)
      saver.save(sess, log_dir+"/model.ckpt", i)
      print('Adding run metadata for', i)
    else:  # Record a summary
      summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
      train_writer.add_summary(summary, i)
train_writer.close()
test_writer.close()



