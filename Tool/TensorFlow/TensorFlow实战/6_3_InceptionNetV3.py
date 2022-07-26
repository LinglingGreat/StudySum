# coding:UTF-8
import tensorflow as tf 
from datetime import datetime
import math
import time

slim = tf.contrib.slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)

# 生成网络中经常用到的函数的默认参数，如卷积的激活函数、权重初始化方式、标准化器等
def inception_v3_arg_scope(weight_decay=0.00004, stddev=0.1, batch_norm_var_collection='moving_vars'):
	batch_norm_params = {'decay': 0.997, 'epsilon': 0.001, 'updates_collections': tf.GraphKeys.UPDATE_OPS,
	'variables_collections': {'beta': None, 'gamma': None, 'moving_mean': [batch_norm_var_collection], 
							'moving_variance': [batch_norm_var_collection],}}

	# slim.arg_scope可以给函数的参数自动赋予某些默认值
	# 对[slim.conv2d, slim.fully_connected]这两个函数的参数自动赋值：weights_regularizer=slim.l2_regularizer(weight_decay)
	with slim.arg_scope([slim.conv2d, slim.fully_connected], weights_regularizer=slim.l2_regularizer(weight_decay)):
		with slim.arg_scope([slim.conv2d], weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
			activation_fn = tf.nn.relu, normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params) as sc:
			return sc

# 生成Inception V3网络的卷积部分
def inception_v3_base(inputs, scope=None):
	end_points = {}
	with tf.variable_scope(scope, 'InceptionV3', [inputs]):
		with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='VALID'):
			# 创建卷积层，输入的tensor,输出的通道数，卷积核尺寸，步长(默认为1)，padding模式(默认VALID)
			net = slim.conv2d(inputs, 32, [3, 3], stride=2, scope='Conv2d_1a_3x3')
			net = slim.conv2d(net, 32, [3, 3], scope='Conv2d_2a_3x3')
			net = slim.conv2d(net, 64, [3, 3], padding='SAME', scope='Conv2d_2b_3x3')
			net = slim.max_pool2d(net, [3, 3], stride=2, scope='MaxPool_3a_3x3')
			net = slim.conv2d(net, 80, [1, 1], scope='Conv2d_3b_1x1')
			net = slim.conv2d(net, 192, [3, 3], scope='Conv2d_4a_3x3')
			net = slim.max_pool2d(net, [3, 3], stride=2, scope='MaxPool_5a_3x3')

		# 三个连续的Inception模块组
		with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
			# 第一个Inception模块包含了3个结构类似的Inception Module
			with tf.variable_scope('Mixed_5b'):
				with tf.variable_scope('Branch_0'):
					branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
				with tf.variable_scope('Branch_1'):
					branch_1 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0a_1x1')
					branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope='Conv2d_0b_5x5')
				with tf.variable_scope('Branch_2'):
					branch_2 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
					branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
					branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0c_3x3')
				with tf.variable_scope('Branch_3'):
					branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_1x1')
					branch_3 = slim.conv2d(branch_3, 32, [1, 1], scope='Conv2d_0b_1x1')
				net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)   # 在第三个维度即输出通道上合并

			with tf.variable_scope('Mixed_5c'):
				with tf.variable_scope('Branch_0'):
					branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
				with tf.variable_scope('Branch_1'):
					branch_1 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0a_1x1')
					branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope='Conv2d_0b_5x5')
				with tf.variable_scope('Branch_2'):
					branch_2 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
					branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
					branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0c_3x3')
				with tf.variable_scope('Branch_3'):
					branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_1x1')
					branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
				net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

			with tf.variable_scope('Mixed_5d'):
				with tf.variable_scope('Branch_0'):
					branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
				with tf.variable_scope('Branch_1'):
					branch_1 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0a_1x1')
					branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope='Conv2d_0b_5x5')
				with tf.variable_scope('Branch_2'):
					branch_2 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
					branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
					branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0c_3x3')
				with tf.variable_scope('Branch_3'):
					branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_1x1')
					branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
				net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

			# 第二个Inception没落包含了5个Inception Module
			with tf.variable_scope('Mixed_6a'):
				with tf.variable_scope('Branch_0'):
					branch_0 = slim.conv2d(net, 384, [3, 3], stride=2, padding='VALID', scope='Conv2d_1a_1x1')
				with tf.variable_scope('Branch_1'):
					branch_1 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
					branch_1 = slim.conv2d(branch_1, 96, [3, 3], scope='Conv2d_0b_5x5')
					branch_1 = slim.conv2d(branch_1, 96, [3, 3], stride=2, padding='VALID', scope='Conv2d_1a_1x1')
				with tf.variable_scope('Branch_2'):
					branch_2 = slim.max_pool2d(net, [3, 3],stride=2, padding='VALID', scope='MaxPool_1a_3x3')
				net = tf.concat([branch_0, branch_1, branch_2], 3)

			with tf.variable_scope('Mixed_6b'):
				with tf.variable_scope('Branch_0'):
					branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
				with tf.variable_scope('Branch_1'):
					branch_1 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
					branch_1 = slim.conv2d(branch_1, 128, [1, 7], scope='Conv2d_0b_1x7')
					branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
				with tf.variable_scope('Branch_2'):
					branch_2 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
					branch_2 = slim.conv2d(branch_2, 128, [7, 1], scope='Conv2d_0b_7x1')
					branch_2 = slim.conv2d(branch_2, 128, [1, 7], scope='Conv2d_0c_1x7')
					branch_2 = slim.conv2d(branch_2, 128, [7, 1], scope='Conv2d_0d_7x1')
					branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
				with tf.variable_scope('Branch_3'):
					branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
					branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
				net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

			with tf.variable_scope('Mixed_6c'):
				with tf.variable_scope('Branch_0'):
					branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
				with tf.variable_scope('Branch_1'):
					branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
					branch_1 = slim.conv2d(branch_1, 160, [1, 7], scope='Conv2d_0b_1x7')
					branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
				with tf.variable_scope('Branch_2'):
					branch_2 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
					branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0b_7x1')
					branch_2 = slim.conv2d(branch_2, 160, [1, 7], scope='Conv2d_0c_1x7')
					branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0d_7x1')
					branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
				with tf.variable_scope('Branch_3'):
					branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
					branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
				net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

			with tf.variable_scope('Mixed_6d'):
				with tf.variable_scope('Branch_0'):
					branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
				with tf.variable_scope('Branch_1'):
					branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
					branch_1 = slim.conv2d(branch_1, 160, [1, 7], scope='Conv2d_0b_1x7')
					branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
				with tf.variable_scope('Branch_2'):
					branch_2 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
					branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0b_7x1')
					branch_2 = slim.conv2d(branch_2, 160, [1, 7], scope='Conv2d_0c_1x7')
					branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0d_7x1')
					branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
				with tf.variable_scope('Branch_3'):
					branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
					branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
				net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

			with tf.variable_scope('Mixed_6e'):
				with tf.variable_scope('Branch_0'):
					branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
				with tf.variable_scope('Branch_1'):
					branch_1 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
					branch_1 = slim.conv2d(branch_1, 192, [1, 7], scope='Conv2d_0b_1x7')
					branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
				with tf.variable_scope('Branch_2'):
					branch_2 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
					branch_2 = slim.conv2d(branch_2, 192, [7, 1], scope='Conv2d_0b_7x1')
					branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0c_1x7')
					branch_2 = slim.conv2d(branch_2, 192, [7, 1], scope='Conv2d_0d_7x1')
					branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
				with tf.variable_scope('Branch_3'):
					branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
					branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
				net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
			end_points['Mixed_6e'] = net   # 将Mixed_6e存储于end_points中，作为Auxiliary Classifier辅助模型的分类

			# 第三个Inception模块包含了3个Inception Module
			with tf.variable_scope('Mixed_7a'):
				with tf.variable_scope('Branch_0'):
					branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
					branch_0 = slim.conv2d(branch_0, 320, [3, 3], stride=2, padding='VALID', scope='Conv2d_1a_3x3')
				with tf.variable_scope('Branch_1'):
					branch_1 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
					branch_1 = slim.conv2d(branch_1, 192, [1, 7], scope='Conv2d_0b_1x7')
					branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
					branch_1 = slim.conv2d(branch_1, 192, [3, 3], stride=2, padding='VALID', scope='Conv2d_1a_3x3')
				with tf.variable_scope('Branch_2'):
					branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='MaxPool_1a_3x3')
				net = tf.concat([branch_0, branch_1, branch_2], 3)

			with tf.variable_scope('Mixed_7b'):
				with tf.variable_scope('Branch_0'):
					branch_0 = slim.conv2d(net, 320, [1, 1], scope='Conv2d_0a_1x1')
				with tf.variable_scope('Branch_1'):
					branch_1 = slim.conv2d(net, 384, [1, 1], scope='Conv2d_0a_1x1')
					branch_1 = tf.concat([
						slim.conv2d(branch_1, 384, [1, 3], scope='Conv2d_0a_1x3'),
						slim.conv2d(branch_1, 384, [3, 1], scope='Conv2d_0b_3x1')], 3)
				with tf.variable_scope('Branch_2'):
					branch_2 = slim.conv2d(net, 448, [1, 1], scope='Con2d_0a_1x1')
					branch_2 = slim.conv2d(branch_2, 384, [3, 3], scope='Con2d_0b_3x3')
					branch_2 = tf.concat([
						slim.conv2d(branch_2, 384, [1, 3], scope='Conv2d_0c_1x3'),
						slim.conv2d(branch_2, 384, [3, 1], scope='Conv2d_0d_3x1')], 3)
				with tf.variable_scope('Branch_3'):
					branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
					branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
				net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

			with tf.variable_scope('Mixed_7c'):
				with tf.variable_scope('Branch_0'):
					branch_0 = slim.conv2d(net, 320, [1, 1], scope='Conv2d_0a_1x1')
				with tf.variable_scope('Branch_1'):
					branch_1 = slim.conv2d(net, 384, [1, 1], scope='Conv2d_0a_1x1')
					branch_1 = tf.concat([
						slim.conv2d(branch_1, 384, [1, 3], scope='Conv2d_0b_1x3'),
						slim.conv2d(branch_1, 384, [3, 1], scope='Conv2d_0c_3x1')], 3)
				with tf.variable_scope('Branch_2'):
					branch_2 = slim.conv2d(net, 448, [1, 1], scope='Con2d_0a_1x1')
					branch_2 = slim.conv2d(branch_2, 384, [3, 3], scope='Con2d_0b_3x3')
					branch_2 = tf.concat([
						slim.conv2d(branch_2, 384, [1, 3], scope='Conv2d_0c_1x3'),
						slim.conv2d(branch_2, 384, [3, 1], scope='Conv2d_0d_3x1')], 3)
				with tf.variable_scope('Branch_3'):
					branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
					branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
				net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
			return net, end_points

# 全局平均池化、Softmax和Auxiliary Logits
def inception_v3(inputs, 
	num_classes=1000, 
	is_training=True, 
	dropout_keep_prob=0.8, 
	prediction_fn=slim.softmax, 
	spatial_squeeze=True,
	reuse=None, 
	scope='InceptionV3'):
	with tf.variable_scope(scope, 'InceptionV3', [inputs, num_classes], reuse=reuse) as scope:
		with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
			net, end_points = inception_v3_base(inputs, scope=scope)

			# Auxiliary Logits
			with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
				aux_logits = end_points['Mixed_6e']
				with tf.variable_scope('AuxLogits'):
					aux_logits = slim.avg_pool2d(aux_logits, [5, 5], stride=3, padding='VALID', scope='AvgPool_1a_5x5')
					aux_logits = slim.conv2d(aux_logits, 128, [1, 1], scope='Conv2d_1b_1x1')
					aux_logits = slim.conv2d(aux_logits, 768, [5, 5], weights_initializer=trunc_normal(0.01),
						padding='VALID', scope='Conv2d_2a_5x5')
					aux_logits = slim.conv2d(aux_logits, num_classes, [1, 1], activation_fn=None, normalizer_fn=None,
						weights_initializer=trunc_normal(0.001), scope='Conv2d_2b_1x1')
					if spatial_squeeze:
						aux_logits = tf.squeeze(aux_logits, [1, 2], name='SpatialSqueeze')
					end_points['AuxLogits'] = aux_logits

			# 正常的分类预测的逻辑
			with tf.variable_scope('Logits'):
				net = slim.avg_pool2d(net, [8, 8], padding='VALID', scope='AvgPool_1a_8x8')
				net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout_1b')
				end_points['PreLogits'] = net
				logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='Conv2d_1c_1x1')
				if spatial_squeeze:
					logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
			end_points['Logits'] = logits
			end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
	return logits, end_points

def time_tensorflow_run(session, target, info_string):
	num_steps_burn_in = 10
	total_duration = 0.0
	total_duration_squared = 0.0

	for i in range(num_batches + num_steps_burn_in):
		start_time = time.time()
		_ = session.run(target)
		duration = time.time() - start_time
		if i >= num_steps_burn_in:
			if not i % 10:
				print('%s: step %d, duration = %.3f' % (datetime.now(), i - num_steps_burn_in, duration))
			total_duration += duration
			total_duration_squared += duration * duration

	mn = total_duration / num_batches
	vr = total_duration_squared /num_batches - mn * mn
	sd = math.sqrt(vr)
	print('%s: %s across %d steps, %.3f +/- %.3f sec / batch' % (datetime.now(), info_string, num_batches, mn, sd))

batch_size = 32
height, width = 299, 299
inputs = tf.random_uniform((batch_size, height, width, 3))
with slim.arg_scope(inception_v3_arg_scope()):
	logits, end_points = inception_v3(inputs, is_training=False)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
num_batches= 100
time_tensorflow_run(sess, logits, "Forward")