import collections
import tensorflow as tf 
import time
from datetime import datetime
import math

slim = tf.contrib.slim

# 定义一个典型的Block.需要输入三个参数：scope,unit_fn,args,分别表示名称、残差学习单元、参数列表
class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
	'A named tuple describing a ResNet block.'

# 降采样，参数为输入、采样因子和scope
def subsample(inputs, factor, scope=None):
	if factor == 1:
		return inputs
	else:
		return slim.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)

# 创建卷积层
def conv2d_same(inputs, num_outputs, kernel_size, stride, scope=None):
	if stride == 1:
		return slim.conv2d(inputs, num_outputs, kernel_size, stride=1, padding='SAME', scope=scope)
	else:
		pad_total = kernel_size - 1
		pad_beg = pad_total // 2
		pad_end = pad_total - pad_beg
		inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
		return slim.conv2d(inputs, num_outputs, kernel_size, stride=stride, padding='VALID', scope=scope)

# 堆叠Blocks的函数
@slim.add_arg_scope
def stack_blocks_dense(net, blocks, outputs_collections=None):
	for block in blocks:
		with tf.variable_scope(block.scope, 'block', [net]) as sc:
			# 每个block中每个Residual Unit的args，并展开为depth,depth_bottleneck和stride
			# 比如(256, 64, 3)代表构建的bottleneck残差学习单元(每个残差学习单元包含三个卷积层)中，第三层输出通道数depth为256
			# 前两层输出通道数depth_bottleneck为64，且中间那层的步长stride为3.
			# 这个残差学习单元结构即为[(1x1/s1, 64), (3x3/s2, 64), (1x1/s1, 256)]
			for i, unit in enumerate(block.args):
				with tf.variable_scope('unit_%d' % (i+1), values=[net]):
					unit_depth, unit_depth_bottleneck, unit_stride = unit
					# 利用残差学习单元的生成函数顺序地创建并连接所有的残差学习单元
					net = block.unit_fn(net, depth=unit_depth, depth_bottleneck=unit_depth_bottleneck, stride=unit_stride)
			# 将输出 net添加到collection中
			net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)
	return net

# 定义某些函数的参数默认值
def resnet_arg_scope(is_training=True, weight_decay=0.0001, batch_norm_decay=0.997, 
	batch_norm_epsilon=1e-5, batch_norm_scale=True):
	batch_norm_params = {
	'is_training': is_training,
	'decay': batch_norm_decay,
	'epsilon': batch_norm_epsilon,
	'scale': batch_norm_scale,
	'updates_collections': tf.GraphKeys.UPDATE_OPS,
	}

	with slim.arg_scope(
		[slim.conv2d],
		weights_regularizer=slim.l2_regularizer(weight_decay),
		weights_initializer=slim.variance_scaling_initializer(),
		activation_fn=tf.nn.relu,
		normalizer_fn=slim.batch_norm,
		normalizer_params=batch_norm_params):
		with slim.arg_scope([slim.batch_norm], **batch_norm_params):
			with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
				return arg_sc

# 定义核心的bottleneck残差学习单元
@slim.add_arg_scope
def bottleneck(inputs, depth, depth_bottleneck, stride, outputs_collections=None, scope=None):
	with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
		# 获取输入的最后一个维度，即输出通道数，限定最少为4个维度
		depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
		# 对输入进行batch normalization，并使用ReLu函数进行预激活Preactivate
		preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')
		# 定义shortcut(即直连的x)
		if depth == depth_in:
			# 使用subsample按步长为stride对inputs进行空间上的降采样
			# 确保空间尺寸和残差一致，因为残差中间那层的卷积步长为stride
			shortcut = subsample(inputs, stride, 'shortcut')
		else:
			# 用步长为stride的1x1卷积改变其通道数，使得与输出通道数一致
			shortcut = slim.conv2d(preact, depth, [1, 1], stride=stride, normalizer_fn=None, 
				activation_fn=None, scope='shortcut')

		# 定义残差，有3层
		# 1x1尺寸，步长为1，输出通道数为depth_bottleneck的卷积
		# 3x3尺寸，步长为stride,输出通道数为depth_bottleneck的卷积
		# 1x1卷积，步长为1，输出通道数为depth的卷积
		residual = slim.conv2d(preact, depth_bottleneck, [1, 1], stride=1, scope='conv1')
		residual = conv2d_same(residual, depth_bottleneck, 3, stride, scope='conv2')
		residual = slim.conv2d(residual, depth, [1, 1], stride=1, normalizer_fn=None,
			activation_fn=None, scope='conv3')

		output = shortcut + residual

		return slim.utils.collect_named_outputs(outputs_collections, sc.name, output)  # 将output添加进collection

# 定义生成ResNet V2的主函数,global_pool标志是否加上最后一层全局平均池化
# include_root_block标志是否加上ResNet网络最前面通常使用的7x7卷积和最大池化
def resnet_v2(inputs, blocks, num_classes=None, global_pool=True, include_root_block=True, reuse=None, scope=None):
	with tf.variable_scope(scope, 'resnet_v2', [inputs], reuse=reuse) as sc:
		end_points_collection = sc.original_name_scope + '_end_points'
		with slim.arg_scope([slim.conv2d, bottleneck, stack_blocks_dense], outputs_collections=end_points_collection):
			net = inputs
			if include_root_block:
				with slim.arg_scope([slim.conv2d], activation_fn=None, normalizer_fn=None):
					# 输出通道为64，步长为2的7x7卷积
					net = conv2d_same(net, 64, 7, stride=2, scope='conv1')
				net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
			net = stack_blocks_dense(net, blocks)   # 将残差学习模块组生成好
			net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm')
			if global_pool:    # 添加全局平均池化，用reduce_mean实现全局平均池化，效率比直接用avg_pool高
				net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
			if num_classes is not None:
				net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='logits')
			end_points = slim.utils.convert_collection_to_dict(end_points_collection)
			if num_classes is not None:
				end_points['predictions'] = slim.softmax(net, scope='predictions')
			return net, end_points

# 设计层数为50的ResNet
def resnet_v2_50(inputs, num_classes=None, global_pool=True, reuse=None, scope='resnet_v2_50'):
	blocks = [
	Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
	Block('block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
	Block('block3', bottleneck, [(1024, 256, 1)] * 5 + [(1024, 256, 2)]),
	Block('block4', bottleneck, [(2048, 512, 1)] * 3)]
	return resnet_v2(inputs, blocks, num_classes, global_pool, include_root_block=True, reuse=reuse, scope=scope)

def resnet_v2_101(inputs, num_classes=None, global_pool=True, reuse=None, scope='resnet_v2_101'):
	blocks = [
	Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
	Block('block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
	Block('block3', bottleneck, [(1024, 256, 1)] * 22 + [(1024, 256, 2)]),
	Block('block4', bottleneck, [(2048, 512, 1)] * 3)]
	return resnet_v2(inputs, blocks, num_classes, global_pool, include_root_block=True, reuse=reuse, scope=scope)

def resnet_v2_152(inputs, num_classes=None, global_pool=True, reuse=None, scope='resnet_v2_152'):
	blocks = [
	Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
	Block('block2', bottleneck, [(512, 128, 1)] * 7 + [(512, 128, 2)]),
	Block('block3', bottleneck, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
	Block('block4', bottleneck, [(2048, 512, 1)] * 3)]
	return resnet_v2(inputs, blocks, num_classes, global_pool, include_root_block=True, reuse=reuse, scope=scope)

def resnet_v2_200(inputs, num_classes=None, global_pool=True, reuse=None, scope='resnet_v2_200'):
	blocks = [
	Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
	Block('block2', bottleneck, [(512, 128, 1)] * 23 + [(512, 128, 2)]),
	Block('block3', bottleneck, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
	Block('block4', bottleneck, [(2048, 512, 1)] * 3)]
	return resnet_v2(inputs, blocks, num_classes, global_pool, include_root_block=True, reuse=reuse, scope=scope)

def time_tensorflow_run(session, target, info_string):
	num_steps_burn_in = 10   # 预热轮数，给程序热身，头几轮迭代有显存加载、cache命中等问题
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
height, width = 224, 224
inputs = tf.random_uniform((batch_size, height, width, 3))
with slim.arg_scope(resnet_arg_scope(is_training=False)):
	net, end_points = resnet_v2_152(inputs, 1000)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
num_batches = 100
time_tensorflow_run(sess, net, "Forward")

	