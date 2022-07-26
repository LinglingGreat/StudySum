import collections
import math
import os
import random
import zipfile
import numpy as np 
import urllib
import tensorflow as tf 

url = 'http://mattmahoney.net/dc/'

# 下载数据的压缩文件并核对文件尺寸
def maybe_download(filename, expected_bytes):
	if not os.path.exists(filename):
		filename, _ = urllib.request.urlretrieve(url + filename, filename)
	statinfo = os.stat(filename)
	if statinfo.st_size == expected_bytes:
		print('Found and verified', filename)
	else:
		print(statinfo.st_size)
		raise Exception('Failed to verify ' + filename + '. Can you get to it with a browser?')
	return filename

filename = maybe_download('text8.zip', 31344016)

# 解压下载的压缩文件，并使用tf.compat.as_str将数据转成单词的列表
def read_data(filename):
	with zipfile.ZipFile(filename) as f:
		data = tf.compat.as_str(f.read(f.namelist()[0])).split()
	return data

words = read_data(filename)
print('Data size', len(words))

# 创建词汇表，将全部单词转为编号
vocabulary_size = 50000

def build_dataset(words):
	count = [['UNK', -1]]
	# 使用collections.Counter统计单词的频数，然后使用most_common方法取top 50000频数的单词作为vocabulary
	count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
	# 将vocabulary放入dictionary中，以便快速查询，查询复杂度为O(1)
	dictionary = dict()
	for word, _ in count:
		dictionary[word] = len(dictionary)
	data = list()
	unk_count = 0
	# 将全部单词转为编号(以频数排序的编号)，top 50000词汇之外的词认定为Unknown，将其编号为0，并统计这类词汇的数量
	for word in words:
		if word in dictionary:
			index = dictionary[word]
		else:
			index = 0
			unk_count += 1
		data.append(index)
	count[0][1] = unk_count
	reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
	# 返回转换后的编码，每个单词的频数统计，词汇表及其反转的形式
	return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words)
del words
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

data_index = 0    # 单词序号

# 生成Word2Vec的训练样本，生成训练用的batch数据，num_skips为对单词生成多少个样本，不能大于skip_window的两倍，并且batch_size必须是它的整数倍
# 确保每个batch包含了一个词汇对应的所有样本
def generate_batch(batch_size, num_skips, skip_window):
	global data_index
	assert batch_size % num_skips == 0
	assert num_skips <= 2 * skip_window
	batch = np.ndarray(shape=(batch_size), dtype=np.int32)
	labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
	# span为对某个单词创建相关样本时会使用到的单词数量，包括目标单词本身和它前后的单词
	span = 2 * skip_window + 1
	# 创建一个最大容量为span的deque,即双向队列，在对deque使用append方法添加变量时，只会保留最后插入的span个变量
	buffer = collections.deque(maxlen=span)

	# 从序号data_index开始，把span个单词顺序读入buffer作为初始值
	for _ in range(span):
		buffer.append(data[data_index])
		data_index = (data_index + 1) % len(data)
	# 每次循环内对一个目标单词生成样本
	for i in range(batch_size // num_skips):
		target = skip_window   # buffer中第skip_window个变量为目标单词
		targets_to_avoid = [skip_window]    # 定义生成样本时需要避免的单词列表
		# 每次循环对一个语境单词生成样本
		for j in range(num_skips):
			# 先产生随机数，直到随机数不在targets_to_avoid中，代表可以使用的语境单词
			while target in targets_to_avoid:
				target = random.randint(0, span - 1)
			targets_to_avoid.append(target)
			# 产生一个样本，feature即目标词汇,label则是语境词汇
			batch[i * num_skips + j] = buffer[skip_window]
			labels[i * num_skips + j, 0] = buffer[target]
		# 在对一个目标单词生成完所有样本后，我们再读入下一个单词，同时会抛掉buffer中第一个单词，即把滑窗向后移动一位
		buffer.append(data[data_index])
		data_index = (data_index + 1) % len(data)
	return batch, labels

batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
	print(batch[i], reverse_dictionary[batch[i]], '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

batch_size = 128
embedding_size = 128
skip_window = 1
num_skips = 2

# 生成验证数据，随机抽取一些频数最高的单词，看向量空间上跟它们最近的单词是否相关性比较高
valid_size = 16    # 抽取的验证单词数
valid_window = 100   # 验证单词只从频数最高的100个单词中抽取
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64    # 训练时用来做负样本的噪声单词的数量

# 开始定义Skip-Gram Word2Vec模型的网络结构
graph = tf.Graph()
with graph.as_default():
	train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
	train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
	valid_datasets = tf.constant(valid_examples, dtype=tf.int32)

	with tf.device('/cpu:0'):    # 限定所有计算在CPU上执行，因为接下去的一些计算操作在GPU上可能还没有实现
		embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
		embed = tf.nn.embedding_lookup(embeddings, train_inputs)
		nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
		nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
	# 使用NCE Loss,计算学习出的词向量在训练数据上的loss,并使用tf.reduce_mean进行汇总
	loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights, biases=nce_biases, labels=train_labels, inputs=embed,
		num_sampled=num_sampled, num_classes=vocabulary_size))
	optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

	# 计算嵌入向量的L2范数norm，再将embeddings除以其L2范数得到标准化后的normalized_embeddings
	norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
	normalized_embeddings = embeddings / norm
	# 查询验证单词的嵌入向量
	valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_datasets)
	# 计算验证单词的嵌入向量欲词汇表中所有单词的相似性
	similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

	init = tf.global_variables_initializer()

num_steps = 100001
with tf.Session(graph=graph) as session:
	init.run()
	print("Initialized")

	average_loss = 0
	for step in range(num_steps):
		batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
		feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

		_, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)    # 执行一次优化器计算(即一次参数更新)和损失计算
		average_loss += loss_val

		# 每2000次循环计算一下平均loss并显示出来
		if step % 2000 == 0:
			if step > 0:
				average_loss /= 2000
			print("Average loss at step ", step, ": ", average_loss)
			average_loss = 0

		# 每10000次循环，计算一次验证单词与全部单词的相似度，并将每个验证单词最相似的8个单词展示出来
		if step % 10000 == 0:
			sim = similarity.eval()
			for i in range(valid_size):
				valid_word = reverse_dictionary[valid_examples[i]]
				top_k = 8
				nearest = (-sim[i, :]).argsort()[1:top_k+1]
				log_str = "Nearest to %s:" % valid_word
				for k in range(top_k):
					close_word = reverse_dictionary[nearest[k]]
					log_str = "%s %s," % (log_str, close_word)
				print(log_str)
	final_embeddings = normalized_embeddings.eval()

# 定义一个用来可视化Word2Vec效果的函数，low_dim_embs是降维到2维的空间向量
def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
	assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
	plt.figure(figsize=(18, 18))
	for i, label in enumerate(labels):
		x, y = low_dim_embs[i, :]
		plt.scatter(x, y)    # 展示每个单词的位置
		# 展示单词本身
		plt.annotate(label, xy=(x,y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
	plt.savefig(filename)

from sklearn.manifold import TSNE 
import matplotlib.pyplot as plt
tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
plot_only = 100
low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
labels = [reverse_dictionary[i] for i in range(plot_only)]
plot_with_labels(low_dim_embs, labels)

