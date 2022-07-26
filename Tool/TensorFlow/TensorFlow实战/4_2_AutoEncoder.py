# 去噪自编码器的实现
import numpy as np
import sklearn.preprocessing as prep 
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data


# xavier initialization的特点是会根据某一层网络的输入、输出节点数量自动调整最合适的分布
def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)


class AdditiveGaussianNoiseAutoencoder(object):
    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus, optimizer=tf.train.AdamOptimizer(), scale=0.1):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function    # 隐藏层激活函数
        self.scale = tf.placeholder(tf.float32)    # 高斯噪声系数
        self.training_scale = scale
        network_weights = self._initialize_weights()
        self.weights = network_weights

        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        # x+噪声
        self.hidden = self.transfer(tf.add(tf.matmul(self.x + scale * tf.random_normal((n_input,)),self.weights['w1']), self.weights['b1']))
        # 没有激活函数
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])

        # 损失函数：平方误差Squared Error
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        self.optimizer = optimizer.minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
        # 因为输出层没有使用激活函数，将W2,b2初始化为0即可
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
        return all_weights

    # 计算损失cost及执行一步训练的函数
    def partial_fit(self, X):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x:X, self.scale:self.training_scale})
        return cost

    # 只求cost的函数，不会触发训练操作
    def calc_total_cost(self, X):
        return self.sess.run(self.cost, feed_dict={self.x:X, self.scale:self.training_scale})

    # 返回自编码器隐含层的输出结果
    def transform(self, X):
        return self.sess.run(self.hidden, feed_dict={self.x:X, self.scale:self.training_scale})

    # 将高阶特征复原为原始数据
    def generate(self, hidden=None):
        if hidden is None:
            hidden = np.random.normal(size=self.weights["b1"])
        return self.sess.run(self.reconstruction, feed_dict={self.hidden:hidden})

    # 整体运行一遍复原过程，包括提取高阶特征和通过高阶特征复原数据
    def reconstruct(self, X):
        return self.sess.run(self.reconstruction, feed_dict={self.x:X, self.scale:self.training_scale})

    # 获取隐藏层的权重
    def getWeights(self):
        return self.sess.run(self.weights['w1'])

    # 获取隐藏层的偏置系数
    def getBiases(self):
        return self.sess.run(self.weights['b1'])

# 对训练、测试数据进行标准化处理的函数，0均值，1标准差，必须保证训练、测试数据都使用完全相同的Scaler
def standard_scale(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test

# 不放回抽样
def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data)-batch_size)
    return data[start_index:(start_index + batch_size)]


if __name__ == "__main__":
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)
    n_samples = int(mnist.train.num_examples)
    training_epochs = 20
    batch_size = 128
    display_step = 1
    autoencoder = AdditiveGaussianNoiseAutoencoder(n_input=784, n_hidden=200, transfer_function=tf.nn.softplus,
                                                    optimizer=tf.train.AdamOptimizer(learning_rate=0.001), scale=0.01)
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(n_samples / batch_size)
        for i in range(total_batch):
            batch_xs = get_random_block_from_data(X_train, batch_size)
            cost = autoencoder.partial_fit(batch_xs)
            avg_cost += cost / n_samples * batch_size

        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print("Total cost: " + str(autoencoder.calc_total_cost(X_test)))

