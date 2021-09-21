import numpy as np 
import tensorflow as tf 
import gym

# 创建CartPole问题的环境env
env = gym.make('CartPole-v0')

# 先测试在CartPole环境中使用随机Action的表现，作为接下来对比的baseline
env.reset()    # 初始化环境
random_episodes = 0
reward_sum = 0
# 10次随机试验
while random_episodes < 10:
	env.render()    # 将CartPole问题的图像渲染出来
	# 执行随机的Action
	observation, reward, done, _ = env.step(np.random.randint(0, 2))
	reward_sum += reward
	# 如果done=True,代表这次试验结束，即任务失败
	if done:
		random_episodes += 1
		print("Reward for this episode was: ", reward_sum)
		reward_sum = 0
		env.reset()    # 重启环境

# 策略网络，使用简单的带有一个隐含层的MLP
H = 50    # 隐含节点数
batch_size = 25
learning_rate = 1e-1
D = 4   # 环境信息observation的维度D
gamma = 0.99   # Reward的discount比例

# 策略网络的具体结构
# 接受observations作为输入信息，最后输出一个概率值用以选择Action
observations = tf.placeholder(tf.float32, [None, D], name="input_x")
W1 = tf.get_variable("W1", shape=[D, H], initializer=tf.contrib.layers.xavier_initializer())
layer1 = tf.nn.relu(tf.matmul(observations, W1))
W2 = tf.get_variable("W2", shape=[H, 1], initializer=tf.contrib.layers.xavier_initializer())
score = tf.matmul(layer1, W2)
probability = tf.nn.sigmoid(score)

# 人工设置的虚拟label
input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")
# 每个Action的潜在价值
advantages = tf.placeholder(tf.float32, name="reward_signal")
# Action取值为1的概率为probability，取值为0的概率为1-probability,label取值与Action相反
# loglik其实就是当前Action对应的概率的对数
loglik = tf.log(input_y * (input_y - probability) + (1 - input_y) * (input_y + probability))
loss = - tf.reduce_mean(loglik * advantages)

# 获取策略网络中全部可训练的参数
tvars = tf.trainable_variables()
newGrads = tf.gradients(loss, tvars)

adam = tf.train.AdamOptimizer(learning_rate=learning_rate)
W1Grad = tf.placeholder(tf.float32, name="batch_grad1")
W2Grad = tf.placeholder(tf.float32, name="batch_grad2")
batchGrad = [W1Grad, W2Grad]
updateGrads = adam.apply_gradients(zip(batchGrad, tvars))

# 估算每一个Action对应的潜在价值discount_r
# 输入数据r为每个Action实际获得的Reward
def discount_rewards(r):
	discounted_r = np.zeros_like(r)
	# 每个Action除直接获得的Reward外的潜在价值
	running_add = 0
	for t in reversed(range(r.size)):
		# 每一个Action的潜在价值，即为后一个Action的潜在价值乘以衰减系数再加上它直接获得的reward
		running_add = running_add * gamma + r[t]
		discounted_r[t] = running_add
	return discounted_r

# xs为环境信息observation的列表,ys为我们定义的label的列表，drs为我们记录的每一个Action的Reward
xs, ys, drs = [], [], []
reward_sum = 0
episode_number = 1
total_episodes = 10000

with tf.Session() as sess:
	# 因为render会带来比较大的延迟，所以一开始不太成熟的模型还没必要去观察
	rendering = False   
	init = tf.global_variables_initializer()
	sess.run(init)

	observation = env.reset()

	# 获取所有参数模型，用来创建储存参数梯度的缓冲器gradBuffer，并将gradBuffer全部初始化为零
	gradBuffer = sess.run(tvars)
	for ix, grad in enumerate(gradBuffer):
		gradBuffer[ix] = grad * 0

	while episode_number <= total_episodes:
		# 当某个batch的平均reward达到100以上时，即Agent表现良好时，对试验环境进行展示
		if reward_sum / batch_size > 100 or rendering == True:
			env.render()
			rendering = True

		# 将observation变形为策略网络输入的格式
		x = np.reshape(observation, [1, D])

		tfprob = sess.run(probability, feed_dict={observations: x})
		action = 1 if np.random.uniform() < tfprob else 0

		xs.append(x)
		y = 1 - action
		ys.append(y)

		observation, reward, done, info = env.step(action)
		reward_sum += reward
		drs.append(reward)

		if done:
			episode_number += 1
			# 将几个列表xs,ys,drs中的元素纵向堆叠起来，得到epx,epy,epr，即为一次试验中获得的所有observation,label,reward的列表
			epx = np.vstack(xs)
			epy = np.vstack(ys)
			epr = np.vstack(drs)
			xs, ys, drs = [], [], []

			# 计算每一步Action的潜在价值，并进行标准化，得到一个零均值标准差为1的分布
			# 这么做是因为discount_reward会参与到模型损失的计算，而分布稳定的discount_reward有利于训练的稳定
			discounted_epr = discount_rewards(epr)
			discounted_epr -= np.mean(discounted_epr)
			discounted_epr /= np.std(discounted_epr)

			tGrad = sess.run(newGrads, feed_dict={observations: epx, input_y: epy, advantages: discounted_epr})
			for ix, grad in enumerate(tGrad):
				gradBuffer[ix] += grad

			if episode_number % batch_size == 0:
				# 累计了组都多的梯度，更新
				sess.run(updateGrads, feed_dict={W1Grad: gradBuffer[0], W2Grad: gradBuffer[1]})
				for ix,grad in enumerate(gradBuffer):
					gradBuffer[ix] = grad * 0

				print('Average reward for episode %d : %f.' % (episode_number, reward_sum/batch_size))

				if reward_sum / batch_size > 200:
					print("Task solved in", episode_number, 'episodes!')
					break

				reward_sum = 0

			observation = env.reset()
