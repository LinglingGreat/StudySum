#%%
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
import numpy as np
import random
import itertools
import scipy.misc
import matplotlib.pyplot as plt
import os
import tensorflow as tf


# Grid World

# 物体对象的class,包含属性：x,y坐标，尺寸，亮度值，RGB颜色通道，奖励值，名称
class gameOb():
    def __init__(self,coordinates,size,intensity,channel,reward,name):
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.size = size
        self.intensity = intensity
        self.channel = channel
        self.reward = reward
        self.name = name

# GridWorld环境的class        
class gameEnv():
    def __init__(self,size):
        self.sizeX = size
        self.sizeY = size
        self.actions = 4
        self.objects = []
        a = self.reset()
        # 展示初始的observation(即GridWorld的图像)
        plt.imshow(a,interpolation="nearest")
        
        
    def reset(self):
        # 1个hero,4个goal,2个fire
        self.objects = []
        # 用户控制的对象hero,随机选择一个没有被占用的位置,channel为2(蓝色)
        hero = gameOb(self.newPosition(),1,1,2,None,'hero')
        self.objects.append(hero)
        # reward为1的goal,channel为1(绿色)
        goal = gameOb(self.newPosition(),1,1,1,1,'goal')
        self.objects.append(goal)
        # reward为-1的fire,channel为0(红色)
        hole = gameOb(self.newPosition(),1,1,0,-1,'fire')
        self.objects.append(hole)
        goal2 = gameOb(self.newPosition(),1,1,1,1,'goal')
        self.objects.append(goal2)
        hole2 = gameOb(self.newPosition(),1,1,0,-1,'fire')
        self.objects.append(hole2)
        goal3 = gameOb(self.newPosition(),1,1,1,1,'goal')
        self.objects.append(goal3)
        goal4 = gameOb(self.newPosition(),1,1,1,1,'goal')
        self.objects.append(goal4)
        # 将GridWorld的图像绘制出来,即state
        state = self.renderEnv()
        self.state = state
        return state

    # 移动英雄角色的方法
    def moveChar(self,direction):
        # 0 - up, 1 - down, 2 - left, 3 - right
        hero = self.objects[0]
        heroX = hero.x
        heroY = hero.y
        # 如果移动到该方向会导致出界，则不移动
        if direction == 0 and hero.y >= 1:
            hero.y -= 1
        if direction == 1 and hero.y <= self.sizeY-2:
            hero.y += 1
        if direction == 2 and hero.x >= 1:
            hero.x -= 1
        if direction == 3 and hero.x <= self.sizeX-2:
            hero.x += 1     
        self.objects[0] = hero
    
    def newPosition(self):
        iterables = [ range(self.sizeX), range(self.sizeY)]
        points = []
        # itertools.product方法可以得到几个变量的所有组合
        for t in itertools.product(*iterables):
            points.append(t)
        currentPositions = []
        # 获取目前所有物体位置的集合
        for objectA in self.objects:
            if (objectA.x,objectA.y) not in currentPositions:
                currentPositions.append((objectA.x,objectA.y))
        for pos in currentPositions:
            points.remove(pos)
        location = np.random.choice(range(len(points)),replace=False)
        return points[location]

    # 检查hero是否触碰了goal或者fire
    def checkGoal(self):
        others = []
        # 获取hero,并将其它物体存入others
        for obj in self.objects:
            if obj.name == 'hero':
                hero = obj
            else:
                others.append(obj)
        # 如果有物体坐标与hero完全一致，可判定为触碰，销毁该物体，并重新生成一个该物体
        for other in others:
            if hero.x == other.x and hero.y == other.y:
                self.objects.remove(other)
                if other.reward == 1:
                    self.objects.append(gameOb(self.newPosition(),1,1,1,1,'goal'))
                else: 
                    self.objects.append(gameOb(self.newPosition(),1,1,0,-1,'fire'))
                return other.reward,False
        return 0.0,False

    def renderEnv(self):
        #a = np.zeros([self.sizeY,self.sizeX,3])
        # 创建一个长宽为size+2,颜色通道数为3的图片，初始值全部为1，代表全为白色
        a = np.ones([self.sizeY+2,self.sizeX+2,3])
        # 将最外边一圈内部的像素的颜色值全部赋为0，代表黑色
        a[1:-1,1:-1,:] = 0
        # 设置物体对象的亮度值
        for item in self.objects:
            a[item.y+1:item.y+item.size+1,item.x+1:item.x+item.size+1,item.channel] = item.intensity
        # 将图像从原始大小resize为84x84x3的尺寸，即一个正常的游戏图像尺寸
        b = scipy.misc.imresize(a[:,:,0],[84,84,1],interp='nearest')
        c = scipy.misc.imresize(a[:,:,1],[84,84,1],interp='nearest')
        d = scipy.misc.imresize(a[:,:,2],[84,84,1],interp='nearest')
        a = np.stack([b,c,d],axis=2)
        return a

    # 执行一步Action的方法
    def step(self,action):
        # 移动hero的位置
        self.moveChar(action)
        # 检测hero是否有触碰物体
        reward,done = self.checkGoal()
        # 获取环境的图像state
        state = self.renderEnv()
        return state,reward,done




# Value NetWork
env = gameEnv(size=5)


class Qnetwork():
    def __init__(self,h_size):
        #The network recieves a frame from the game, flattened into an array.
        #It then resizes it and processes it through four convolutional layers.
        self.scalarInput =  tf.placeholder(shape=[None,21168],dtype=tf.float32)
        self.imageIn = tf.reshape(self.scalarInput,shape=[-1,84,84,3])
        self.conv1 = tf.contrib.layers.convolution2d( \
            inputs=self.imageIn,num_outputs=32,kernel_size=[8,8],stride=[4,4],padding='VALID', biases_initializer=None)
        self.conv2 = tf.contrib.layers.convolution2d( \
            inputs=self.conv1,num_outputs=64,kernel_size=[4,4],stride=[2,2],padding='VALID', biases_initializer=None)
        self.conv3 = tf.contrib.layers.convolution2d( \
            inputs=self.conv2,num_outputs=64,kernel_size=[3,3],stride=[1,1],padding='VALID', biases_initializer=None)
        self.conv4 = tf.contrib.layers.convolution2d( \
            inputs=self.conv3,num_outputs=512,kernel_size=[7,7],stride=[1,1],padding='VALID', biases_initializer=None)
        
        # Dueling DQN
        #We take the output from the final convolutional layer and split it into separate advantage and value streams.
        # 将第4个卷积层的输出在第3个维度上平均拆成两段，即Advantage Function(Action带来的价值)和Value Function(环境本身的价值)
        self.streamAC,self.streamVC = tf.split(self.conv4,2,3)
        self.streamA = tf.contrib.layers.flatten(self.streamAC)
        self.streamV = tf.contrib.layers.flatten(self.streamVC)
        # 创建线性全连接层参数
        self.AW = tf.Variable(tf.random_normal([h_size//2,env.actions]))
        self.VW = tf.Variable(tf.random_normal([h_size//2,1]))
        self.Advantage = tf.matmul(self.streamA,self.AW)
        self.Value = tf.matmul(self.streamV,self.VW)
        
        #Then combine them together to get our final Q-values.
        # value加上减去均值的advantage,reduction_indices=1代表Action数量的维度
        self.Qout = self.Value + tf.subtract(self.Advantage,tf.reduce_mean(self.Advantage,reduction_indices=1,keep_dims=True))
        # 输出Q值最大的Action
        self.predict = tf.argmax(self.Qout,1)
        
        # Double DQN中的目标Q值targetQ以及Agent的动作actions
        # 在计算目标Q值时，action由主DQN选择，Q值则由辅助的target DQN生成
        # 在计算预测Q值时，，将scalar形式的actions转为onehot编码的形式，然后将主DQN生成的Qout乘以actions_onehot，得到预测Q值
        # Qout和actions都来自主DQN
        #Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None],dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions,env.actions,dtype=tf.float32)
        
        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), reduction_indices=1)
        
        # 定义loss
        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)
        

# 实现Experience Replay策略
class experience_buffer():
    def __init__(self, buffer_size = 50000):
        self.buffer = []
        self.buffer_size = buffer_size    # 存储样本的最大容量
    
    # 向buffer中添加元素的方法
    def add(self,experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience)+len(self.buffer))-self.buffer_size] = []
        self.buffer.extend(experience)
            
    # 对样本进行抽样的方法
    def sample(self,size):
        return np.reshape(np.array(random.sample(self.buffer,size)),[size,5])
        
        
# 将84x84x3的states扁平化为1维向量的函数
def processState(states):
    return np.reshape(states,[21168])


# 更新target DQN模型参数的方法(主DQN则是直接使用DQN class中的self.updateModel方法更新模型参数)
# 输入TensorFlow Graph中的全部参数，target DQN向主DQN学习的速率    
def updateTargetGraph(tfVars,tau):
    total_vars = len(tfVars)
    op_holder = []
    # 取tfVars中前一半参数，即主DQN的模型参数，再令辅助的target DQN的参数朝向主DQN的参数前进一个很小的比例tau,让target DQN缓慢地学习主DQN
    for idx,var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
    return op_holder

def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)



batch_size = 32 #How many experiences to use for each training step.
update_freq = 4 #How often to perform a training step.
y = .99 #Discount factor on the target Q-values
startE = 1 #Starting chance of random action 起始的执行随机Action的概率
endE = 0.1 #Final chance of random action
anneling_steps = 10000. #How many steps of training to reduce startE to endE.
num_episodes = 10000#How many episodes of game environment to train network with.
pre_train_steps = 10000 #How many steps of random actions before training begins.
max_epLength = 50 #The max allowed length of our episode. 每个episode进行多少步Action
load_model = False #Whether to load a saved model.
path = "./dqn" #The path to save our model to.
h_size = 512 #The size of the final convolutional layer before splitting it into Advantage and Value streams.
tau = 0.001 #Rate to update target network toward primary network


# 初始化
tf.reset_default_graph()
mainQN = Qnetwork(h_size)
targetQN = Qnetwork(h_size)

init = tf.global_variables_initializer()

trainables = tf.trainable_variables()
targetOps = updateTargetGraph(trainables,tau)

myBuffer = experience_buffer()

#Set the rate of random action decrease. 
e = startE
# e在每一步应该衰减的值
stepDrop = (startE - endE)/anneling_steps

#create lists to contain total rewards and steps per episode
rList = []
total_steps = 0

#Make a path for our model to be saved in.
saver = tf.train.Saver()
if not os.path.exists(path):
    os.makedirs(path)
#%%
with tf.Session() as sess:
    if load_model == True:
        print('Loading Model...')
        # 检查模型文件路径的checkpoint
        ckpt = tf.train.get_checkpoint_state(path)
        saver.restore(sess,ckpt.model_checkpoint_path)
    sess.run(init)
    updateTarget(targetOps,sess) #Set the target network to be equal to the primary network.
    # 进行GridWorld试验的循环
    for i in range(num_episodes+1):
        # 创建每个episode内部的experience_buffer,这些内部的buffer不会参与当前迭代的训练，训练只会使用之前episode的样本
        episodeBuffer = experience_buffer()
        #Reset environment and get first new observation
        s = env.reset()
        s = processState(s)    # 扁平化
        d = False    # done标记
        rAll = 0     # episode内总reward值
        j = 0    # episode内的步数
        #The Q-Network
        # 每次迭代执行一个Action
        while j < max_epLength: #If the agent takes longer than 200 moves to reach either of the blocks, end the trial.
            j+=1
            #Choose an action by greedily (with e chance of random action) from the Q-network
            if np.random.rand(1) < e or total_steps < pre_train_steps:
                a = np.random.randint(0,4)
            else:
                # 传入当前状态s给主DQN，预测得到应该执行的Action
                a = sess.run(mainQN.predict,feed_dict={mainQN.scalarInput:[s]})[0]
            s1,r,d = env.step(a)    # 执行一步action，得到状态，reward和done标记
            s1 = processState(s1)
            total_steps += 1
            episodeBuffer.add(np.reshape(np.array([s,a,r,s1,d]),[1,5])) #Save the experience to our episode buffer.
            
            if total_steps > pre_train_steps:
                # 持续降低随机选择Action的概率e,直到达到最低值endE
                if e > endE:
                    e -= stepDrop
                
                if total_steps % (update_freq) == 0:
                    trainBatch = myBuffer.sample(batch_size) #Get a random batch of experiences.
                    #Below we perform the Double-DQN update to the target Q-values
                    # 将训练样本中第3列信息即下一个状态s1，传入mainQN并执行mainQN.predict,得到主模型选择的action
                    A = sess.run(mainQN.predict,feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:,3])})
                    # 再将s1传入辅助的targetQN,并得到s1状态下所有Action的Q值
                    Q = sess.run(targetQN.Qout,feed_dict={targetQN.scalarInput:np.vstack(trainBatch[:,3])})
                    doubleQ = Q[range(batch_size),A]
                    # 当前的reward加上doubleQ乘以衰减系数y,得到学习目标
                    targetQ = trainBatch[:,2] + y*doubleQ
                    #Update the network with our target values.
                    # 传入当前的状态s，学习目标targetQ和这一步实际采取的Action,执行updateModel操作更新一次主模型的参数(即执行一次训练操作)
                    _ = sess.run(mainQN.updateModel, \
                        feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:,0]),mainQN.targetQ:targetQ, mainQN.actions:trainBatch[:,1]})
                    # 同时也执行一次targetQN模型参数的更新(缓慢地向mainQN学习)
                    updateTarget(targetOps,sess) #Set the target network to be equal to the primary network.
            # 每个step结束时，累计当前这步获取的reward,并更新当前状态为下一步试验做准备
            rAll += r
            s = s1
            
            if d == True:
                break
        
        #Get all experiences from this episode and discount their rewards.
        myBuffer.add(episodeBuffer.buffer)
        rList.append(rAll)
        #Periodically save the model.

        if i>0 and i % 25 == 0:
            print('episode',i,', average reward of last 25 episode',np.mean(rList[-25:]))
        if i>0 and i % 1000 == 0:
            saver.save(sess,path+'/model-'+str(i)+'.cptk')
            print("Saved Model")            
    saver.save(sess,path+'/model-'+str(i)+'.cptk')


#%%
# 计算每100个episodes的平均reward，并使用plt.plot展示reward变化的趋势
rMat = np.resize(np.array(rList),[len(rList)//100,100])
rMean = np.average(rMat,1)
plt.plot(rMean)
