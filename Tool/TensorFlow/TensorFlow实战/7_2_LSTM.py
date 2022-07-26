import time
import numpy as np 
import tensorflow as tf 
import reader

# 下载数据
# wget http://www.fit.vutbr.cz/~imikolov/rnn/lm/simple-examples.tgz
# tar xvf simple-examples.tgz

# 下载TensorFlow Models库，并进入目录models/tutorials/rnn/ptb,载入PTB reader，借助它读取数据内容
# 主要是将单词转为唯一的数字编码，以便神经网络处理
# git clone https://github.com/tensorflow/models.git
# cd models/tutorials/rnn/ptb

# 定义语言模型处理输入数据的class,PTBInput
class PTBInput(object):

    def __init__(self, config, data, name=None):
        self.batch_size = batch_size = config.batch_size
        # LSTM的展开步数(unrolled steps of LSTM)
        self.num_steps = num_steps = config.num_steps
        # 计算每个epoch内需要多少轮训练的迭代
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        # 用reader.ptb_producer获取特征数据input_data以及label数据targets
        # 这里的input_data和targets都已经是定义好的tensor，每次执行都会获取一个batch的数据
        self.input_data, self.targets = reader.ptb_producer(data, batch_size, num_steps, name=name)

# 定义语言模型的class, PTBModel
class PTBModel(object):

    def __init__(self, is_training, config, input_):
        self._input = input_

        batch_size = input_.batch_size
        num_steps = input_.num_steps
        size = config.hidden_size    # LSTM的节点数
        vocab_size = config.vocab_size    # 词汇表的大小

        # 设置默认的LSTM单元，隐含节点数为size,state_is_tuple=True代表接受和返回的state将是2-tuple的形式
        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)
        attn_cell = lstm_cell
        # 如果在训练状态且Dropout的keep_prob小于1，则在前面的lstm_cell之后接一个Dropout层
        if is_training and config.keep_prob < 1:
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(lstm_cell(), output_keep_prob=config.keep_prob)
        # 将前面构造的lstm_cell多层堆叠得到cell,堆叠次数为config.num_layers
        cell = tf.contrib.rnn.MultiRNNCell(
            [attn_cell() for _ in range(config.num_layers)],
            state_is_tuple=True)
        # 设置LSTM单元的初始化状态为0
        # LSTM单元可以读入一个单词并结合之前储存的状态state计算下一个单词出现的概率分布，并且每次读取一个单词后它的状态state会被更新
        self._initial_state = cell.zero_state(batch_size, tf.float32)

        # 创建网络的词嵌入部分
        with tf.device("/cpu:0"):
            # 初始化
            embedding = tf.get_variable("embedding", [vocab_size, size], dtype=tf.float32)
            inputs = tf.nn.embedding_lookup(embedding, input_.input_data)

        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        # 定义输出outputs
        outputs = []
        state = self._initial_state
        with tf.variable_scope("RNN"):
            # 设置一个循环来控制梯度的传播
            for time_step in range(num_steps):
                # 从第2次循环开始设置复用变量
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                # 传入inputs和state到堆叠的LSTM单元中
                # input有3个维度：batch中的第几个样本，样本中的第几个单词，单词的向量表达的维度
                # 这里代表所有样本的time_step个单词
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)

        # 转为一个很长的一维向量
        output = tf.reshape(tf.concat(outputs, 1), [-1, size])
        softmax_w = tf.get_variable("softmax_w", [size, vocab_size], dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=tf.float32)
        logits = tf.matmul(output, softmax_w) + softmax_b
        # sequence loss即target words的average negative log probability
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [logits], 
            [tf.reshape(input_.targets, [-1])], 
            [tf.ones([batch_size * num_steps], dtype = tf.float32)])
        # 平均到每个样本的误差
        self._cost = cost = tf.reduce_sum(loss) / batch_size
        self._final_state = state

        if not is_training:
            return

        # 定义学习速率，并设为不可训练
        self._lr = tf.Variable(0.0, trainable=False)
        # 获取全部可训练的参数
        tvars = tf.trainable_variables()
        # 计算tvars的梯度，并用tf.clip_by_global_norm设置梯度的最大范数max_grad_norm，即Gradient Clipping
        # 控制梯度的最大范数，某种程度上起到正则化的效果，Gradient Clipping可以防止梯度爆炸的问题
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        # 用optimizer.apply_gradients将前面clip过的梯度应用到所有可训练的参数tvars上
        # 然后使用tf.contrib.framwork.get_or_creat_global_step生成全局统一的训练步数
        self._train_op = optimizer.apply_gradients(zip(grads, tvars), 
            global_step=tf.contrib.framework.get_or_create_global_step())

        # 设置_new_lr用以控制学习速率
        self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    # 用来在外部控制模型的学习速率
    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    # 定义一些property,@property装饰器可以将返回变量设为只读，防止修改变量引发的问题
    @property
    def input(self):
        return self._input

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost
    
    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op

# 定义几种不同大小的模型的参数
class SmallConfig(object):
    init_scale = 0.1    # 网络中权重值的初始scale
    learning_rate = 1.0
    max_grad_norm = 5   # 梯度的最大范数
    num_layers = 2     # LSTM可以堆叠的层数
    num_steps = 20     # LSTM梯度反向传播的展开步数
    hidden_size = 200    # LSTM内的隐含节点数
    max_epoch = 4      # 初始学习速率可训练的epoch数
    max_max_epoch = 13    # 总共可训练的epoch数
    keep_prob = 1.0
    lr_decay = 0.5     # 学习速率的衰减速率
    batch_size = 20
    vocab_size = 10000

class MediumConfig(object):
    init_scale = 0.05    # 网络中权重值的初始scale,小一些有利于温和的训练
    learning_rate = 1.0
    max_grad_norm = 5   # 梯度的最大范数
    num_layers = 2     # LSTM可以堆叠的层数
    num_steps = 35     # LSTM梯度反向传播的展开步数
    hidden_size = 650    # LSTM内的隐含节点数
    max_epoch = 6      # 初始学习速率可训练的epoch数
    max_max_epoch = 39    # 总共可训练的epoch数
    keep_prob = 0.5
    lr_decay = 0.8     # 学习速率的衰减速率,因为学习的迭代次数增大，因此将衰减速率也减小了
    batch_size = 20
    vocab_size = 10000

class LargeConfig(object):
    init_scale = 0.04    # 网络中权重值的初始scale
    learning_rate = 1.0
    max_grad_norm = 10   # 梯度的最大范数
    num_layers = 2     # LSTM可以堆叠的层数
    num_steps = 35     # LSTM梯度反向传播的展开步数
    hidden_size = 1500    # LSTM内的隐含节点数
    max_epoch = 14      # 初始学习速率可训练的epoch数
    max_max_epoch = 55    # 总共可训练的epoch数
    keep_prob = 0.35
    lr_decay = 1 / 1.15     # 学习速率的衰减速率
    batch_size = 20
    vocab_size = 10000

class TestConfig(object):
    init_scale = 0.1    # 网络中权重值的初始scale
    learning_rate = 1.0
    max_grad_norm = 1   # 梯度的最大范数
    num_layers = 1     # LSTM可以堆叠的层数
    num_steps = 2     # LSTM梯度反向传播的展开步数
    hidden_size = 2    # LSTM内的隐含节点数
    max_epoch = 1      # 初始学习速率可训练的epoch数
    max_max_epoch = 1    # 总共可训练的epoch数
    keep_prob = 1.0
    lr_decay = 0.5     # 学习速率的衰减速率
    batch_size = 20
    vocab_size = 10000

# 定义训练一个epoch数据的函数run_epoch
def run_epoch(session, model, eval_op=None, verbose=False):
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = session.run(model.initial_state)    # 初始化状态

    # 输出结果的字典表
    fetches = {
    "cost": model.cost,
    "final_state": model.final_state,
    }
    # 如果有评测操作
    if eval_op is not None:
        fetches["eval_op"] = eval_op

    for step in range(model.input.epoch_size):
        feed_dict = {}
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]
        state = vals["final_state"]

        costs += cost
        iters += model.input.num_steps

        if verbose and step % (model.input.epoch_size // 10) == 10:
            # perplexity即平均cost的自然常数对数，是语言模型中用来比较模型性能的重要指标，越低代表模型输出的概率分布在预测样本上越好
            # 训练速度(单词数每秒)
            print("%.3f perplexity: %.3f speed: %.0f wps" % 
                (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
                    iters * model.input.batch_size / (time.time() - start_time)))

    # 返回perplexity
    return np.exp(costs / iters)

raw_data = reader.ptb_raw_data('simple-examples/data/')
train_data, valid_data, test_data, _ = raw_data

# 测试配置eval_config需和训练配置一致
config = SmallConfig()
eval_config = SmallConfig()
eval_config.batch_size = 1
eval_config.num_steps = 1

with tf.Graph().as_default():
    # 设置参数的初始化器，令参数范围在[-config.init_scale, config.init_scale]之间
    initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)

    # 创建一个用来训练的模型m，以及用来验证的模型mvalid和测试的模型mtest
    with tf.name_scope("Train"):
        train_input = PTBInput(config=config, data=train_data, name="TrainInput")
        with tf.variable_scope("Model", reuse=None, initializer=initializer):
            m = PTBModel(is_training=True, config=config, input_=train_input)

    with tf.name_scope("Valid"):
        valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
            mvalid = PTBModel(is_training=False, config=config, input_=valid_input)

    with tf.name_scope("Test"):
        test_input = PTBInput(config=eval_config, data=test_data, name="TestInput")
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
            mtest = PTBModel(is_training=False, config=eval_config, input_=test_input)

    # 创建训练的管理器
    sv = tf.train.Supervisor()
    # 创建默认session
    with sv.managed_session() as session:
        for i in range(config.max_max_epoch):
            # 累计的学习速率衰减值
            lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
            # 更新学习速率
            m.assign_lr(session, config.learning_rate * lr_decay)

            print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
            train_perplexity = run_epoch(session, m, eval_op=m.train_op, verbose=True)
            print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
            valid_perplexity = run_epoch(session, mvalid)
            print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

        test_perplexity = run_epoch(session, mtest)
        print("Test Perplexity: %.3f" % test_perplexity)


    
    
    