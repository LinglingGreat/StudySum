论文链接：[https://arxiv.org/abs/1811.00855](https://arxiv.org/abs/1811.00855)

SR-GNN是中科院提出的一种基于会话序列建模的推荐系统，这里所谓的会话是指用户的交互过程（每个会话表示一次用户行为和对应的服务，所以每个用户记录都会构建成一张图），这里说的会话序列应该是专门表示一个用户过往一段时间的交互序列。 基于会话的推荐是现在比较常用的一种推荐方式，比较常用的会话推荐包括 循环神经网络、马尔科夫链。但是这些常用的会话推荐方式有以下的两个缺点

-   当一个会话中用户的行为数量十分有限时，这种方法较难捕获用户的行为表示。比如使用RNN神经网络进行会话方式的推荐建模时，如果前面时序的动作项较少，会导致最后一个输出产生推荐项时，推荐的结果并不会怎么准确。
-   根据以往的工作发现，物品之前的转移模式在会话推荐中是十分重要的特征，但RNN和马尔科夫过程只对相邻的两个物品的单项转移向量进行 建模，而忽略了会话中其他的物品。

## 论文核心方法

![](img/Pasted%20image%2020221120152319.png)


-   1.将用户的行为序列构造成 Session Graph
-   2.我们通过GNN来对所得的 Session Graph进行特征提取，得到每一个Item的向量表征
-   3.在经过GNN提取Session Graph之后，我们需要对所有的Item的向量表征进行融合，以此得到User的向量表征 在得到了用户的向量表征之后，我们就可以按照序列召回的思路来进行模型训练/模型验证了

## 构建Session Graph
首先需要根据用户的行为序列来进行构图。这里是针对每一条用户的行为序列都需要构建一张图，我们将其视作是有向图，如果$v_2$和$v_1$在用户的行为序列里面是相邻的，并且$v_2$在$v_1$之后，则我们连出一条从$v_2$到$v_1$的边，按照这个规则我们可以构建出一张图。

在完成构图之后，我们需要使用变量来存储这张图，这里用$A_s$来表示构图的结果，这个矩阵是一个$(d,2d)$的矩阵 ，其分为一个$(d,d)$的Outing矩阵和一个$(d,d)$的Incoming矩阵。对于Outing矩阵，直接数节点向外伸出去的边的条数，如果节点向外伸出的节点数大于1，则进行归一化操作(例如节点$v_2$向外伸出了两个节点$v_3,v_4$，则节点$v_2$到节点$v_3,v_4$的值都为0.5)。Incoming矩阵同理

![](img/Pasted%20image%2020221120153022.png)

## 通过GNN学习Item的向量表征

设$v_{i}^{t}$表示在第t次GNN迭代后的item i的向量表征，$A_{s,i} \in R^{1 \times 2n}$表示$A_{s}$矩阵中的第$i$行，即代表着第$i$个item的相关邻居信息。则我们这里通过公式(1)来**对其邻居信息进行聚合**，这里主要通过矩阵$A_{s,i}$和用户的序列$[v_{1}^{t-1},...,v_{n}^{t-1}]^{T} \in R^{n \times d}$的乘法进行聚合的，不过要注意这里的公式写的不太严谨，实际情况下两个$R^{1 \times 2n}和R^{n \times d}$的矩阵是无法直接做乘法的，在代码实现中，是将矩阵A分为in和out两个矩阵分别和用户的行为序列进行乘积的

$$a_{s,i}^{t}=A_{s,i}[v_{1}^{t-1},...,v_{n}^{t-1}]^{T}\textbf{H}+b \tag{1}$$

```python
'''
A : [batch,n,2n] 图的矩阵
hidden : [batch,n,d] 用户序列的emb
in矩阵：A[:, :, :A.size(1)]
out矩阵：A[:, :, A.size(1):2 * A.size(1)]
inputs : 就是公式1中的 a 
'''
input_in = paddle.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
input_out = paddle.matmul(A[:, :, A.shape[1]:], self.linear_edge_out(hidden)) + self.b_ioh
# [batch_size, max_session_len, embedding_size * 2]
inputs = paddle.concat([input_in, input_out], 2)
```

在得到公式(1)中的$a_{s,i}^{t}$之后，根据公式(2)(3)计算出两个中间变量$z_{s,i}^{t},r_{s,i}^{t}$可以简单的类比LSTM，认为$z_{s,i}^{t},r_{s,i}^{t}$分别是遗忘门和更新门

$$z_{s,i}^{t}=\sigma(W_{z}a_{s,i}^{t}+U_{z}v_{i}^{t-1}) \in R^{d} \tag{2}$$

$$r_{s,i}^{t}=\sigma(W_{r}a_{s,i}^{t}+U_{r}v_{i}^{t-1}) \in R^{d} \tag{3}$$

这里需要注意，我们在计算$z_{s,i}^{t},r_{s,i}^{t}$的逻辑是完全一样的，唯一的区别就是用了不同的参数权重而已.
在得到公式(2)(3)的中间变量之后,我们通过公式(4)计算出更新门下一步更新的特征，以及根据公式(5)来得出最终结果

$${v_{i}^{t}}^{\sim}=tanh(W_{o}a_{s,i}^{t}+U_{o}(r_{s,i}^{t} \odot v_{i}^{t-1})) \in R^{d}\tag{4}$$

$$v_{i}^{t}=(1-z_{s,i}^{t}) \odot v_{i}^{t-1} + z_{s,i}^{t} \odot {v_{i}^{t}}^{\sim} \in R^{d} \tag{5}$$

这里我们可以看出，公式(4)实际上是计算了在第t次GNN层的时候的Update部分，也就是${v_{i}^{t}}^{\sim}$,而在公式(5)中通过遗忘门$z_{s,i}^{t}$来控制第t次GNN更新时，$v_{i}^{t-1}$和${v_{i}^{t}}^{\sim}$所占的比例。这样就完成了GNN部分的item的表征学习

这里在写代码的时候要注意，对于公式(3)(4)(5)，我们仔细观察，对于$a_{s,i}^{t},v_{i}^{t-1}$这两个变量而言，每个变量都和三个矩阵进行了相乘，这里的计算逻辑相同，所以将$Wa,Uv$当作一次计算单元，在公式(3)(4)(5)中，均涉及了一次这样的操作，所以我们可以将这三次操作放在一起做，然后在将结果切分为3份，还原三个公式，相关代码如下

```python
'''
inputs : 公式(1)中的a
hidden : 用户序列，也就是v^{t-1}
这里的gi就是Wa，gh就是Uv，但是要注意这里不管是gi还是gh都包含了公式3~5的三个部分
'''

# gi.size equals to gh.size, shape of [batch_size, max_session_len, embedding_size * 3]

gi = paddle.matmul(inputs, self.w_ih) + self.b_ih
gh = paddle.matmul(hidden, self.w_hh) + self.b_hh
# (batch_size, max_session_len, embedding_size)
i_r, i_i, i_n = gi.chunk(3, 2)   # 三个W*a
h_r, h_i, h_n = gh.chunk(3, 2)   # 三个U*v
reset_gate = F.sigmoid(i_r + h_r)  #公式(2)
input_gate = F.sigmoid(i_i + h_i)  #公式(3)
new_gate = paddle.tanh(i_n + reset_gate * h_n)  #公式(4)
hy = (1 - input_gate) * hidden + input_gate * new_gate  # 公式(5)
```

## 生成User 向量表征(Generating Session Embedding)

在通过GNN获取了Item的嵌入表征之后，我们的工作就完成一大半了，剩下的就是将用户序列的多个Item的嵌入表征融合成一个整体的序列的嵌入表征

这里SR-GNN首先利用了Attention机制来获取序列中每一个Item对于序列中最后一个Item $v_{n}(s_1)$的attention score，然后将其加权求和，其具体的计算过程如下

$$a_{i}=\textbf{q}^{T} \sigma(W_{1}v_{n}+W_{2}v_{i}+c) \in R^{1} \tag{6} \\
  s_{g}= \sum_{i=1}^{n}a_{i}v_{I}\in R^{d}$$

在得到$s_g$之后，我们将$s_g$与序列中的最后一个Item信息相结合，得到最终的序列的嵌入表征

$$s_h = W_{3}[ s_1 ;s_g] \in R^{d} \tag{7} $$

```python
'''
seq_hidden : 序列中每一个item的emb
ht ： 序列中最后一个item的emb，就是公式6~7中的v_n(s_1)
q1 : 公式(6)中的 W_1 v_n
q2 : 公式(6)中的 W_2 v_i 
alpha : 公式(6)中的alpha
a : 公式(6)中的s_g
'''
seq_hidden = paddle.take_along_axis(hidden,alias_inputs,1)
# fetch the last hidden state of last timestamp
item_seq_len = paddle.sum(mask,axis=1)
ht = self.gather_indexes(seq_hidden, item_seq_len - 1)
q1 = self.linear_one(ht).reshape([ht.shape[0], 1, ht.shape[1]])
q2 = self.linear_two(seq_hidden)

alpha = self.linear_three(F.sigmoid(q1 + q2))
a = paddle.sum(alpha * seq_hidden * mask.reshape([mask.shape[0], -1, 1]), 1)
user_emb = self.linear_transform(paddle.concat([a, ht], axis=1))
```

至此我们就完成了SR-GNN的用户向量生产了，剩下的部分就可以按照传统的序列召回的方法来进行了。

## GNN模型定义
```python
class GNN(nn.Layer):
    def __init__(self, embedding_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.embedding_size = embedding_size
        self.input_size = embedding_size * 2
        self.gate_size = embedding_size * 3
        
        self.w_ih = self.create_parameter(shape=[self.input_size, self.gate_size]) 
        self.w_hh = self.create_parameter(shape=[self.embedding_size, self.gate_size])
        self.b_ih = self.create_parameter(shape=[self.gate_size])
        self.b_hh = self.create_parameter(shape=[self.gate_size])
        self.b_iah = self.create_parameter(shape=[self.embedding_size])
        self.b_ioh = self.create_parameter(shape=[self.embedding_size])

        self.linear_edge_in = nn.Linear(self.embedding_size, self.embedding_size)
        self.linear_edge_out = nn.Linear(self.embedding_size, self.embedding_size)

    def GNNCell(self, A, hidden):
        input_in = paddle.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = paddle.matmul(A[:, :, A.shape[1]:], self.linear_edge_out(hidden)) + self.b_ioh
        # [batch_size, max_session_len, embedding_size * 2]
        inputs = paddle.concat([input_in, input_out], 2)

        # gi.size equals to gh.size, shape of [batch_size, max_session_len, embedding_size * 3]
        gi = paddle.matmul(inputs, self.w_ih) + self.b_ih
        gh = paddle.matmul(hidden, self.w_hh) + self.b_hh
        # (batch_size, max_session_len, embedding_size)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        reset_gate = F.sigmoid(i_r + h_r)
        input_gate = F.sigmoid(i_i + h_i)
        new_gate = paddle.tanh(i_n + reset_gate * h_n)
        hy = (1 - input_gate) * hidden + input_gate * new_gate
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden
```

```python
class SRGNN(nn.Layer):
    r"""SRGNN regards the conversation history as a directed graph.
    In addition to considering the connection between the item and the adjacent item,
    it also considers the connection with other interactive items.

    Such as: A example of a session sequence(eg:item1, item2, item3, item2, item4) and the connection matrix A

    Outgoing edges:
        === ===== ===== ===== =====
         \    1     2     3     4
        === ===== ===== ===== =====
         1    0     1     0     0
         2    0     0    1/2   1/2
         3    0     1     0     0
         4    0     0     0     0
        === ===== ===== ===== =====

    Incoming edges:
        === ===== ===== ===== =====
         \    1     2     3     4
        === ===== ===== ===== =====
         1    0     0     0     0
         2   1/2    0    1/2    0
         3    0     1     0     0
         4    0     1     0     0
        === ===== ===== ===== =====
    """

    def __init__(self, config):
        super(SRGNN, self).__init__()

        # load parameters info
        self.config = config
        self.embedding_size = config['embedding_dim']
        self.step = config['step']
        self.n_items = self.config['n_items']

        # define layers and loss
        # item embedding
        self.item_emb = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        # define layers and loss
        self.gnn = GNN(self.embedding_size, self.step)
        self.linear_one = nn.Linear(self.embedding_size, self.embedding_size)
        self.linear_two = nn.Linear(self.embedding_size, self.embedding_size)
        self.linear_three = nn.Linear(self.embedding_size, 1, bias_attr=False)
        self.linear_transform = nn.Linear(self.embedding_size * 2, self.embedding_size)
        self.loss_fun = nn.CrossEntropyLoss()


        # parameters initialization
        self.reset_parameters()

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
#         gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        gather_index = gather_index.reshape([-1, 1, 1])
        gather_index = paddle.repeat_interleave(gather_index,output.shape[-1],2)
        output_tensor = paddle.take_along_axis(output, gather_index, 1)
        return output_tensor.squeeze(1)

    def calculate_loss(self,user_emb,pos_item):
        all_items = self.item_emb.weight
        scores = paddle.matmul(user_emb, all_items.transpose([1, 0]))
        return self.loss_fun(scores,pos_item)

    def output_items(self):
        return self.item_emb.weight

    def reset_parameters(self, initializer=None):
        for weight in self.parameters():
            paddle.nn.initializer.KaimingNormal(weight)

    def _get_slice(self, item_seq):
        # Mask matrix, shape of [batch_size, max_session_len]
        mask = (item_seq>0).astype('int32')
        items, n_node, A, alias_inputs = [], [], [], []
        max_n_node = item_seq.shape[1]
        item_seq = item_seq.cpu().numpy()
        for u_input in item_seq:
            node = np.unique(u_input)
            items.append(node.tolist() + (max_n_node - len(node)) * [0])
            u_A = np.zeros((max_n_node, max_n_node))

            for i in np.arange(len(u_input) - 1):
                if u_input[i + 1] == 0:
                    break

                u = np.where(node == u_input[i])[0][0]
                v = np.where(node == u_input[i + 1])[0][0]
                u_A[u][v] = 1

            u_sum_in = np.sum(u_A, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A, u_sum_in)
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()
            A.append(u_A)

            alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
        # The relative coordinates of the item node, shape of [batch_size, max_session_len]
        alias_inputs = paddle.to_tensor(alias_inputs)
        # The connecting matrix, shape of [batch_size, max_session_len, 2 * max_session_len]
        A = paddle.to_tensor(A)
        # The unique item nodes, shape of [batch_size, max_session_len]
        items = paddle.to_tensor(items)

        return alias_inputs, A, items, mask

    def forward(self, item_seq, mask, item, train=True):
        if train:
            alias_inputs, A, items, mask = self._get_slice(item_seq)
            hidden = self.item_emb(items)
            hidden = self.gnn(A, hidden)
            alias_inputs = alias_inputs.reshape([-1, alias_inputs.shape[1],1])
            alias_inputs = paddle.repeat_interleave(alias_inputs, self.embedding_size, 2)
            seq_hidden = paddle.take_along_axis(hidden,alias_inputs,1)
            # fetch the last hidden state of last timestamp
            item_seq_len = paddle.sum(mask,axis=1)
            ht = self.gather_indexes(seq_hidden, item_seq_len - 1)
            q1 = self.linear_one(ht).reshape([ht.shape[0], 1, ht.shape[1]])
            q2 = self.linear_two(seq_hidden)

            alpha = self.linear_three(F.sigmoid(q1 + q2))
            a = paddle.sum(alpha * seq_hidden * mask.reshape([mask.shape[0], -1, 1]), 1)
            user_emb = self.linear_transform(paddle.concat([a, ht], axis=1))

            loss = self.calculate_loss(user_emb,item)
            output_dict = {
                'user_emb': user_emb,
                'loss': loss
            }
        else:
            alias_inputs, A, items, mask = self._get_slice(item_seq)
            hidden = self.item_emb(items)
            hidden = self.gnn(A, hidden)
            alias_inputs = alias_inputs.reshape([-1, alias_inputs.shape[1],1])
            alias_inputs = paddle.repeat_interleave(alias_inputs, self.embedding_size, 2)
            seq_hidden = paddle.take_along_axis(hidden, alias_inputs,1)
            # fetch the last hidden state of last timestamp
            item_seq_len = paddle.sum(mask, axis=1)
            ht = self.gather_indexes(seq_hidden, item_seq_len - 1)
            q1 = self.linear_one(ht).reshape([ht.shape[0], 1, ht.shape[1]])
            q2 = self.linear_two(seq_hidden)

            alpha = self.linear_three(F.sigmoid(q1 + q2))
            a = paddle.sum(alpha * seq_hidden * mask.reshape([mask.shape[0], -1, 1]), 1)
            user_emb = self.linear_transform(paddle.concat([a, ht], axis=1))
            output_dict = {
                'user_emb': user_emb,
            }
        return output_dict
```



## 参考资料

https://aistudio.baidu.com/bj-cpu-01/user/35749/5082402/lab

