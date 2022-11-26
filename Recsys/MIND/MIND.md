MIND：Multi-Interest Network with Dynamic Routing for Recommendation at Tmall

论文链接：https://arxiv.org/pdf/1904.08030

改论文是阿里巴巴2019年在CIKM上发表的一篇关于多兴趣召回的论文，改论文属于U2I召回的一种，传统的召回方法往往针对User的行为序列只生产单一的向量，但是在用户行为序列中往往蕴含着多种兴趣，生产单一的用户的嵌入向量无法很好的对用户的行为进行建模，所以阿里巴巴提出了对用户的行为序列生产多个向量以表示用户的多个兴趣表征，从而大大的提高了用户的行为表征能力.

![](img/Pasted%20image%2020221126220743.png)


## 问题定义
我们将$I_{u}$计作是User$u$的行为序列，$P_{u}$计作是User$u$的基本特征(例如年龄，性别等),$F_{i}$为Item $i$ 的基本特征(例如item_id,cate_id等)，则我们的目标就是学习一个函数$f_{user}$使得:
$$V_{u}=f_{user}(I_{u},P_{u})$$
这里的$V_{u}=(\mathop{v_{u}^{1}}\limits ^{\rightarrow},...,\mathop{v_{u}^{K}}\limits ^{\rightarrow}) \in \mathbb{R}^{d \times K}$,其中d表示User的Embedding向量的维度，K代表着所提取出的用户的兴趣个数，当$K=1$的时候，其退化为正常的U2I序列召回，对于Item我们也需要对其学习一个函数$f_{item}$：
$$\mathop{e_{i}}\limits^{\rightarrow}=f_{item}(F_{i})$$
这里的$\mathop{e_{i}}\limits^{\rightarrow} \in \mathbb{R}^{d \times 1}$，表示每个Item的Embedding向量表征。在获取了User和Item的向量表征之后，那我们该如何对User的多兴趣表征与指定的Item进行打分呢，我们对其采用以下公式进行打分：
$$ f_{score}(V_{u},\mathop{e_{i}}\limits^{\rightarrow})=\mathop{max}\limits_{1\leq k \leq K} {\mathop{e_{i}}\limits^{\rightarrow}}^\mathrm{T}  \mathop{v_{u}^{k}}\limits ^{\rightarrow} $$

实际上可以看出，这里只是对User的所有兴趣向量挨个与Item的向量进行内积，从所有的内积结果中挑出最大的得分作为最终的得分

## Embedding & Pooling Layer

从上图可以看出，我们模型的输入有三部分组成，也就是我们在上一小节中的$I_{u},P_{u},F_{i}$,这里对所有的离散特征都使用Embedding Layer获取其Embedding向量，但是对于User/Item而言都是由一个或者多个Embedding向量组成，这里如果User/Item的向量表征有多个的话，我们对这些向量进行**mean pooling**操作，将其变成单一的Embedding向量

## Multi-Interest Extractor Layer

在获取了用户行为序列的Embedding之后，我们就需要对用户的行为序列的Embedding向量进行多兴趣建模了。

MIND中对于多兴趣建模师参考了胶囊网络中的动态路由的方法来做的，这里将输入的用户的历史行为序列中的每一个Item当作一个胶囊，将所提取出的多兴趣向量的每一个兴趣向量作为所提取出的胶囊，这里我们将用户的行为序列(胶囊)计作 $\{\mathop{e_{i}}\limits^{\rightarrow},i \in I_{u} \}$,将所得的用户的多兴趣向量(胶囊)计作 $\{\mathop{u_{j}}\limits^{\rightarrow}\,j=1,...,K_{u}^{'}\}$ ,其主要的算法流程如下所示:

![](https://ai-studio-static-online.cdn.bcebos.com/33e5df724b6846069ba389aac509c4772a157b80a3e349059df46a8ed3a7ad4e)


我们针对针对该流程进行逐一的解释，首先流程的第一行，我们需要根据输入的用户User的长度来动态的计算所提取的User的兴趣向量个数，可以简单的认为一个User的行为序列越长，其所包含的潜在兴趣的个数就越多，这里用户兴趣向量的计算公式如下：
$$ K_{u}^{'} = max(1,min(K,{log}_{2}I_{u})) $$
这里的$K$是我们设置的一个超参数，代表着最小兴趣的个数,当然了，我们也可以直接给所有用户都指定提取出固定个数的兴趣向量，这一部分在作者的实验部分会有体现。第2行就是我们需要给User的行为胶囊 $i$ 和User的兴趣胶囊 $j$ 之间声明一个系数 $b_{ij}$ ,这里我们对 $b_{ij}$ 使用高斯分布进行初始化。第三行开始是一个for循环，这里是对动态路由的次数进行for循环，在原论文中，这里是循环三次，即做三次动态路由。下面我们来具体分析一下每一次动态路由需要做什么操作
* 1.首先对 $b_{ij}$ 进行softmax，这里要注意我们softmax的维度是-1，这也就是让每个User的行为胶囊i的所有兴趣胶囊j的和为1，即：
$$ w_{ij}=\frac{exp(b_{ij})}{\sum_{i=1}^{m}exp(b_{ik})} $$
* 2.在得到 $w_{ij}$ 之后，我们来生成兴趣胶囊 $z_{j}$ ,其生成方法就是遍历的User行为序列，对每一个User的行为胶囊i执行第五行的公式，这样就可以将user行为序列的信息聚合到每一个兴趣胶囊中了。这里要注意 $S \in \mathbb{R}^{d \times d}$ 是一个可学习参数， $S$  相当于是对User的行为胶囊和兴趣胶囊进行信息交互融合的一个权重。
* 3.在得到兴趣胶囊 $z_j$ 之后，我们对其使用**squash**激活函数,其公式如下:
$$ squash(z_j)=\frac{{||z_{j}||}^2}{1+{||z_{j}||}^2}\frac{z_{j}}{{||z_{j}||}} $$
这里的 $||z_{j}||$ 代表向量的模长
* 4.最后一步我们需要来根据第七行中的公式来更新所有的 $b_{ij}$ ,然后继续重复这一循环过程，直到达到预设的循环次数。

在完成动态路由之后，我们就针对User的行为序列得到了他的多兴趣表征了，下面我们来看一下怎么根据用户的多兴趣表征来进行模型的训练

## Label-aware Attention Layer

我们在得到了用户的多兴趣表征之后，我们就要对其进行Loss计算，以此来进行模型的反向传播了，那这个时候我们手里有的信息其实只有两个，第一个就是我们上一小节刚刚得到的用户的多兴趣向量 $V_{u} \in \mathbb{R}^{d \times K}$ ,另一个就是我们User的“标签”，也就是User下一个点击的Item，我们同样也可以获取到这一个Item的Embedding向量 $\mathop{e_{i}}\limits^{\rightarrow} \in \mathbb{R}^{d}$ ,对于传统的序列召回，我们得到的只有单一的User/Item的Embedding向量，这时我们可以通过Sample Softmax，Softmax或者是一些基于Pair-Wise的方法来对其进行Loss的计算，但是我们这里对于User的Embedding表征的向量是有多个，这里就涉及到了怎么把User的多兴趣表征重新整合成单一的向量了，这里作者进行了详细的实验与讨论。

首先作者通过attention计算出各个兴趣向量对目标item的权重，用于将多兴趣向量融合为用户向量，其中p为可调节的参数
$$ \mathop{v_{u}}\limits^{\rightarrow} = Attention(\mathop{e_{i}}\limits^{\rightarrow},V_{u},V_{u})=V_{u}softmax(pow(V_{u}^{T}\mathop{e_{i}}\limits^{\rightarrow},p)) $$
可以看出，上述式子中只有 $p$ 为超参数，下面作者对 $p$ 的不同取值进行了讨论
* 1.当p=0的时候,每个用户兴趣向量有相同的权重,相当于对所有的User兴趣向量做了一个Mean Pooling
* 2.当p>1的时候，p越大，与目标商品向量点积更大的用户兴趣向量会有更大的权重
* 3.当p趋于无穷大的时候，实际上就相当于只使用与目标商品向量点积最大的用户兴趣向量，忽略其他用户向量，可以理解是每次只激活一个兴趣向量

在作者的实际实验中发现，当p趋于无穷大的时候模型的训练效果是最好的，即在进行Loss计算的时候，我们只对和目标Item最相似的那个兴趣向量进行Loss计算，这样我们就将多兴趣模型的Loss计算重新转换为一个一般的序列召回的问题，这里作者采用了Sample-Softmax当作损失函数，这里的Sample-Softmax可以认为是Softmax的简化版，其核心也是一个多分类任务，但是当Item的个数很大的时候，直接当作多分类做会有很大的计算压力，所以这里引出了Sample-Softmax，这里的Sample-Softmax是对负样本进行了随机采样，这样就可以极大的缩小多分类的总类别数，也就可以在大规模的推荐数据上进行训练了，在后面的代码实践中，由于我们使用的是公开数据集，其Item的个数比较小，所以我们就没有使用Sample-Softmax而是直接使用了Softmax

## 代码实践

```python
class CapsuleNetwork(nn.Layer):

    def __init__(self, hidden_size, seq_len, bilinear_type=2, interest_num=4, routing_times=3, hard_readout=True,
                 relu_layer=False):
        super(CapsuleNetwork, self).__init__()
        self.hidden_size = hidden_size  # h
        self.seq_len = seq_len  # s
        self.bilinear_type = bilinear_type
        self.interest_num = interest_num
        self.routing_times = routing_times
        self.hard_readout = hard_readout
        self.relu_layer = relu_layer
        self.stop_grad = True
        self.relu = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size, bias_attr=False),
            nn.ReLU()
        )
        if self.bilinear_type == 0:  # MIND
            self.linear = nn.Linear(self.hidden_size, self.hidden_size, bias_attr=False)
        elif self.bilinear_type == 1:
            self.linear = nn.Linear(self.hidden_size, self.hidden_size * self.interest_num, bias_attr=False)
        else:  # ComiRec_DR
            self.w = self.create_parameter(
                shape=[1, self.seq_len, self.interest_num * self.hidden_size, self.hidden_size])

    def forward(self, item_eb, mask):
        if self.bilinear_type == 0:  # MIND
            item_eb_hat = self.linear(item_eb)  # [b, s, h]
            item_eb_hat = paddle.repeat_interleave(item_eb_hat, self.interest_num, 2) # [b, s, h*in]
        elif self.bilinear_type == 1:
            item_eb_hat = self.linear(item_eb)
        else:  # ComiRec_DR
            u = paddle.unsqueeze(item_eb, 2)  # shape=(batch_size, maxlen, 1, embedding_dim)
            item_eb_hat = paddle.sum(self.w[:, :self.seq_len, :, :] * u,
                                    3)  # shape=(batch_size, maxlen, hidden_size*interest_num)

        item_eb_hat = paddle.reshape(item_eb_hat, (-1, self.seq_len, self.interest_num, self.hidden_size))
        item_eb_hat = paddle.transpose(item_eb_hat, perm=[0,2,1,3])
        # item_eb_hat = paddle.reshape(item_eb_hat, (-1, self.interest_num, self.seq_len, self.hidden_size))

        # [b, in, s, h]
        if self.stop_grad:  # 截断反向传播，item_emb_hat不计入梯度计算中
            item_eb_hat_iter = item_eb_hat.detach()
        else:
            item_eb_hat_iter = item_eb_hat

        # b的shape=(b, in, s)
        if self.bilinear_type > 0:  # b初始化为0（一般的胶囊网络算法）
            capsule_weight = paddle.zeros((item_eb_hat.shape[0], self.interest_num, self.seq_len))
        else:  # MIND使用高斯分布随机初始化b
            capsule_weight = paddle.randn((item_eb_hat.shape[0], self.interest_num, self.seq_len))

        for i in range(self.routing_times):  # 动态路由传播3次
            atten_mask = paddle.repeat_interleave(paddle.unsqueeze(mask, 1), self.interest_num, 1) # [b, in, s]
            paddings = paddle.zeros_like(atten_mask)

            # 计算c，进行mask，最后shape=[b, in, 1, s]
            capsule_softmax_weight = F.softmax(capsule_weight, axis=-1)
            capsule_softmax_weight = paddle.where(atten_mask==0, paddings, capsule_softmax_weight)  # mask
            capsule_softmax_weight = paddle.unsqueeze(capsule_softmax_weight, 2)

            if i < 2:
                # s=c*u_hat , (batch_size, interest_num, 1, seq_len) * (batch_size, interest_num, seq_len, hidden_size)
                interest_capsule = paddle.matmul(capsule_softmax_weight,
                                                item_eb_hat_iter)  # shape=(batch_size, interest_num, 1, hidden_size)
                cap_norm = paddle.sum(paddle.square(interest_capsule), -1, keepdim=True)  # shape=(batch_size, interest_num, 1, 1)
                scalar_factor = cap_norm / (1 + cap_norm) / paddle.sqrt(cap_norm + 1e-9)  # shape同上
                interest_capsule = scalar_factor * interest_capsule  # squash(s)->v,shape=(batch_size, interest_num, 1, hidden_size)

                # 更新b
                delta_weight = paddle.matmul(item_eb_hat_iter,  # shape=(batch_size, interest_num, seq_len, hidden_size)
                                            paddle.transpose(interest_capsule, perm=[0,1,3,2])
                                            # shape=(batch_size, interest_num, hidden_size, 1)
                                            )  # u_hat*v, shape=(batch_size, interest_num, seq_len, 1)
                delta_weight = paddle.reshape(delta_weight, (
                -1, self.interest_num, self.seq_len))  # shape=(batch_size, interest_num, seq_len)
                capsule_weight = capsule_weight + delta_weight  # 更新b
            else:
                interest_capsule = paddle.matmul(capsule_softmax_weight, item_eb_hat)
                cap_norm = paddle.sum(paddle.square(interest_capsule), -1, keepdim=True)
                scalar_factor = cap_norm / (1 + cap_norm) / paddle.sqrt(cap_norm + 1e-9)
                interest_capsule = scalar_factor * interest_capsule

        interest_capsule = paddle.reshape(interest_capsule, (-1, self.interest_num, self.hidden_size))

        if self.relu_layer:  # MIND模型使用book数据库时，使用relu_layer
            interest_capsule = self.relu(interest_capsule)

        return interest_capsule
```

```python
class MIND(nn.Layer):
    def __init__(self, config):
        super(MIND, self).__init__()

        self.config = config
        self.embedding_dim = self.config['embedding_dim']
        self.max_length = self.config['max_length']
        self.n_items = self.config['n_items']

        self.item_emb = nn.Embedding(self.n_items, self.embedding_dim, padding_idx=0)
        self.capsule = CapsuleNetwork(self.embedding_dim, self.max_length, bilinear_type=0,
                                      interest_num=self.config['K'])
        self.loss_fun = nn.CrossEntropyLoss()
        self.reset_parameters()

    def calculate_loss(self,user_emb,pos_item):
        all_items = self.item_emb.weight
        scores = paddle.matmul(user_emb, all_items.transpose([1, 0]))
        return self.loss_fun(scores,pos_item)

    def output_items(self):
        return self.item_emb.weight

    def reset_parameters(self, initializer=None):
        for weight in self.parameters():
            paddle.nn.initializer.KaimingNormal(weight)

    def forward(self, item_seq, mask, item, train=True):

        if train:
            seq_emb = self.item_emb(item_seq)  # Batch,Seq,Emb
            item_e = self.item_emb(item).squeeze(1)

            multi_interest_emb = self.capsule(seq_emb, mask)  # Batch,K,Emb

            cos_res = paddle.bmm(multi_interest_emb, item_e.squeeze(1).unsqueeze(-1))
            k_index = paddle.argmax(cos_res, axis=1)

            best_interest_emb = paddle.rand((multi_interest_emb.shape[0], multi_interest_emb.shape[2]))
            for k in range(multi_interest_emb.shape[0]):
                best_interest_emb[k, :] = multi_interest_emb[k, k_index[k], :]

            loss = self.calculate_loss(best_interest_emb,item)
            output_dict = {
                'user_emb': multi_interest_emb,
                'loss': loss,
            }
        else:
            seq_emb = self.item_emb(item_seq)  # Batch,Seq,Emb
            multi_interest_emb = self.capsule(seq_emb, mask)  # Batch,K,Emb
            output_dict = {
                'user_emb': multi_interest_emb,
            }
        return output_dict

```

