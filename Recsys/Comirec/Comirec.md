# Comirec：Controllable Multi-Interest Framework for Recommendation

论文链接：[https://arxiv.org/abs/2005.09347](https://arxiv.org/abs/2005.09347)

![](https://ai-studio-static-online.cdn.bcebos.com/05053936e26448f492ca442724b96d16237be82b90de4698b854ed5fa6b76dc8)

Comirec是阿里发表在KDD 2020上的一篇工作，这篇论文对MIND多行为召回进行了扩展，一方面改进了MIND中的动态路由算法，另一方面提出了一种新的多兴趣召回方法，同时对推荐的多样性层面也做出了一定的贡献，通过一种贪心的做法，在损失较小的推荐效果的情况下可以显著的提高推荐的多样性，从而极大的提高用户的使用体验，可以更进一步的探索用户的潜在兴趣。

这里我们将这篇论文分为上下两部分，我们在这个项目中介绍Comirec-DR模型，在下一篇教程中介绍Comirec-SA模型

## 1.1 Comirec-DR模型

![](https://ai-studio-static-online.cdn.bcebos.com/102202b9010c4b8b8519940de3fe746c5113ae2f83914cc58fd9a1081f509e49)

在这里Comirec-DR也是同样的使用了胶囊网络中的动态路由算法来对其进行多兴趣表征的提取，我们这里同样的对这里流程的每一行进行解析，在解析完这里的动态路由算法之后，我们会将Comirec-DR与MIND进行对比，通过对比可以更加深入的理解这两篇论文对动态路由的使用。

首先，我们记输入的序列为 $\{ e_i,i=1,...,r \}$,其中每一个item可以看作为一个胶囊，所提取出的多兴趣表征为 $\{ v_j,j=1,...,K \}$，其中每一个兴趣向量也可以看作是一个胶囊。很容易就知道，这里的$r$表示用户行为序列的长度，$K$表示所提取出的多兴趣向量的个数，下面我们按照上述流程图中逐一理解每一行流程的含义：
* 1.第一行对输入序列胶囊$i$与所产生的兴趣胶囊$j$的权重$b_{ij}$初始化为0
* 2.第二行开始进行动态路由，这里和MIND一样，我们同样进行三次动态路由
* 3.第三行是对每一个序列胶囊$i$对应的所有兴趣胶囊$j$的权重$\{b_{ij},j=1,...,K\}$进行Softmax归一化
* 4.第四行是对每一个兴趣胶囊$j$对应所有的序列胶囊$i$执行第四行中的计算，这里要注意$W_{ij} \in \mathbb{R}^{d \times d}$为序列胶囊$i$到兴趣胶囊$j$的映射矩阵,这样就完成了对序列到单个兴趣胶囊的特征提取，以此类推我们可以得到所有的兴趣胶囊
* 5.我们这里对4中得到的兴趣胶囊的表征通过**squash**激活函数激活

$$ squash(z_j)=\frac{{||z_{j}||}^2}{1+{||z_{j}||}^2}\frac{z_{j}}{{||z_{j}||}} $$

* 6.最后我们通过第6行中的公式来更新$b_{ij}$
* 7.至此就完成了一次动态路由，我们将这个过程重复三次就得到了完整的动态路由，也就完成了多兴趣表征的建模

## 1.2 Comirec-DR与MIND的异同

![](https://ai-studio-static-online.cdn.bcebos.com/a46c88df98e34534b6c12653162bb3fc6a4d9e78952f4a1abe897a196abed1c2)

在了解Comirec-DR的具体做法之后，可以看出，Comirec-DR与MIND的核心区别主要有两个：
* 1.$b_{ij}$的初始化方式不一样，在Comirec-DR中对$b_{ij}$全部初始化为0，在MIND中对$b_{ij}$全部用高斯分布分布进行初始化
* 2.在进行序列胶囊与兴趣胶囊之间的映射转换时的变量声明方式不一样，在Comirec-DR中对于不同的序列胶囊$i$与兴趣胶囊$j$，我们都有一个独立的$W_{ij} \in \mathbb{R}^{d \times d}$来完成序列胶囊$i$到兴趣胶囊$j$之间的映射，但是在MIND中，其提出的B2I Dynamic Routing中将所有的序列胶囊$i$与兴趣胶囊$j$的映射矩阵使用同一矩阵$S \in \mathbb{R}^{d \times d}$

在其他部分Comirec-DR与MIND就并无差别了，可以看出Comirec-DR在核心思路上与MIND时统一的，只是在个别细节上的处理稍有不同

## Comic-DR模型定义

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
class ComirecDR(nn.Layer):
    def __init__(self, config):
        super(ComirecDR, self).__init__()

        self.config = config
        self.embedding_dim = self.config['embedding_dim']
        self.max_length = self.config['max_length']
        self.n_items = self.config['n_items']

        self.item_emb = nn.Embedding(self.n_items, self.embedding_dim, padding_idx=0)
        self.capsule = CapsuleNetwork(self.embedding_dim, self.max_length, bilinear_type=2,
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
