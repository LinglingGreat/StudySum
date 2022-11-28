Comirec-SA是一种基于Attention的多兴趣建模方法，我们先来看如何通过Attention提取单一兴趣，随即再将其推广到多兴趣建模

## 1.1 基于Attention的单一兴趣建模

这里我们将用户的行为序列计作$H \in \mathbb{R}^{d \times n}$,这里的$n$表示用户行为序列的长度,$d$表示Embedding向量的维度，这里的$H$就代表着用户行为序列中所有Item的Embedding向量，这里我们引入两个可学习参数$W_1 \in \mathbb{R}^{d_a \times d} ,w_2 \in \mathbb{R}^{d_a}$,我们知道引入Attention的核心目的是为了学习出某些东西的权重，以此来凸显重要性搞的特征，淡化重要性低的特征，在我们的Comirec-SA中，我们的特征重要性（也就是我们学习出来的Attention Score）是针对序列中每个Item的Attention Score，在有了Attention Score之后就可以对序列中的Item进行加权求和得到序列的单一兴趣表征了，我们下面先来看单一兴趣建模时的Attention Score计算：

$$ a = {softmax(w_{2}^{T} tanh(W_{1}H))}^{T} $$

可以看出，我们所得的$a \in \mathbb{R}^{n}$，这里的$a$就是我们对序列的Attention Score，再将$a$与序列的所有Item的Embedding进行加权求和就可以得到单一兴趣的建模，下面我们将其推广到多兴趣建模上

## 1.2 基于Attention的多兴趣建模

我们在1.1中得到了单一兴趣建模的方法，我们可以把1.1中的$w_2 \in \mathbb{R}^{d_a}$ 扩充至 $W_2 \in \mathbb{R}^{d_a \times K}$,这样的话，我们的Attention Score的计算公式就变成：

$$ A = {softmax(W_{2}^{T} tanh(W_{1}H))}^{T} $$

可以得到$A  \in \mathbb{R}^{n \times K}$,这时我们可以通过

$$ V_u = HA$$

得到用户的多兴趣表征，这里的$V_u \in \mathbb{R}^{d \times K}$，即为K个兴趣表征，其核心代码如下：

```python
class MultiInterest_SA(nn.Layer):
    def __init__(self, embedding_dim, interest_num, d=None):
        super(MultiInterest_SA, self).__init__()
        self.embedding_dim = embedding_dim
        self.interest_num = interest_num
        if d == None:
            self.d = self.embedding_dim*4

        self.W1 = self.create_parameter(shape=[self.embedding_dim, self.d])
        self.W2 = self.create_parameter(shape=[self.d, self.interest_num]) 

    def forward(self, seq_emb, mask = None):
        '''
        seq_emb : batch,seq,emb
        mask : batch,seq,1
        '''
        H = paddle.einsum('bse, ed -> bsd', seq_emb, self.W1).tanh()
        mask = mask.unsqueeze(-1)
        A = paddle.einsum('bsd, dk -> bsk', H, self.W2) + -1.e9 * (1 - mask)
        A = F.softmax(A, axis=1)
        A = paddle.transpose(A,perm=[0, 2, 1])
        multi_interest_emb = paddle.matmul(A, seq_emb)
        return multi_interest_emb
```

## 1.3 多样性控制

这里论文提出了一种多样性策略，即我们希望给用户所推荐的商品的多样性更强一些，那我们该怎么衡量多样性呢，这里作者提出将Item的类别作为衡量多样性的基础，例如，我们可以通过下式来衡量两个Item i，j之间的多样性：
$$ g(i,j)= \sigma(CATE(i) \ne CATE(j)) $$
这里的$\sigma$为指示函数，这里就是如果两个item的类别不相同，那么其结果就是1，反之就是0，可以看出如果一个推荐集合中两两Item的多样性得分大的话，那么可以认为这个推荐结果中的Item的类别分布较为分散，也就可以认为推荐结果的多样性较高了，但是这个时候要注意，当推荐结果的多样性较高的时候，往往推荐的精度就会下降，这是一个Trade Off，所以我们需要一个目标来量化我们的推荐结果，这里的目标函数如下：

![](https://ai-studio-static-online.cdn.bcebos.com/0e480f9bbab7489889b09deff46e9db067ca3982a94e411fb2454b04a39a76fb)

可以看出，我们的目标函数中即包含了多样性的指标，还包含了推荐精度的指标，这里通过 $\lambda$ 来控制这两部分的占比，$\lambda$越大，可以认为对多样性的需求就越高与此同时，模型的精度可能就会小一点。在有了目标函数之后，作者这里提出使用一种贪心的做法来进行目标函数的优化，其优化的流程如下：

![](https://ai-studio-static-online.cdn.bcebos.com/9ae4f109448942289ab72a2e6e3ee3627a97fe6c24544df280940fd83020bdd6)

我们这里就不对这个多样性进行代码实现了，感兴趣的小伙伴可以查看作者原汁原味的实现:https://github.com/THUDM/ComiRec/blob/a576eed8b605a531f2971136ce6ae87739d47693/src/train.py#L50-L57


## Comirec-SA模型定义


```python
class MultiInterest_SA(nn.Layer):
    def __init__(self, embedding_dim, interest_num, d=None):
        super(MultiInterest_SA, self).__init__()
        self.embedding_dim = embedding_dim
        self.interest_num = interest_num
        if d == None:
            self.d = self.embedding_dim*4

        self.W1 = self.create_parameter(shape=[self.embedding_dim, self.d])
        self.W2 = self.create_parameter(shape=[self.d, self.interest_num]) 

    def forward(self, seq_emb, mask = None):
        '''
        seq_emb : batch,seq,emb
        mask : batch,seq,1
        '''
        H = paddle.einsum('bse, ed -> bsd', seq_emb, self.W1).tanh()
        mask = mask.unsqueeze(-1)
        A = paddle.einsum('bsd, dk -> bsk', H, self.W2) + -1.e9 * (1 - mask)
        A = F.softmax(A, axis=1)
        A = paddle.transpose(A,perm=[0, 2, 1])
        multi_interest_emb = paddle.matmul(A, seq_emb)
        return multi_interest_emb
```

```python
class ComirecSA(nn.Layer):
    def __init__(self, config):
        super(ComirecSA, self).__init__()

        self.config = config
        self.embedding_dim = self.config['embedding_dim']
        self.max_length = self.config['max_length']
        self.n_items = self.config['n_items']

        self.item_emb = nn.Embedding(self.n_items, self.embedding_dim, padding_idx=0)
        self.multi_interest_layer = MultiInterest_SA(self.embedding_dim,interest_num=self.config['K'])
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

            multi_interest_emb = self.multi_interest_layer(seq_emb, mask)  # Batch,K,Emb

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
            multi_interest_emb = self.multi_interest_layer(seq_emb, mask)  # Batch,K,Emb
            output_dict = {
                'user_emb': multi_interest_emb,
            }
        return output_dict
```
