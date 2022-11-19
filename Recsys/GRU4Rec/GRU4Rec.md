背景知识RNN和GRU：[RNN及其应用](../../BasicKnow/RNN/RNN及其应用.md)


通过RNN提取出输入序列的特征有两种做法：

* 取出$y^n$的向量表征作为序列的特征，这里可以认为 $y^n$ 包含了 $x^1,x^2,...,x^n$ ,的所有信息，所有可以简单的认为$y^n$的结果代表序列的表征
* 对每一个时间步的特征输出做一个**Mean Pooling**，也就是对$Y = [y^1,y^2,...,y^n]$ 做均值处理，以此得到序列的表征

基于GRU的序列召回

## 数据预处理
```python
class SeqnenceDataset(Dataset):
    def __init__(self, config, df, phase='train'):
        self.config = config
        self.df = df
        self.max_length = self.config['max_length']
        self.df = self.df.sort_values(by=['user_id', 'timestamp'])
        self.user2item = self.df.groupby('user_id')['item_id'].apply(list).to_dict()
        self.user_list = self.df['user_id'].unique()
        self.phase = phase

    def __len__(self, ):
        return len(self.user2item)

    def __getitem__(self, index):
        if self.phase == 'train':
            user_id = self.user_list[index]
            item_list = self.user2item[user_id]
            hist_item_list = []
            hist_mask_list = []

            k = random.choice(range(4, len(item_list)))  # 从[8,len(item_list))中随机选择一个index
            # k = np.random.randint(2,len(item_list))
            item_id = item_list[k]  # 该index对应的item加入item_id_list

            if k >= self.max_length:  # 选取seq_len个物品
                hist_item_list.append(item_list[k - self.max_length: k])
                hist_mask_list.append([1.0] * self.max_length)
            else:
                hist_item_list.append(item_list[:k] + [0] * (self.max_length - k))
                hist_mask_list.append([1.0] * k + [0.0] * (self.max_length - k))

            return paddle.to_tensor(hist_item_list).squeeze(0), paddle.to_tensor(hist_mask_list).squeeze(
                0), paddle.to_tensor([item_id])
        else:
            user_id = self.user_list[index]
            item_list = self.user2item[user_id]
            hist_item_list = []
            hist_mask_list = []

            k = int(0.8 * len(item_list))
            # k = len(item_list)-1

            if k >= self.max_length:  # 选取seq_len个物品
                hist_item_list.append(item_list[k - self.max_length: k])
                hist_mask_list.append([1.0] * self.max_length)
            else:
                hist_item_list.append(item_list[:k] + [0] * (self.max_length - k))
                hist_mask_list.append([1.0] * k + [0.0] * (self.max_length - k))

            return paddle.to_tensor(hist_item_list).squeeze(0), paddle.to_tensor(hist_mask_list).squeeze(
                0), item_list[k:]

    def get_test_gd(self):
        self.test_gd = {}
        for user in self.user2item:
            item_list = self.user2item[user]
            test_item_index = int(0.8 * len(item_list))
            self.test_gd[user] = item_list[test_item_index:]
        return self.test_gd

```

## 模型训练部分

我们这里首先对Item进行Embedding操作，这里的Embedding层可以认为是一张表，他内部存储的是每一个Item ID到其向量表征的映射，例如：我有10个Item，我想对每个Item表征成一个4维的向量，那么我们可以有如下的Embedding层:

```python
emb_layer = nn.Embedding(10,4) #声明一个10个Item，维度为4的Embedding表
query_index = paddle.to_tensor([1,0,1,2,3]) #比如我查询index为 [1,0,1,2,3]的向量

print(emb_layer.weight) # 可以看出embedding内部的存储是一个10x4的二维矩阵，其中每一行都是一个4维的向量，也就是一个item的向量表征
print(emb_layer(query_index)) # 查询结果确实是对应index对应在embedding里面的行向量
```
输出为:
```bash
Parameter containing:
Tensor(shape=[10, 4], dtype=float32, place=Place(cpu), stop_gradient=False,
       [[ 0.48999012,  0.23852265, -0.24145952,  0.57672238],
        [-0.09576172,  0.08986044, -0.63121289, -0.02598906],
        [-0.44023734,  0.31829000, -0.65259022, -0.31957576],
        [ 0.37807786, -0.14285791, -0.29132205,  0.50795472],
        [ 0.49052703, -0.49909633, -0.55534846,  0.17601246],
        [-0.49354345,  0.61451089,  0.12685758,  0.37117445],
        [ 0.62036407, -0.59030831, -0.55749607, -0.58575040],
        [ 0.18010908,  0.34986722, -0.10237777, -0.34165010],
        [ 0.17282718, -0.58883876, -0.33249515,  0.11425638],
        [-0.01826757,  0.17947799, -0.21948734, -0.17575613]])
Tensor(shape=[5, 4], dtype=float32, place=Place(cpu), stop_gradient=False,
       [[-0.09576172,  0.08986044, -0.63121289, -0.02598906],
        [ 0.48999012,  0.23852265, -0.24145952,  0.57672238],
        [-0.09576172,  0.08986044, -0.63121289, -0.02598906],
        [-0.44023734,  0.31829000, -0.65259022, -0.31957576],
        [ 0.37807786, -0.14285791, -0.29132205,  0.50795472]])
```

在获取Item的Embedding向量之后，我们就对序列进行GRU特征提取，我们选择的是GRU输出的最后一个节点的向量，我们将这个向量作为序列整体的特征表达，核心代码如下：
```python
seq_emb = self.item_emb(item_seq)
seq_emb,_ = self.gru(seq_emb)
user_emb = seq_emb[:,-1,:] #取GRU输出的最后一个Hidden作为User的Embedding
```
在得到用户的向量表征之后就好办了，这里直接通过多分类进行损失计算，多分类的标签就是用户下一次点击的Item的index，我们直接通过User的向量表征和所有的Item的向量做内积算出User对所有Item的点击概率，然后通过**Softmax**进行多分类损失计算，核心代码如下：
```python
def calculate_loss(self,user_emb,pos_item):
    all_items = self.item_emb.weight
    scores = paddle.matmul(user_emb, all_items.transpose([1, 0]))
    return self.loss_fun(scores,pos_item)
```
在可以计算模型的Loss之后，我们就可以开始训练模型了～～～

## 模型验证

数据集是：Movielens-20M，我们对其按照User进行了数据划分，按照8:1:1的比例做成了train/valid/test数据集，对于测试阶段，我们选择用户的前80%的行为作为序列输入，我们的标签是用户后20%的行为，这里涉及到使用Faiss进行向量召回：
```python
# 第一步：我们获取所有Item的Embedding表征，然后将其插入Faiss(向量数据库)中
item_embs = model.output_items().cpu().detach().numpy()
item_embs = normalize(item_embs, norm='l2')
gpu_index = faiss.IndexFlatIP(hidden_size)
gpu_index.add(item_embs)
# 第二步：根据用户的行为序列生产User的向量表征
user_embs = model(item_seq,mask,None,train=False)['user_emb']
user_embs = user_embs.cpu().detach().numpy()
# 第三步：对User的向量表征在所有Item的向量中进行Top-K检索
D, I = gpu_index.search(user_embs, topN)  # Inner Product近邻搜索，D为distance，I是index
```

完整版预测

```python
def get_predict(model, test_data, hidden_size, topN=20):

    item_embs = model.output_items().cpu().detach().numpy()
    item_embs = normalize(item_embs, norm='l2')
    gpu_index = faiss.IndexFlatIP(hidden_size)
    gpu_index.add(item_embs)
    
    test_gd = dict()
    preds = dict()
    
    user_id = 0

    for (item_seq, mask, targets) in tqdm(test_data):

        # 获取用户嵌入
        # 多兴趣模型，shape=(batch_size, num_interest, embedding_dim)
        # 其他模型，shape=(batch_size, embedding_dim)
        user_embs = model(item_seq,mask,None,train=False)['user_emb']
        user_embs = user_embs.cpu().detach().numpy()

        # 用内积来近邻搜索，实际是内积的值越大，向量越近（越相似）
        if len(user_embs.shape) == 2:  # 非多兴趣模型评估
            user_embs = normalize(user_embs, norm='l2').astype('float32')
            D, I = gpu_index.search(user_embs, topN)  # Inner Product近邻搜索，D为distance，I是index
#             D,I = faiss.knn(user_embs, item_embs, topN,metric=faiss.METRIC_INNER_PRODUCT)
            for i, iid_list in enumerate(targets):  # 每个用户的label列表，此处item_id为一个二维list，验证和测试是多label的
                test_gd[user_id] = iid_list
                preds[user_id] = I[i,:]
                user_id +=1
        else:  # 多兴趣模型评估
            ni = user_embs.shape[1]  # num_interest
            user_embs = np.reshape(user_embs,
                                   [-1, user_embs.shape[-1]])  # shape=(batch_size*num_interest, embedding_dim)
            user_embs = normalize(user_embs, norm='l2').astype('float32')
            D, I = gpu_index.search(user_embs, topN)  # Inner Product近邻搜索，D为distance，I是index
#             D,I = faiss.knn(user_embs, item_embs, topN,metric=faiss.METRIC_INNER_PRODUCT)
            for i, iid_list in enumerate(targets):  # 每个用户的label列表，此处item_id为一个二维list，验证和测试是多label的
                recall = 0
                dcg = 0.0
                item_list_set = []

                # 将num_interest个兴趣向量的所有topN近邻物品（num_interest*topN个物品）集合起来按照距离重新排序
                item_list = list(
                    zip(np.reshape(I[i * ni:(i + 1) * ni], -1), np.reshape(D[i * ni:(i + 1) * ni], -1)))
                item_list.sort(key=lambda x: x[1], reverse=True)  # 降序排序，内积越大，向量越近
                for j in range(len(item_list)):  # 按距离由近到远遍历推荐物品列表，最后选出最近的topN个物品作为最终的推荐物品
                    if item_list[j][0] not in item_list_set and item_list[j][0] != 0:
                        item_list_set.append(item_list[j][0])
                        if len(item_list_set) >= topN:
                            break
                test_gd[user_id] = iid_list
                preds[user_id] = item_list_set
                user_id +=1
    return test_gd, preds

def evaluate(preds,test_gd, topN=50):
    total_recall = 0.0
    total_ndcg = 0.0
    total_hitrate = 0
    for user in test_gd.keys():
        recall = 0
        dcg = 0.0
        item_list = test_gd[user]
        for no, item_id in enumerate(item_list):
            if item_id in preds[user][:topN]:
                recall += 1
                dcg += 1.0 / math.log(no+2, 2)
            idcg = 0.0
            for no in range(recall):
                idcg += 1.0 / math.log(no+2, 2)
        total_recall += recall * 1.0 / len(item_list)
        if recall > 0:
            total_ndcg += dcg / idcg
            total_hitrate += 1
    total = len(test_gd)
    recall = total_recall / total
    ndcg = total_ndcg / total
    hitrate = total_hitrate * 1.0 / total
    return {f'recall@{topN}': recall, f'ndcg@{topN}': ndcg, f'hitrate@{topN}': hitrate}

# 指标计算
def evaluate_model(model, test_loader, embedding_dim,topN=20):
    test_gd, preds = get_predict(model, test_loader, embedding_dim, topN=topN)
    return evaluate(preds, test_gd, topN=topN)
```
