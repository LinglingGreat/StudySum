理论部分：[协同过滤](../../协同过滤/协同过滤.md)

## 核心代码

### 构造样本

```python
df = pd.read_csv(config['valid_path'])

df['user_count'] = df['user_id'].map(df['user_id'].value_counts())

df = df[df['user_count']>20].reset_index(drop=True)

pos_dict = df.groupby('user_id')['item_id'].apply(list).to_dict()

# 负采样
ratio = 3
# 构造样本
train_user_list = []
train_item_list = []
train_label_list = []

test_user_list = []
test_item_list = []
test_label_list = []
if config['debug_mode']:
    user_list = df['user_id'].unique()[:100]
else:
    user_list = df['user_id'].unique()
    
item_list = df['item_id'].unique()
item_num = df['item_id'].nunique()

for user in tqdm(user_list):
    # 训练集正样本，行为序列的前n-1个
    for i in range(len(pos_dict[user])-1):
        train_user_list.append(user)
        train_item_list.append(pos_dict[user][i])
        train_label_list.append(1)
        
    # 测试集正样本，行为序列的最后一个
    test_user_list.append(user)
    test_item_list.append(pos_dict[user][-1])
    test_label_list.append(1)
    
    # 训练集：每个用户负样本数，是正样本数的ratio倍
    user_count = len(pos_dict[user])-1 # 训练集 用户行为序列长度
    neg_sample_per_user = user_count * ratio
    for i in range(neg_sample_per_user):
        train_user_list.append(user)
        temp_item_index = random.randint(0, item_num - 1)
        # 为了防止 负采样选出来的Item 在用户的正向历史行为序列(pos_dict)当中
        while item_list[temp_item_index] in pos_dict[user]:
            temp_item_index = random.randint(0, item_num - 1)
        train_item_list.append(item_list[temp_item_index])
        train_label_list.append(0)
    
    # 测试集合：每个用户负样本数为 100(论文设定)
    for i in range(100):
        test_user_list.append(user)
        temp_item_index = random.randint(0, item_num - 1)
        # 为了防止 负采样选出来的Item 在用户的正向历史行为序列(pos_dict)当中
        while item_list[temp_item_index] in pos_dict[user]:
            temp_item_index = random.randint(0, item_num - 1)
        test_item_list.append(item_list[temp_item_index])
        test_label_list.append(0)
        
train_df = pd.DataFrame()
train_df['user_id'] = train_user_list
train_df['item_id'] = train_item_list
train_df['label'] = train_label_list

test_df = pd.DataFrame()
test_df['user_id'] = test_user_list
test_df['item_id'] = test_item_list
test_df['label'] = test_label_list

```

### 模型搭建

```python
class NCF(paddle.nn.Layer):
    def __init__(self,
                embedding_dim = 16,
                vocab_map = None,
                loss_fun = 'nn.BCELoss()'):
        super(NCF, self).__init__()
        self.embedding_dim = embedding_dim
        self.vocab_map = vocab_map
        self.loss_fun = eval(loss_fun) # self.loss_fun  = paddle.nn.BCELoss()
        
        self.user_emb_layer = nn.Embedding(self.vocab_map['user_id'],
                                          self.embedding_dim)
        self.item_emb_layer = nn.Embedding(self.vocab_map['item_id'],
                                          self.embedding_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(2*self.embedding_dim,self.embedding_dim),
            nn.ReLU(),
            nn.BatchNorm1D(self.embedding_dim),
            nn.Linear(self.embedding_dim,1),
            nn.Sigmoid()
        )
        
    def forward(self,data):
        user_emb = self.user_emb_layer(data['user_id']) # [batch,emb]
        item_emb = self.item_emb_layer(data['item_id']) # [batch,emb]
        mlp_input = paddle.concat([user_emb, item_emb],axis=-1).squeeze(1)
        y_pred = self.mlp(mlp_input)
        if 'label' in data.keys():
            loss = self.loss_fun(y_pred.squeeze(),data['label'])
            output_dict = {'pred':y_pred,'loss':loss}
        else:
            output_dict = {'pred':y_pred}
        return output_dict
```

## 指标

[评估指标](../../评估指标/评估指标.md)


```python
def hitrate(test_df,k=20):
	user_num = test_df['user_id'].nunique()
	test_gd_df = test_df[test_df['ranking']<=k].reset_index(drop=True)
	return test_gd_df['label'].sum() / user_num
```


```python
def ndcg(test_df,k=20):
    '''
    idcg@k 一定为1
    dcg@k 1/log_2(ranking+1) -> log(2)/log(ranking+1)
    '''
    user_num = test_df['user_id'].nunique()
    test_gd_df = test_df[test_df['ranking']<=k].reset_index(drop=True)
    
    test_gd_df = test_gd_df[test_gd_df['label']==1].reset_index(drop=True)
    test_gd_df['ndcg'] = math.log(2) / np.log(test_gd_df['ranking']+1)
    return test_gd_df['ndcg'].sum() / user_num
```



