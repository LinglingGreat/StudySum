Numpy  基础数据类型，关注数据的结构表达，维度：数据间关系  
Pandas  扩展数据类型，关注数据的应用表达，维度：数据与索引间关系

# 基础

## 匿名函数与map方法

```python
my_func = lambda x: 2*x
my_func(3)

[(lambda x: 2*x)(i) for i in range(5)]

list(map(lambda x: 2*x, range(5)))
list(map(lambda x, y: str(x)+'_'+y, range(5), list('abcde')))

```

## 字典构建

根据列表构建

```python
# L1是key，L2是val
L1, L2, L3 = list('abc'), list('def'), list('hij')
dict(zip(L1, L2))

```

## zip

```python
zipped = list(zip(L1, L2, L3))
[('a', 'd', 'h'), ('b', 'e', 'i'), ('c', 'f', 'j')]

list(zip(*zipped)) # 三个元组分别对应原来的列表
[('a', 'b', 'c'), ('d', 'e', 'f'), ('h', 'i', 'j')]

```

# 构造df

## Series

`Series` 一般由四个部分组成，分别是序列的值 `data` 、索引 `index` 、存储类型 `dtype` 、序列的名字 `name` 。其中，索引也可以指定它的名字，默认为空。

```python
In [22]: s = pd.Series(data = [100, 'a', {'dic1':5}],
   ....:               index = pd.Index(['id1', 20, 'third'], name='my_idx'),
   ....:               dtype = 'object',
   ....:               name = 'my_name')
   ....: 

In [23]: s
Out[23]: 
my_idx
id1              100
20                 a
third    {'dic1': 5}
Name: my_name, dtype: object
```

`object` 代表了一种混合类型，正如上面的例子中存储了整数、字符串以及 `Python` 的字典数据结构。此外，目前 `pandas` 把纯字符串序列也默认认为是一种 `object` 类型的序列，但它也可以用 `string` 类型存储

对于这些属性，可以通过 `.` 的方式来获取：

```python
In [24]: s.values
Out[24]: array([100, 'a', {'dic1': 5}], dtype=object)

In [25]: s.index
Out[25]: Index(['id1', 20, 'third'], dtype='object', name='my_idx')

In [26]: s.dtype
Out[26]: dtype('O')

In [27]: s.name
Out[27]: 'my_name'

In [29]: s['third']
Out[29]: {'dic1': 5}

```

## DataFrame

`DataFrame` 在 `Series` 的基础上增加了列索引，一个数据框可以由二维的 `data` 与行列索引来构造：

```python
In [30]: data = [[1, 'a', 1.2], [2, 'b', 2.2], [3, 'c', 3.2]]

In [31]: df = pd.DataFrame(data = data,
   ....:                   index = ['row_%d'%i for i in range(3)],
   ....:                   columns=['col_0', 'col_1', 'col_2'])
   ....: 

In [32]: df
Out[32]: 
       col_0 col_1  col_2
row_0      1     a    1.2
row_1      2     b    2.2
row_2      3     c    3.2
```

但一般而言，更多的时候会采用从列索引名到数据的映射来构造数据框，同时再加上行索引：

```python
In [33]: df = pd.DataFrame(data = {'col_0': [1,2,3], 'col_1':list('abc'),
   ....:                           'col_2': [1.2, 2.2, 3.2]},
   ....:                   index = ['row_%d'%i for i in range(3)])
   ....: 

In [34]: df
Out[34]: 
       col_0 col_1  col_2
row_0      1     a    1.2
row_1      2     b    2.2
row_2      3     c    3.2
```

由于这种映射关系，在 `DataFrame` 中可以用 `[col_name]` 与 `[col_list]` 来取出相应的列与由多个列组成的表，结果分别为 `Series` 和 `DataFrame`

与 `Series` 类似，在数据框中同样可以取出相应的属性：

```python
In [37]: df.values
Out[37]: 
array([[1, 'a', 1.2],
       [2, 'b', 2.2],
       [3, 'c', 3.2]], dtype=object)

In [38]: df.index
Out[38]: Index(['row_0', 'row_1', 'row_2'], dtype='object')

In [39]: df.columns
Out[39]: Index(['col_0', 'col_1', 'col_2'], dtype='object')

In [40]: df.dtypes # 返回的是值为相应列数据类型的Series
Out[40]: 
col_0      int64
col_1     object
col_2    float64
dtype: object

In [41]: df.shape
Out[41]: (3, 3)
```

转置

```python
In [42]: df.T
Out[42]: 
      row_0 row_1 row_2
col_0     1     2     3
col_1     a     b     c
col_2   1.2   2.2   3.2

```

# 常用基本函数

## 特征统计函数

在 `Series` 和 `DataFrame` 上定义了许多统计函数，最常见的是 `sum, mean, median, var, std, max, min` 。

`df_demo.mean()`

此外，需要介绍的是 `quantile, count, idxmax`,idxmin 这三个函数，它们分别返回的是分位数、非缺失值个数、最大值对应的索引

`df_demo.quantile(0.75)`

上面这些所有的函数，由于操作后返回的是标量，所以又称为聚合函数，它们有一个公共参数 `axis` ，默认为0代表逐列聚合，如果设置为1则表示逐行聚合。

## 唯一值函数

对序列使用 `unique` 和 `nunique` 可以分别得到其唯一值组成的列表和唯一值的个数

`value_counts` 可以得到唯一值和其对应出现的频数

如果想要观察多个列组合的唯一值，可以使用 `drop_duplicates` 。其中的关键参数是 `keep` ，默认值 `first` 表示每个组合保留第一次出现的所在行， `last` 表示保留最后一次出现的所在行， `False` 表示把所有重复组合所在的行剔除。

```python
In [60]: df_demo = df[['Gender','Transfer','Name']]

In [61]: df_demo.drop_duplicates(['Gender', 'Transfer'])
Out[61]: 
    Gender Transfer            Name
0   Female        N    Gaopeng Yang
1     Male        N  Changqiang You
12  Female      NaN        Peng You
21    Male      NaN   Xiaopeng Shen
36    Male        Y    Xiaojuan Qin
43  Female        Y      Gaoli Feng

In [62]: df_demo.drop_duplicates(['Gender', 'Transfer'], keep='last')
Out[62]: 
     Gender Transfer            Name
147    Male      NaN        Juan You
150    Male        Y   Chengpeng You
169  Female        Y   Chengquan Qin
194  Female      NaN     Yanmei Qian
197  Female        N  Chengqiang Chu
199    Male        N     Chunpeng Lv

In [63]: df_demo.drop_duplicates(['Name', 'Gender'],
   ....:                      keep=False).head() # 保留只出现过一次的性别和姓名组合
   ....: 
Out[63]: 
   Gender Transfer            Name
0  Female        N    Gaopeng Yang
1    Male        N  Changqiang You
2    Male        N         Mei Sun
4    Male        N     Gaojuan You
5  Female        N     Xiaoli Qian

In [64]: df['School'].drop_duplicates() # 在Series上也可以使用
Out[64]: 
0    Shanghai Jiao Tong University
1                Peking University
3                 Fudan University
5              Tsinghua University
Name: School, dtype: object
```

此外， `duplicated` 和 `drop_duplicates` 的功能类似，但前者返回了是否为唯一值的布尔列表，其 `keep` 参数与后者一致。其返回的序列，把重复元素设为 `True` ，否则为 `False` 。 `drop_duplicates` 等价于把 `duplicated` 为 `True` 的对应行剔除。

```python
In [65]: df_demo.duplicated(['Gender', 'Transfer']).head()
Out[65]: 
0    False
1    False
2     True
3     True
4     True
dtype: bool

In [66]: df['School'].duplicated().head() # 在Series上也可以使用
Out[66]: 
0    False
1    False
2     True
3    False
4     True
Name: School, dtype: bool

```

## 替换函数

在 `replace` 中，可以通过字典构造，或者传入两个列表来进行替换：

```python
In [67]: df['Gender'].replace({'Female':0, 'Male':1}).head()
Out[67]: 
0    0
1    1
2    1
3    0
4    1
Name: Gender, dtype: int64

In [68]: df['Gender'].replace(['Female', 'Male'], [0, 1]).head()
Out[68]: 
0    0
1    1
2    1
3    0
4    1
Name: Gender, dtype: int64
```

另外， `replace` 还有一种特殊的方向替换，指定 `method` 参数为 `ffill` 则为用前面一个最近的未被替换的值进行替换， `bfill` 则使用后面最近的未被替换的值进行替换。

逻辑替换包括了 `where` 和 `mask` ，这两个函数是完全对称的： `where` 函数在传入条件为 `False` 的对应行进行替换，而 `mask` 在传入条件为 `True` 的对应行进行替换，当不指定替换值时，替换为缺失值。

```python
In [72]: s = pd.Series([-1, 1.2345, 100, -50])

In [73]: s.where(s<0)
Out[73]: 
0    -1.0
1     NaN
2     NaN
3   -50.0
dtype: float64

In [74]: s.where(s<0, 100)
Out[74]: 
0     -1.0
1    100.0
2    100.0
3    -50.0
dtype: float64

In [75]: s.mask(s<0)
Out[75]: 
0         NaN
1      1.2345
2    100.0000
3         NaN
dtype: float64

In [76]: s.mask(s<0, -50)
Out[76]: 
0    -50.0000
1      1.2345
2    100.0000
3    -50.0000
dtype: float64
```

需要注意的是，传入的条件只需是与被调用的 `Series` 索引一致的布尔序列即可：

```python
In [77]: s_condition= pd.Series([True,False,False,True],index=s.index)

In [78]: s.mask(s_condition, -50)
Out[78]: 
0    -50.0000
1      1.2345
2    100.0000
3    -50.0000
dtype: float64
```

数值替换包含了 `round, abs, clip` 方法，它们分别表示按照给定精度四舍五入、取绝对值和截断：

```python
In [79]: s = pd.Series([-1, 1.2345, 100, -50])

In [80]: s.round(2)
Out[80]: 
0     -1.00
1      1.23
2    100.00
3    -50.00
dtype: float64

In [81]: s.abs()
Out[81]: 
0      1.0000
1      1.2345
2    100.0000
3     50.0000
dtype: float64

In [82]: s.clip(0, 2) # 前两个数分别表示上下截断边界
Out[82]: 
0    0.0000
1    1.2345
2    2.0000
3    0.0000
dtype: float64

```

## 排序函数

排序共有两种方式，其一为值排序，其二为索引排序，对应的函数是 `sort_values` 和 `sort_index` 。

`df_demo.sort_values(['Weight','Height'],ascending=[True,False]).head()`

索引排序的用法和值排序完全一致，只不过元素的值在索引中，此时需要指定索引层的名字或者层号，用参数 `level` 表示。另外，需要注意的是字符串的排列顺序由字母顺序决定。

```df_demo.sort_index(level=['Grade','Name'],ascending=[True,False]).head()`

## apply方法

> 得益于传入自定义函数的处理， `apply` 的自由度很高，但这是以性能为代价的。一般而言，使用 `pandas` 的内置函数处理和 `apply` 来处理同一个任务，其速度会相差较多，因此只有在确实存在自定义需求的情境下才考虑使用 `apply` 。

# 窗口对象

`pandas` 中有3类窗口，分别是滑动窗口 `rolling` 、扩张窗口 `expanding` 以及指数加权窗口 `ewm` 。

要使用滑窗函数，就必须先要对一个序列使用 `.rolling` 得到滑窗对象，其最重要的参数为窗口大小 `window` 。

```python
In [95]: s = pd.Series([1,2,3,4,5])

In [96]: roller = s.rolling(window = 3)

In [97]: roller
Out[97]: Rolling [window=3,center=False,axis=0]
```

在得到了滑窗对象后，能够使用相应的聚合函数进行计算，需要注意的是窗口包含当前行所在的元素，例如在第四个位置进行均值运算时，应当计算(2+3+4)/3，而不是(1+2+3)/3：

```python
In [98]: roller.mean()
Out[98]: 
0    NaN
1    NaN
2    2.0
3    3.0
4    4.0
dtype: float64

In [99]: roller.sum()
Out[99]: 
0     NaN
1     NaN
2     6.0
3     9.0
4    12.0
dtype: float64
```

对于滑动相关系数或滑动协方差的计算，可以如下写出：

```python
In [100]: s2 = pd.Series([1,2,6,16,30])

In [101]: roller.cov(s2)
Out[101]: 
0     NaN
1     NaN
2     2.5
3     7.0
4    12.0
dtype: float64

In [102]: roller.corr(s2)
Out[102]: 
0         NaN
1         NaN
2    0.944911
3    0.970725
4    0.995402
dtype: float64
```

此外，还支持使用 `apply` 传入自定义函数，其传入值是对应窗口的 `Series` ，例如上述的均值函数可以等效表示：

```python
In [103]: roller.apply(lambda x:x.mean())
Out[103]: 
0    NaN
1    NaN
2    2.0
3    3.0
4    4.0
dtype: float64
```

`shift, diff, pct_change` 是一组类滑窗函数，它们的公共参数为 `periods=n` ，默认为1，分别表示取向前第 `n` 个元素的值、与向前第 `n` 个元素做差（与 `Numpy` 中不同，后者表示 `n` 阶差分）、与向前第 `n` 个元素相比计算增长率。这里的 `n` 可以为负，表示反方向的类似操作。

```python
In [104]: s = pd.Series([1,3,6,10,15])

In [105]: s.shift(2)
Out[105]: 
0    NaN
1    NaN
2    1.0
3    3.0
4    6.0
dtype: float64

In [106]: s.diff(3)
Out[106]: 
0     NaN
1     NaN
2     NaN
3     9.0
4    12.0
dtype: float64

In [107]: s.pct_change()
Out[107]: 
0         NaN
1    2.000000
2    1.000000
3    0.666667
4    0.500000
dtype: float64

In [108]: s.shift(-1)
Out[108]: 
0     3.0
1     6.0
2    10.0
3    15.0
4     NaN
dtype: float64

In [109]: s.diff(-2)
Out[109]: 
0   -5.0
1   -7.0
2   -9.0
3    NaN
4    NaN
dtype: float64
```

将其视作类滑窗函数的原因是，它们的功能可以用窗口大小为 `n+1` 的 `rolling` 方法等价代替：

```python
In [110]: s.rolling(3).apply(lambda x:list(x)[0]) # s.shift(2)
Out[110]: 
0    NaN
1    NaN
2    1.0
3    3.0
4    6.0
dtype: float64

In [111]: s.rolling(4).apply(lambda x:list(x)[-1]-list(x)[0]) # s.diff(3)
Out[111]: 
0     NaN
1     NaN
2     NaN
3     9.0
4    12.0
dtype: float64

In [112]: def my_pct(x):
   .....:     L = list(x)
   .....:     return L[-1]/L[0]-1
   .....: 

In [113]: s.rolling(2).apply(my_pct) # s.pct_change()
Out[113]: 
0         NaN
1    2.000000
2    1.000000
3    0.666667
4    0.500000
dtype: float64
```

扩张窗口又称累计窗口，可以理解为一个动态长度的窗口，其窗口的大小就是从序列开始处到具体操作的对应位置，其使用的聚合函数会作用于这些逐步扩张的窗口上。具体地说，设序列为a1, a2, a3, a4，则其每个位置对应的窗口即[a1]、[a1, a2]、[a1, a2, a3]、[a1, a2, a3, a4]。

```python
In [114]: s = pd.Series([1, 3, 6, 10])

In [115]: s.expanding().mean()
Out[115]: 
0    1.000000
1    2.000000
2    3.333333
3    5.000000
dtype: float64
```

# 常见应用
## groupby之后写入文件

```Python
def my_func(y):

  return ','.join([str(x) for x in y])

data = pd.read_excel("./result/旅拍场景识别_level3及以上视频.xlsx", sheet_name="Sheet1")

print(data.shape)

data = data.groupby(['article_id', 'level', 'url', 'videourls']).agg({'centraltagid': list, 'tagname': my_func}).reset_index()

print(data.shape)

data.to_csv('test.csv', index=False, encoding='utf_8_sig')
```

## 如何把DataFrame中以list类型存放的单元进行纵向展开？一行变多行

`df.explode('value', ignore_index=True)`

[https://www.cnblogs.com/traditional/p/11967360.html](https://www.cnblogs.com/traditional/p/11967360.html)

## 嵌套的字典转DataFrame

[https://blog.csdn.net/sinat_26811377/article/details/100065580](https://blog.csdn.net/sinat_26811377/article/details/100065580)

## 新的列是groupby后再sum

df['Total Amount'] = df.groupby('Id', sort=False)["Amount"].transform('sum')

## df append

新建一个df，然后df2 = df2.append(newdf2, ignore_index=True)

## 两列转字典

item.set_index('item_id')['item_category'].to_dict()

[('a', 'd', 'h'), ('b', 'e', 'i'), ('c', 'f', 'j')]

[('a', 'b', 'c'), ('d', 'e', 'f'), ('h', 'i', 'j')]

索引排序的用法和值排序完全一致，只不过元素的值在索引中，此时需要指定索引层的名字或者层号，用参数 level 表示。另外，需要注意的是字符串的排列顺序由字母顺序决定。

In [103]: roller.apply(lambda x:x.mean()) Out[103]: 0 NaN 1 NaN 2 2.0 3 3.0 4 4.0 dtype: float64

# 历史笔记

### Series类型  
Series类型由一组数据及与之相关的数据索引组成  
**创建Series**：  

- Python列表,index与列表元素个数一致  
  b = pd.Series([9,8,7,6],index=['a','b','c','d'])    

- 标量值,index表达Series类型的尺寸  
  s = pd.Series(25, index=['a','b','c'])  

- Python字典，键值对中的"键"是索引，index从字典中进行选择操作    
  e = pd.Series({'a':9,'b':8,'c':7},index=['c','a','b','d'])  

- ndarray,索引和数据都可以通过ndarray类型创建    
  m = pd.Series(np.arange(5), index=np.arange(9,4,-1))  

- 其他函数，range()函数等  

Series类型包括index和values两部分 
   b[['c','d','a']]

Series类型的操作类似ndarray类型 

- 索引方法相同，采用[]  
- NumPy中运算和操作可用于Series类型  
- 可以通过自定义索引的列表进行切片  
- 可以通过自动索引进行切片，如果存在自定义索引，则一同被切片   

Series类型的操作类似Python字典类型  

- 通过自定义索引访问  
- 保留字in操作  
- 使用.get()方法  
  b.get('f',100)

Series+Series  
Series类型在运算中会自动对齐不同索引的数据  

Series对象和索引都可以有一个名字，存储在属性.name中  

### DataFrame类型  
DataFrame类型由共用相同索引的一组列组成  

**创建DataFrame**：  

- 二维ndarray对象   
  d = pd.DataFrame(np.arange(10).reshape(2,5))

- 由一维ndarray、列表、字典、元组或Series构成的字典  

```python
dt = {'one':pd.Series([1,2,3],index=['a','b','c']),
      'two':pd.Series([9,8,7,6],index=['a','b','c','d'])}  
d = pd.DataFrame(dt)  
pd.DataFrame(dt,index=['b','c','d'],columns=['two','three'])

d1 = {'one':[1,2,3,4],'two':[9,8,7,6]}
d = pd.DataFrame(d1,index=['a','b','c','d'])
```
- Series类型  

- 其他的DataFrame类型  

### 索引操作  
**修改索引**

```
s4 = pd.Series(np.array([1,1,2,3,5,8]))
s4.index = ['a','b','c','d','e','f']
```

**数据的获取**

```
s4[3]
s4['e']
s4[[1,3,5]]
s4[['a','b','d','f']]
s4[:4]
s4['c':]
s4['b':'e']
```

如果通过索引标签获取数据的话，末端标签所对应的值是可以返回的

.sort_index()方法在指定轴上根据索引进行排序，默认升序  
.sort_index(axis=0, ascending=True)  

增加或重排：重新索引  
.reindex()能够改变或重排Series和DataFrame索引   
.reindex(index=None, columns=None, …)的参数  
index, columns 新的行列自定义索引  
fill_value 重新索引中，用于填充缺失位置的值  
method 填充方法, ffill当前值向前填充，bfill向后填充  
limit 最大填充量  
copy 默认True，生成新的对象，False时，新旧相等不复制  
newc = d.columns.insert(4,'新增')  
newd = d.reindex(columns=newc,fill_value=200)

Series和DataFrame的索引是Index类型  
Index对象是不可修改类型  

索引类型的常用方法  
.append(idx) 连接另一个Index对象，产生新的Index对象  
.diff(idx) 计算差集，产生新的Index对象  
.intersection(idx) 计算交集  
.union(idx) 计算并集  
.delete(loc) 删除loc位置处的元素  
.insert(loc,e) 在loc位置增加一个元素e  

删除：drop  
.drop()能够删除Series和DataFrame指定行或列索引  

###查询数据

**查询指定的行**

```
student.iloc[[0,2,4,5,7]] #这里的iloc索引标签函数必须是中括号[]
```

**查询指定的列**

```
student[['Name','Height','Weight']].head() #如果多个列的话，必须使用双重中括号
```

**也可以通过ix索引标签查询指定的列**

```
student.ix[:,['Name','Height','Weight']].head()
```

**查询指定的行和列**

```
student.ix[[0,2,4,5,7],['Name','Height','Weight']].head()
```

**条件查询**

```
student[student['Sex']=='F']
student[(student['Sex']=='F') & (student['Age']>12)]
student[(student['Sex']=='F') & (student['Age']>12)][['Name','Height','Weight']]
```



### 统计分析

```python
np.random.seed(1234)
d1 = pd.Series(2*np.random.normal(size = 100)+3)
d2 = np.random.f(2,4,size = 100)
d3 = np.random.randint(1,100,size = 100)
d1.count() #非空元素计算
d1.min() #最小值
d1.max() #最大值
d1.idxmin() #最小值的位置，类似于R中的which.min函数
d1.idxmax() #最大值的位置，类似于R中的which.max函数
d1.quantile(0.1) #10%分位数
d1.sum() #求和
d1.mean() #均值
d1.median() #中位数
d1.mode() #众数
d1.var() #方差
d1.std() #标准差
d1.mad() #平均绝对偏差
d1.skew() #偏度
d1.kurt() #峰度
d1.describe() #一次性输出多个描述性统计指标
适用于Series类型  
.argmin() .argmax() 计算数据最大值、最小值所在位置的索引位置（自动索引） 
.idxmin() .idxmax() 计算数据最大值、最小值所在位置的索引（自定义索引）
```

```python
# 应用到每一列，用apply
df = pd.DataFrame(np.array([d1,d2,d3]).T,columns=['x1','x2','x3'])
df.head()
df.apply(stats)
```

```python
df.corr()   # 相关系数
df.corr('spearman')  # 可以调用pearson方法或kendell方法或spearman方法，默认使用pearson方法
df.corrwith(df['x1'])  # 只关心x1与其余变量的相关系数
df.cov()  # 数值型变量间的协方差矩阵
```

累计统计分析函数  
适用于Series和DataFrame类型，累计计算  
.cumsum() 依次给出前1、2、…、n个数的和  
.cumprod() 依次给出前1、2、…、n个数的积  
.cummax() 依次给出前1、2、…、n个数的最大值  
.cummin() 依次给出前1、2、…、n个数的最小值  

适用于Series和DataFrame类型，滚动计算（窗口计算）  
.rolling(w).sum() 依次计算相邻w个元素的和  
.rolling(w).mean() 依次计算相邻w个元素的算术平均值  
.rolling(w).var() 依次计算相邻w个元素的方差  
.rolling(w).std() 依次计算相邻w个元素的标准差  
.rolling(w).min() .max() 依次计算相邻w个元素的最小值和最大值  

### 类似于SQL的操作

**表的Union，会自动对齐列，在原始数据student下新增student2的数据行**

```
student3 = pd.concat([student, student2])
```

**新增一列，这里Score是新增的列，没有赋值，是NaN**

```
pd.DataFrame(student2, columns=['Age','Height','Name','Sex','Weight','Score'])
```

**删除表、观测行或变量行**

```
del student2  # del可以删除Python的所有对象
student.drop(['a']) # 删除行索引为a的数据行
student.drop(['Height','Weight'],axis=1)  # 删除指定的列
```

**修改原始记录的值**

```
# 修改姓名为Liushunxiang的学生的身高
student3.ix[student3['Name']=='LiuShunxiang','Height']=170
```

**数据聚合**

```python
# 根据性别分组，计算各组其它属性的平均值
student.groupby('Sex').mean()
# 如果不想对年龄计算平均值的话，需要剔除变量
student.drop('Age',axis=1).groupby('Sex').mean()
# 多个分组字段
student.groupby(['Age','Sex']).mean()
# 计算多个统计量
student.drop('Age',axis=1).groupby('Sex').agg([np.mean, np.median])
```

**排序**

```
student.sort_values(by=['Sex','Age'])
```

**多表连接**

```
stu_score1=pd.merge(student,score,on='Name',how='left')
```

左连接实现的是保留student表中的所有信息，同时将score表的信息与之配对，能配多少配多少，对于没有配上的Name，将会显示成绩为NaN

**只想要b列值为5和13的行**

```
df[df.b.isin([5,13])]
```

**如果想要除了这两行以外的数据呢？**

原理是先把b取出来准换为列表，然后再从列表中把不需要的行（值）去除，然后再在df中使用isin()

```
test = list(df.b)
test.remove(5)
test.remove(13)
df[df.b.isin(test)]
```



### 缺失值处理

常用的有三大类方法，即删除法、填补法和插值法。
删除法：当数据中的某个变量大部分值都是缺失值，可以考虑删除改变量；当缺失值是随机分布的，且缺失的数量并不是很多是，也可以删除这些缺失的观测。
替补法：对于连续型变量，如果变量的分布近似或就是正态分布的话，可以用均值替代那些缺失值；如果变量是有偏的，可以使用中位数来代替那些缺失值；对于离散型变量，我们一般用众数去替换那些存在缺失的观测。
插补法：插补法是基于蒙特卡洛模拟法，结合线性模型、广义线性模型、决策树等方法计算出来的预测值替换缺失值。

```python
# 检测有多少缺失值
sum(pd.isnull(s))
# 直接删除
s.dropna()
s.dropna(how='all')  # 或者any
# 填补缺失值
df.fillna(0)
df.fillna(method='ffill')  # ffill前一个观测值填充,bfill后一个
df.fillna({'x1':1,'x2':2,'x3':3})
df.fillna({'x1':df['x1'].median(),'x2':df['x2'].median(),'x3':df['x3'].median()})
```

### 数据透视表

pivot_table()

### 数据类型运算  

二维和一维、一维和零维间为广播运算,一维Series默认在轴1参与运算，用axis=0可以令一维Series参与轴0运算    
采用+ ‐ * /符号进行的二元运算产生新的对象  

方法形式的运算  
.add(d, \*\*argws) 类型间加法运算，可选参数  
.sub(d, \*\*argws) 类型间减法运算，可选参数  
.mul(d, \**argws) 类型间乘法运算，可选参数  
.div(d, **argws) 类型间除法运算，可选参数  

a = pd.DataFrame(np.arange(12).reshape(3,4))  
b = pd.DataFrame(np.arange(20).reshape(4,5))  
b.add(a,fill_value = 100)  
a.mul(b,fill_value = 0)  

比较运算法则  
比较运算只能比较相同索引的元素，不进行补齐  
采用> < >= <= == !=等符号进行的二元运算产生布尔对象  

### 遍历

**遍历dataframe**

```python
import pandas as pd

dict=[[1,2,3,4,5,6],[2,3,4,5,6,7],[3,4,5,6,7,8],[4,5,6,7,8,9],[5,6,7,8,9,10]]
data=pd.DataFrame(dict)
print(data)
for indexs in data.index:
    print(data.loc[indexs].values[0:-1])
    
# 输出
   0  1  2  3  4   5
0  1  2  3  4  5   6
1  2  3  4  5  6   7
2  3  4  5  6  7   8
3  4  5  6  7  8   9
4  5  6  7  8  9  10
[1 2 3 4 5]
[2 3 4 5 6]
[3 4 5 6 7]
[4 5 6 7 8]
[5 6 7 8 9]
```

```python
for index, row in df.iterrows():
    print row["c1"], row["c2"]
```

```python
for row in df.itertuples(index=True, name='Pandas'):
    print getattr(row, "c1"), getattr(row, "c2")
```

```python
for i in range(0, len(df)):
    print df.iloc[i]['c1'], df.iloc[i]['c2']
```

### 类型转换

**dataframe转成dict**

`DataFrame.``to_dict`(*orient='dict'*, *into=<class 'dict'>*)[[source\]](http://github.com/pandas-dev/pandas/blob/v0.23.4/pandas/core/frame.py#L987-L1102) 

http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.to_dict.html

