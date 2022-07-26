## Pandas库
Numpy  
基础数据类型，关注数据的结构表达，维度：数据间关系  
Pandas  
扩展数据类型，关注数据的应用表达，维度：数据与索引间关系

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

###遍历

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

###类型转换

**dataframe转成dict**

`DataFrame.``to_dict`(*orient='dict'*, *into=<class 'dict'>*)[[source\]](http://github.com/pandas-dev/pandas/blob/v0.23.4/pandas/core/frame.py#L987-L1102) 

http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.to_dict.html

