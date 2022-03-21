### Attention

输入：询问(query)，键值对(key-value pairs)

计算注意力分数：

$a_i=\alpha(q,k_i)$

其中$\alpha$函数是用来计算query和key的相似性的。

归一化注意力分数，得到注意力权重：

$b_1, ..., b_n=softmax(a_1, ..., a_n)$

注意力权重与value计算加权求和，得到最终的输出：

$o=\sum_{i=1}^n b_iv_i$



### repeat函数

```python
valid_length=torch.FloatTensor([2,3])
valid_length.numpy().repeat(shape[1], axis=0)
# [2,2,3,3]
```

### **超出2维矩阵的乘法** 

```python
torch.bmm(torch.ones((2,1,3), dtype = torch.float), torch.ones((2,3,2), dtype = torch.float))

# 输出
tensor([[[3., 3.]],

        [[3., 3.]]])
```

 X 和 Y 是维度分别为(b,n,m) 和(b,m,k)的张量，进行 b次二维矩阵乘法后得到 Z, 维度为 (b,n,k)。



 

