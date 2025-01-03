
**外推性** 的含义是在长文本表征过程中，如何在训练阶段只需要学习有限的长度，即可以在推理阶段能够延伸长度至若干倍且依然保持不错的性能和效果。

长度外推性是一个训练和预测的长度不一致的问题，主要体现在两个方面：

- 预测的时候用到了没训练过的位置编码（不论是绝对位置还是相对位置）；
    
- 预测的时候注意力机制所处理的token数量远超训练时的数量。
    

解决长文本外推性问题的一个简单有效的方法是Attention Mask，如图所示：

![](img/Pasted%20image%2020231126165946.png)

- 通过类似滑动窗口的结构，约束一个每个token只能对局部区域的token计算Attention值，因此对于相对位置大小不会超过窗口大小，解决了第一个问题；
    
- Attention只会在窗口内计算，避免了对大量的token的Attention进行加权平均导致最终权重过度“平滑”现象。

在实现过程中，本质上是在计算完$q_m^Tk_n$之后减去一个矩阵，即$q_m^Tk_n-M$，其中M的形状如下图所示：

![](img/Pasted%20image%2020231126170051.png)

可以看出，蓝色区域（即滑动窗口内的局部区域）为0，说明保持原始的Attention归一化前的值；其他区域则为一个INT内最大的整数，说明Attention值是一个非常小的数（在softmax归一化后几乎为0）。

#### ALIBI

论文：《Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation》

其与上面的思想一样，只是改进了上面的 M矩阵为$\lambda|m-n|$，即Attention在归一化前的计算为：$q_m^Tk_n-\lambda|m-n|$，其中$\lambda$为超参数，Transformer的多头注意力中的每个头的值可设置不同。矩阵$\lambda|m-n|$的形状如下所示：

![](img/Pasted%20image%2020231126170350.png)

相比于原始的方法，相对距离越长，$\lambda|m-n|$值就越大，越远的位置Attention值被归一化后就约小，相对于“滑动窗口”采用的方法是hard（在窗口内就计算attention，不在窗口内不计算），AIBLI是比较soft的（离得近attention就会比较大，离得远就比较小）。

#### KERPLE

论文：《KERPLE: Kernelized Relative Positional Embedding for Length Extrapolation》

![](img/Pasted%20image%2020231126170556.png)

#### Sandwich

论文：《Receptive Field Alignment Enables Transformer Length Extrapolation》

![](img/Pasted%20image%2020231126170626.png)

#### XPOS

论文：《A Length-Extrapolatable Transformer》  
参考解读：Transformer升级之路：7、长度外推性与局部注意力

![](img/Pasted%20image%2020231126170714.png)

## 外推性的其他探索

#### （1）混合注意力Mask

在解决长文本位置表征时，典型的代表有Transformer-XL、BigBird、LongFormer，他们除了局部注意力机制以外，还引入了随机位置的性质：

![](img/Pasted%20image%2020231126170753.png)

如上图，第2张图为局部注意力（滑动窗口），第3章图为有限的全局感知（例如只限制前两个token可以看见所有的token）。而第一张图则是随机mask，以缓解过度hard的局部注意力。三者注意力混合起来后得到第四张图，这也是普遍训练超长文本大模型时采用的方法。

#### （2）随机位置表征

> 论文：《Randomized Positional Encodings Boost Length Generalization of Transformers》

![](img/Pasted%20image%2020231126170811.png)


绝对位置表征时，会存在位置上的OOV问题，随机位置编码则是通过在训练过程中采用如下策略：

![](img/Pasted%20image%2020231126170830.png)

对应的代码也很简单：

```python
def random_position_ids(N, L=2048):       
	"""从[0, L)中随机不重复挑N个整数，并从小到大排列       
	"""       
	return np.sort(np.random.permutation(L)[:N])
```

苏神对随机位置编码的新探索：

![](img/Pasted%20image%2020231126171028.png)

对应的代码为：

```python
def random_position_ids(N):  
    """先随机采样n，然后从[0, n]均匀取N个点  
    """  
    n = sample_from_xxx()  
    return np.linspace(0, 1, N) * n
```

#### （3） logn Attention Scale

![](img/Pasted%20image%2020231126171113.png)

#### （4）全局依赖

滑动窗口的方法如果在一层Transformer Layer里看，本质上类似长度为 w的N-Gram模型，即如下图所示：
![](img/Pasted%20image%2020231126171204.png)

如果Transformer又L层，那么，从输入层开始，长度为w的窗口内的信息，可以在经过L层之后传给一个更广的区域，区域长度为(w-1)L + 1 ，如下图所示：

![](img/Pasted%20image%2020231126171302.png)

苏神给出的一种新的想法，就是假设我有L层Transformer，则可以在前L-1层利用这种扩张特性，得到最终(w-1)(L-1)+1长度的区域后，在最后一层采用上面提到的 logn Attention Scale方法，讲前L-1层扩张的信息快速在最后一层与所有token进行交互。引入苏神的原文为：

![](img/Pasted%20image%2020231126171401.png)

这种局部注意力+Attention Scale的结合也是一种很巧妙的idea。实验也发现这种策略的外推性很惊艳。

## 参考资料

[# 聊聊大模型位置编码及其外推性](https://mp.weixin.qq.com/s/KHvQsUB3YmWNVosIxjYtig)

[匿名论文提出奇招！增强大模型长文本能力居然还能这么做Temp-Lora](https://mp.weixin.qq.com/s/V9C0s4HR2cQinz1Bgrjsiw)

[一览大模型长文本能力](https://mp.weixin.qq.com/s/grvA5xUfURLROsDK1vK4jw)

[RoPE外推的缩放法则 —— 尝试外推RoPE至1M上下文 - 知乎](https://zhuanlan.zhihu.com/p/660073229)

[大模型处理长上下文方法一览](https://mp.weixin.qq.com/s/81NHGf5W8HEscW2dBK8MRg)

[2024.5横向对比各家LLM的Long Context（128k篇）](https://mp.weixin.qq.com/s/L8Iiv9vbDlKAFMvYMF-jQw)

[解锁大模型长上下文能力](https://mp.weixin.qq.com/s/FTewPxSr5fcwkxAgRZm7Wg)



