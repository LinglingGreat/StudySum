[社区供稿 | Mixtral-8x7B Pytorch 实现](https://mp.weixin.qq.com/s/HProBDSA9WxyD-JuKpJ9ew)

[全网最细致大模型MoE原理+代码手撕版](https://mp.weixin.qq.com/s/76a-7fDJumv6iB08L2BUKg)


MoE 架构通过用混合专家替换部分或全部 FFN 来修改典型的transformer，其中每个专家本身就是一个小型 FFN，并且 MoE 层容纳多个此类专家。

MoE 层通过为每个输入令牌有选择地激活一些专家网络来增加transformer模型的容量，同时保持计算效率。专家的选择是通过门控机制执行的，允许模型动态地将代币路由给最相关的专家。

门控机制由一个 softmax 层组成，该层计算每个令牌的可用专家的概率分布。嵌入 xi 的第 i 个令牌的门输出 g 由下式给出

![](img/Pasted%20image%2020240629114906.png)

![](img/Pasted%20image%2020240629115016.png)


