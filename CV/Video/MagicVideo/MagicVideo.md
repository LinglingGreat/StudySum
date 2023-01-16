---
title: MagicVideo
created: 2023-01-15
tags: 文生视频 ByteDance
type: 论文
papername: MagicVideo Efficient Video Generation With Latent Diffusion Models
---

## 论文基本信息

标题：MagicVideo: Efficient Video Generation With Latent Diffusion Models

链接： http://arxiv.org/abs/2211.11018

主页： https://magicvideo.github.io/#

框架图：

![](img/Pasted%20image%2020230115225418.png)

三个部分：keyframe generation, frame interpolation, and superresolution.

### 定义

时间步t的视频帧的序列$x_t=[x_t^1, ..., x_t^F]$

VAE的encoder和decoder分别是$\xi(·)$, $D(·)$

视频帧依次映射到潜在空间，得到$z_t=[\xi(x_t^1), ..., \xi(x_t^F)]$

用CLIP编码给定的文本prompt $y$，得到embedding $\tau(y)$.

diffusion模型在潜在空间的denoiser: $\epsilon_{\theta}(z_t, t,\tau(y))$

### Key Frame Generation

我们生成了16个关键帧，然后用一个separate模型去插值这些帧，得到一个长的视频序列。

训练的时候，我们将输入图像通过预训练的VAE（stable diffusion的VAE）映射到潜在空间，然后用高斯扩散过程以及随机采样的时间步t来崩塌(corrupt)输入帧（不断加躁），然后再用一个3D的UNet decoder来去噪。

#### 2D+adaptor

直接用3D卷积的话，计算量太大，一般的做法是用一个空间上的2D卷积+时间上的1D卷积。

我们提出了2D+adaptor，其中adaptor比起1D卷积更加简单。

给定F个视频帧，经过共享的2D卷积来抽取它们的空间特征，之后通过分布调整参数来调整每个帧的中间特征的均值和方差：

$z_t^i=S^i \cdot Conv2d(z_t^i)+B^i$

S和B是两个可学习参数的group。为什么要这样设计呢？这是基于这样一个观测：每个视频短片的帧是语义相似的。帧之间的细小差异可能没有必要专门用到1D卷积层。我们通过一个小的参数群来建模这些差异。

#### Spatial and directed temporal attention

$z_t=S-Attn(z_{t-1})+T-Attn(z_{t-1})$

$S-Attn=Cross-Attn(LN(MHSA(LN(z_{t-1}))))$

MHSA是一个vision transformers中用到的标准的多头自注意力模块，LN是layer normalization, cross-Attn是交叉自注意力模块，其中的attention矩阵是计算frame tokens $z_{t-1}$和文本embedding $y$之间的

考虑到帧是有时间顺序的，我们提出了有向的时间上的attention。

$z_t$的维度是$F\times C\times H\times W$, 它们分别是特征通道的数量，batch size, 帧的数量和特征的空间维度。

首先将其reshape到维度$HW\times \text{Heads}\times F\times \frac{C}{\text {Heads}}$

Heads是注意力头的个数，将每个帧的每个pixel看作一个token。

时间注意力是应用到不同帧之间的特定空间位置的tokens中，建模它们的dynamics

我们通过三个线性transformations得到Q，K，V矩阵，时间注意力矩阵（维度是Heads x F x F）

$A_t=softmax(Q_tK_t^T/\sqrt d)\odot M$

d是每个头的embedding维度，M是一个下三角矩阵，$M_{p,q}=0$如果p > q，否则是1.

通过mask的实现，当前token只会被之前的token影响，跟未来的token是独立的。

#### Video Frame Sampling
一个视频包括了很多很多帧，成百上千个。

典型的做法是采样一个子集，用它来代表整个视频。但是这样会使得the same subset of sample keyframes包括了不同的信息，提高了训练的难度。

我们的做法是随机采样视频的一小部分，然后uniformly地在选择的视频子集中采样16帧，用来训练。然后计算对应的frame-per-second(FPS) v, based on the sampling frames and add it to the data features.将它转化为embedding

$emb_v=Linear(SiLU(Linear(Sin(v))))$

Sin是sinusoidal position embedding，会把FPS转成一个C维的embedding。SiLU是sigmoid ReLU。

这个embedding会加到视频帧特征z里面，在训练的时候，可以通过选择不同的FPS值控制帧的smoothness

#### Training Objective
帧重建的损失

![](img/Pasted%20image%2020230116155202.png)


### Frame interpolation
为了提升temporal resolution and smooth the generated video,我们用了一个separate frame interpolation network来在相邻的关键帧之间插入新帧。

### Super-resolution
为了生成高分辨率的视频，我们训练了一个基于diffusion的基于pixel空间的超分辨率模型(SR)，可以upsample 256x256到1024x1024。模型是用图片数据集训练的


## 核心亮点

## 主要收获

## 个人评价
