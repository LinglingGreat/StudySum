
![](img/Pasted%20image%2020220927102938.png)

代码：

https://github.com/huggingface/diffusers （huggingface版本推理）

https://github.com/CompVis/stable-diffusion （源码版本推理）

博客：https://github.com/huggingface/blog/blob/main/stable_diffusion.md

https://towardsdatascience.com/how-to-fine-tune-stable-diffusion-using-textual-inversion-b995d7ecc095

https://jalammar.github.io/illustrated-stable-diffusion/

论文：https://ommer-lab.com/research/latent-diffusion-models/

[生成扩散模型漫谈（一）：DDPM = 拆楼 + 建楼](https://kexue.fm/archives/9119)


[从大一统视角理解扩散模型（Diffusion Models）](https://mp.weixin.qq.com/s/pESFKnWU0M4BqGF1KI8xQw)


自动编码器重构过程可以学习到隐变量z，但容易过拟合。当模型遇到没有出现过的新样本时，有时得不到有意义的生成结果。

变分自动编码机（Variational Auto Encoder，VAE）则是在 AE 的基础上，对隐变量z施加限制。使得z符合一个标准正态分布。这样的好处是，当隐变量z是趋向于一个分布时，对隐变量进行采样，其生成的结果可以是类似于输入样本，但是不完全一样的数据。这样避免了 AE 容易过拟合的问题。VAE 通常用于数据生成，一个典型的应用场景就是通过修改隐变量，生成整体与原样本相似，但是局部特征不同的新人脸数据。

![](img/Pasted%20image%2020221108232451.png)

具体来说，是通过变分推断这一数学方法将p(z)和p(z|X)后验概率设置为标准高斯分布，同时约束生成的样本尽量与输入样本相似。这样通过神经网络可以学习出解码器，也就是p(X|z)。通过这样的约束训练之后，可以使得隐变量z符合标准高斯分布。当我们需要生成新的样本时，可以通过采样隐变量z让变分自动编码机生成多样同时可用的样本。

整个学习过程中，变分自动编码机都在进行“生成相似度”和“生成多样性”之间的一个 trade off。当隐变量  的高斯分布方差变小趋向为 0 时，模型接近 AE。此时模型的生成样本与输入样本相似度较高，但是模型的样本生成采样范围很小，生成新的可用样本的能力不足。

对抗生成网络（GAN）与 VAE 和 AE 的“编码器-解码器”结构不同。GAN 没有 encoder 这一模块。GAN 直接通过生成网络（这里可以理解为 decoder）和一个判别网络（discriminator）的对抗博弈，使得生成网络具有较强的样本生成能力。GAN 可以从随机噪声生成样本，这里可以把随机噪声按照 VAE 中的隐变量理解。

而新出现的扩散模型（Denoising Diffusion Probabilistic Model，DDPM），其实在整体原理上与 VAE 更加接近 [2,3]。 X_0是输入样本，比如是一张原始图片，通过T步前向过程（Forward process）采样变换，最后生成了噪声图像X_T，这里可以理解为隐变量z。这个过程是通过马尔科夫链实现的。

  

在随机过程中，有这样一个定理。一个模型的状态转移如果符合马尔科夫链的状态转移矩阵时，当状态转移到一定次数时，模型状态最终收敛于一个平稳分布。这个过程也可以理解为溶质在溶液中溶解的过程，随着溶解过程的进行，溶质（噪声）最终会整体分布到溶液（样本）中。这个过程可以类比理解为 VAE 中的 encoder。而逆向过程（Reverse process）可以理解为 decoder。通过T步来还原到原始样本。


![](img/Pasted%20image%2020221108232832.png)

DALLE-1 [4] 模型结构如图所示，首先图像在第一阶段通过 dVAE（离散变分自动编码机）训练得到图像的 image tokens。文本 caption 通过文本编码器得到 text tokens。Text tokens 和 image tokens 会一起拼接起来用作 Transformer 的训练。这里 Transformer 的作用是将 text tokens 回归到 image tokens。当完成这样的训练之后，实现了从文本特征到图像特征的对应。

  

在生成阶段，caption 通过编码器得到 text tokens，然后通过 transformer 得到 image tokens，最后 image tokens 在通过第一阶段训练好的 image decoder 部分生成图像。因为图像是通过采样生成，这里还使用了 CLIP 模型对生成的图像进行排序，选择与文本特征相似度最高的图像作为最终的生成对象。

![](img/Pasted%20image%2020221108232930.png)

DALLE-2 [5] 模型结构如图 6 所示。其中的 text encoder 和 image encoder 就是用 CLIP 中的相应模块。在训练阶段通过训练 prior 模块，将 text tokens 和  image tokens 对应起来。同时训练 GLIDE 扩散模型，这一步的目的是使得训练后的 GLIDE 模型可以生成保持原始图像特征，而具体内容不同的图像，达到生成图像的多样性。

  
当生成图像时，模型整体类似在 CLIP 模型中增加了 prior 模块，实现了文本特征到图像特征的对应。然后通过替换 image decoder 为 GLIDE 模型，最终实现了文本到图像的生成。

![](img/Pasted%20image%2020221108232958.png)

Imagen [6] 生成模型还没有公布代码和模型，从论文中的模型结构来看，似乎除了文本编码器之外，是由一个文本-图像扩散模型来实现图像生成和两个超分辨率扩散模型来提升图像质量。


![](img/Pasted%20image%2020221108233123.png)

最新的 Imagic 模型 [7]，号称可以实现通过文本对图像进行 PS 级别的修改内容生成。目前没有公布模型和代码。从原理图来看，似乎是通过在文本-图像扩散模型的基础上，通过对文本嵌入的改变和优化来实现生成内容的改变。如果把扩散模型替换成简单的 encoder 和 decoder，有点类似于在 VAE 模型上做不同人脸的生成。只不过是扩散模型的生成能力和特征空间要远超过 VAE。


![](img/Pasted%20image%2020221108233204.png)

Stable diffusion [8] 是有 Stability AI 公司开发并且开源的一个生成模型。图 9 是它的结构图。其实理解了扩散模型之后，对 Stable diffusion 模型的理解就非常容易了。

  

朴素的 DDPM 扩散模型，每一步都在对图像作“加噪”、“去噪”操作。而在 Stable diffusion 模型中，可以理解为是对图像进行编码后的 image tokens 作加噪去噪。而在去噪（生成）的过程中，加入了文本特征信息用来引导图像生成（也就是图中的右边 Conditioning 部分）。这部分的功能也很好理解，跟过去在 VAE 中的条件 VAE 和 GAN 中的条件 GAN 原理是一样的，通过加入辅助信息，生成需要的图像。

