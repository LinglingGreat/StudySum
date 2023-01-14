#### DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation

作者提出了使用稀缺词加种类词如“beikkpic dog”的组合文本来微调一组照片和这个文本的绑定。但是仅仅用少量的照片来微调一个有着大量参数的模型很明显会带来极为严重的过拟合。并且还会带来一个语言模型里特别常见的事情--灾难性遗忘。这两个问题的表现一个是绑定词的形态很难变换，就如Unitune一样。另一个问题是对种类词里面的种类生成也会快速失去多样性和变化性。于是针对这个问题作者针对性地提出了一个叫自身类别先验保存损失的损失函数。

![](img/Pasted%20image%2020221210161013.png)

这个函数的设计是在用户提供一个指定的类别和这个类别的一组图片（如自家的宠物狗的多张照片）后，模型同时使用“特殊词+类别”对用户照片训练和“类别”与模型生成的该类别图训练。这样做的好处是模型可以在将特定的照片主体与特殊词绑定的时候可以一起学到和其类别的关系，并且同时该类别的信息在不断的被重申以对抗用户照片信息的冲击。作者在训练的时候特意将这两个损失以一比一的比例训练了200个轮次左右。

prompt的基本格式：`a [identifier] [class noun]`，论文里说如果不加class noun，会需要更长的训练时间，效果会变差。

![](img/Pasted%20image%2020221210182506.png)




![](img/Pasted%20image%2020221128203837.png)

我们可以看到，生成的效果还是十分不错的。兼具了多样性以及可控性。虽然依然不是一个实时的算法，但训练成本不算很高，大约是10-15分钟左右的GPU单卡时间。在Huggingface上有不少大众训练的以各种风格训练的DreamBooth-StableDiffusion并且代码全部开源了。

局限性
- 模型可能无法生成少见的prompt context
- 物体的颜色会变化
- 容易过拟合生成和训练集一样的图片
- 某些主体相比其他的主体更容易学，比如猫和狗
- 少见或复杂的主体，模型无法支持主体的多变性（subject variations）
- 主体的保真度 fidelity也存在差异，一些生成的图像可能包含主体的幻觉特征hallucinated features，这取决于模型先验的强度和语义修改的复杂性

![](img/Pasted%20image%2020221210182923.png)



## 训练tips
DreamBooth和TextInversion的结果对比：
https://www.reddit.com/r/StableDiffusion/comments/xjlv19/comparison_of_dreambooth_and_textual_inversion/

huggingface训练DreamBooth的博客：
https://huggingface.co/blog/dreambooth

从dreambooth最初版本fork的项目，有一些训练tips
https://github.com/JoePenna/Dreambooth-Stable-Diffusion/

训练人物：
https://tryolabs.com/blog/2022/10/25/the-guide-to-fine-tuning-stable-diffusion-with-your-own-images
- 使用**10 到 12 张图像会产生更好的结果。**根据经验，我们将使用 2 张图像，包括躯干和 10 张面部，具有不同的背景、风格、表情、看和不看相机等。
- 在我们的实验中显示出良好结果的经验法则是每个训练图像使用 100 到 200 次迭代。
- inpainting实现换头

训练人物：
https://bennycheung.github.io/dreambooth-training-for-personal-embedding
- 使用JoePenna的项目，写了整个训练流程
- fix hand, eye


训练多个主体：
https://www.youtube.com/watch?v=ravETUa84P8&themeRefresh=1

训练人脸
- 当使用 2 的批量大小和 1e-6 的 LR 时，800-1200 步运行良好。
- 事先保存对于避免过度拟合很重要。

如果您看到生成的图像有噪声或质量下降，则可能意味着过度拟合。首先，尝试上述步骤来避免它。如果生成的图像仍然有噪声，请使用 DDIM 调度程序或运行更多推理步骤（~100 在我们的实验中效果很好）。

我们的最佳结果是结合使用文本编码器微调、低 LR 和适当数量的步骤获得的。

Faces are harder to train. In our experiments, a learning rate of `2e-6` with `400` training steps works well for objects but faces required `1e-6` (or `2e-6`) with ~1200 steps.

https://www.reddit.com/r/StableDiffusion/comments/zcr644/make_better_dreambooth_style_models_by_using/






