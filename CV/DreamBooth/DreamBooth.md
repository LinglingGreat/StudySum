#### DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation

作者提出了使用稀缺词加种类词如“beikkpic dog”的组合文本来微调一组照片和这个文本的绑定。但是仅仅用少量的照片来微调一个有着大量参数的模型很明显会带来极为严重的过拟合。并且还会带来一个语言模型里特别常见的事情--灾难性遗忘。这两个问题的表现一个是绑定词的形态很难变换，就如Unitune一样。另一个问题是对种类词里面的种类生成也会快速失去多样性和变化性。于是针对这个问题作者针对性地提出了一个叫自身类别先验保存损失的损失函数。

![](img/Pasted%20image%2020221128203821.png)

这个函数的设计是在用户提供一个指定的类别和这个类别的一组图片（如自家的宠物狗的多张照片）后，模型同时使用“特殊词+类别”对用户照片训练和“类别”与模型生成的该类别图训练。这样做的好处是模型可以在将特定的照片主体与特殊词绑定的时候可以一起学到和其类别的关系，并且同时该类别的信息在不断的被重申以对抗用户照片信息的冲击。作者在训练的时候特意将这两个损失以一比一的比例训练了200个轮次左右。


![](img/Pasted%20image%2020221128203837.png)

我们可以看到，生成的效果还是十分不错的。兼具了多样性以及可控性。虽然依然不是一个实时的算法，但训练成本不算很高，大约是10-15分钟左右的GPU单卡时间。在Huggingface上有不少大众训练的以各种风格训练的DreamBooth-StableDiffusion并且代码全部开源了。


DreamBooth和TextInversion的结果对比：
https://www.reddit.com/r/StableDiffusion/comments/xjlv19/comparison_of_dreambooth_and_textual_inversion/

huggingface训练DreamBooth的博客：
https://huggingface.co/blog/dreambooth
