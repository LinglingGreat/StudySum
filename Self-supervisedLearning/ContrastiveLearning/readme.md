## 简介

对比学习是一种基于对比思想的判别式表示学习框架（或方法），主要用来做无监督（自监督）的表示学习(对比学习也可以用于有监督学习)

思想：将样例与与它语义相似的例子（正样例）和与它语义不相似的例子（负样例）进行对比，希望通过设计模型结构和对比损失，使语义相近的例子对应的表示在表示空间更接近，语义不相近的例子对应的表示距离更远，以达到类似聚类的效果

## 对齐性和均匀性

两个重要的概念，对齐性(alignment)和均匀性(uniformity)

由于对比学习的表示一般都会正则化，因而会集中在一个超球面上。对齐性和均匀性指的是好的表示空间应该满足两个条件：一个是相近样例的表示尽量接近，即对齐性；而不相近样例的表示应该均匀的分布在超球面上，即均匀性。满足这样条件的表示空间是线性可分的，即一个线性分类器就足以用来分类，因而也是我们希望得到的，我们可以通过这两个特性来分析表示空间的好坏。



对比学习有三个重要的组成部分：正负样例、对比损失以及模型结构。

## 正负样例

对于有监督的数据，正负样例很容易构造，同一标签下的例子互为正样例，不同标签下的例子互为负样例，但对于无标签的数据，我们如何得到正负样例呢？

目前的主流做法是对所有样例增加扰动，产生一些新的样例，同一个样例扰动产生的所有样例之间互为正样例，不同样例扰动产生的样例彼此之间互为负样例。现在的问题就变成了如何可以在保留原样例语义不变的情况下增加扰动，构造新样例。

图像领域中的扰动大致可分为两类：空间/几何扰动和外观/色彩扰动。空间/几何扰动的方式包括但不限于图片翻转（flip）、图片旋转（rotation）、图片挖剪（cutout）、图片剪切并放大（crop and resize）。外观扰动包括但不限于色彩失真、加高斯噪声等

自然语言领域的扰动也大致可分为两类：词级别（token-level）和表示级别（embedding-level）。词级别的扰动大致有句子剪裁（crop）、删除词/词块（span）、换序、同义词替换等。表示级别的扰动包括加高斯噪声、dropout等。

但是不同于图像领域，对自然语言的扰动很容易改变语义，这就会引入错误正例（False Positive）从而对模型训练产生不好的影响。同时，错误负例（False Negative）也是一个不容忽视的问题

## **对比损失**



## 参考资料

[谈一谈对比学习](https://mp.weixin.qq.com/s/NbDToz-y1gwBGwqkFHqL_g)