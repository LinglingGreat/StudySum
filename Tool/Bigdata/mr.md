# 1.MapReduce 原理

完整的 `MapReduce` 框架包含两部分：

1. 算法逻辑层面，即 `map`、`shuffle` 以及 `reduce` 三个重要算法组成部分；
2. 实际运行层面，即算法逻辑作业在分布式主机中是以什么形式和什么流程运行的，自 `MapReduce version2` 以后，作业都是提交给 `YARN` 进行管理

P.S.

2.0之前只有MapReduce的运行框架，那么它里面有只有两种节点，一个是master，一个是worker。master既做资源调度又做程序调度，worker只是用来参与计算的。 　　  但是在2.0之后加入了YARN集群，Yarn集群的主节点承担了资源调度，Yarn集群的从节点中会选出一个节点（这个由redourcemanager决定）用作类似于2.0之前的master的工作，来进行应用程序的调度。

资源调度： 处理程序所需要的cpu、内存资源，以及存储数据所需要的硬盘资源都是resourcemanager去分配的。

![img](https://ask.qcloudimg.com/http-save/yehe-1195962/ppsamstmy5.png?imageView2/2/w/1620)

一切都是从最上方的user program开始的，user program链接了MapReduce库，实现了最基本的Map函数和Reduce函数。
图中执行的顺序都用数字标记了。

mapreduce就是分治法的一种，将输入进行分片，然后交给不同的task进行处理，然后合并成最终的解。 
mapreduce实际的处理过程可以理解为**Input->Map->Sort->Combine->Partition->Reduce->Output**。

以上详细内容可见：<https://cloud.tencent.com/developer/article/1023958> 

这篇文章[<http://oserror.com/distributed/mapreduce/>] 还举了一些mapreduce的例子，比较通俗易懂。



2.0之后，MapReduce框架通常由三个操作（或步骤）组成：

1. **Map**：每个工作节点将 `map` 函数应用于本地数据，并将输出写入临时存储。主节点确保仅处理冗余输入数据的一个副本。
2. **Shuffle**：工作节点根据输出键（由 `map` 函数生成）重新分配数据，对数据映射排序、分组、拷贝，目的是属于一个键的所有数据都位于同一个工作节点上。
3. **Reduce**：工作节点现在并行处理每个键的每组输出数据。

流程图：

![img](https://user-gold-cdn.xitu.io/2018/10/4/1663d77230e1bbd5?imageView2/0/w/1280/h/960/format/webp/ignore-error/1)

详细内容参考：

<https://juejin.im/post/5bb59f87f265da0aeb7118f2>



## 2. 使用Hadoop Streaming -python写出WordCount



## 3.使用mr计算movielen中每个用户的平均评分。



<https://www.cnblogs.com/sykblogs/articles/10021703.html>

<https://www.cnblogs.com/sss4/p/10443497.html>



## 4.使用mr实现merge功能。根据item，merge movielen中的 u.data u.item



## 5.使用mr实现去重任务。



## 6.使用mr实现排序。



## 7.使用mapreduce实现倒排索引。



## 8.使用mapreduce计算Jaccard相似度。



## 9.使用mapreduce实现PageRank。