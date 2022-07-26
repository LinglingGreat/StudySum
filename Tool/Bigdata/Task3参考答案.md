### Q1：HDFS是用来解决什么问题的？

A1：HDFS是一个分布式文件系统，常用于大规模文件存储，例如GB/TB/PB级别的文件，具有以下特点：高吞吐，高可靠，高可用，可扩展。

HDFS采用master/slave架构。一个HDFS集群是由一个Namenode(非HA)和一定数目的Datanodes组成。

- Namenode是一个中心服务器，负责管理文件系统的名字空间(namespace)以及客户端对文件的访问。
- Datanode一般是一个节点一个，负责管理它所在节点上的存储。

HDFS暴露了文件系统的名字空间，用户能够以文件的形式在上面存储数据。从内部看，一个文件其实被分成一个或多个数据块，这些块存储在一组Datanode上。Namenode执行文件系统的名字空间操作，比如打开、关闭、重命名文件或目录。它也负责确定数据块到具体Datanode节点的映射。Datanode负责处理文件系统客户端的读写请求。在Namenode的统一调度下进行数据块的创建、删除和复制。
！[HDFS架构](https://hadoop.apache.org/docs/r1.0.4/cn/images/hdfsarchitecture.gif)

### Q2:熟悉hdfs常用命令

A2: 
```bash
# 1. 创建目录
hadoop fs -mkdir /test
# 2. 递归的创建目录，如果上级目录不存在，自动创建
hadoop fs- mkdir  /user/test 
# 3. 上传文件
hadoop fs -put hello.txt /user/test/ 
# 4. 下载文件
hadoop fs -get /user/test
# 5. 重命名或移动文件
hadoop fs -mv /user/test/hello.txt /user/test/olleh.txt
# 6. 列出当前目录
hadoop fs -ls  
# 7. 递归的列出文件
hadoop fs -ls -R /
# 8. 递归删除文件,删除/user/下的所有文件
hadoop fs -rm -R /user/* 
# 9. 统计hdfs对应路径下的目录个数，文件个数，文件总计大小
hadoop fs -count /user/test/
# 10. 显示hdfs对应路径下每个文件夹和文件的大小
hadoop fs -du /user/test 
```
其他更多命令的使用，请自行查阅

### Q3. Python操作HDFS的其他API
A3. 
```python
import pyhdfs
fs =pyhdfs.HdfsClient(hosts='localhost,50070',user_name='yourname')
#返回这个用户的根目录
fs.get_home_directory()
#返回可用的namenode节点
fs.get_active_namenode()

resp = fs.listdir('/')
print(','.join(resp))
```
更多API的使用，请自行查阅

### Q4. 观察上传后的文件，上传大于128M的文件与小于128M的文件有何区别？
A4. 在Hadoop中，大文件的存储需要分块，分块的大小可以配置，配置项为dfs.blocksize，默认的是配置是128M。

### Q5. 启动HDFS后，会分别启动NameNode/DataNode/SecondaryNameNode，这些进程的的作用分别是什么？
#### NameNode
1. 整个文件系统的文件目录树，
2. 文件/目录的元信息和每个文件对应的数据块列表。3. 接收用户的操作请求。

NameNode维护着2张表：
1. 文件系统的目录结构，以及元数据信息
2. 文件与数据块（block）列表的对应关系

元数据存放在fsimage中，在运行的时候加载到内存中的(读写比较快)。
操作日志写到edits中。（类似于LSM树中的log）

（刚开始的写文件会写入到内存中和edits中，edits会记录文件系统的每一步操作，当达到一定的容量会将其内容写入fsimage中）

### SecondaryNameNode
负责合并edits和fsimage，为什么不放在NameNode中合并？NameNode要向外提供文件操作服务，如果把合并操作放在NameNode中，一定程度上会影响NameNode的性能。

### DataNode
1. 使用block形式存储。在hadoop2中，默认的大小是128MB。
2. 使用副本形式保存数据的安全，默认的数量是3个。

### Q6. NameNode是如何组织文件中的元信息的，edits log与fsImage的区别？使用hdfs oiv命令观察HDFS上的文件的metadata

```bash
hdfs oiv -i fsimage_xxxx -o ~/fsimage.xml -p XML
```
```xml
<inode>
    <id>16392</id>
    <type>FILE</type>
    <name>jdk-8u60-linux-x64.tar.gz</name>
    <replication>1</replication>
    <mtime>1452395391526</mtime>
    <atime>1496214541163</atime>
    <perferredBlockSize>134217728</perferredBlockSize>
    <permission>root:supergroup:rw-r--r--</permission>
    
    <blocks>
        <block>
            <id>1073741827</id>
            <genstamp>1003</genstamp>
            <numBytes>134217728</numBytes>
        </block>
        <block>
            <id>1073741828</id>
            <genstamp>1004</genstamp>
            <numBytes>47020915</numBytes>
        </block>
    </blocks>
</inode>
```
可以看到namenode维护了整个文件系统的目录树，同时存了文件与block的映射信息，以及一些文件元信息

### Q7 HDFS文件上传与下载过程

