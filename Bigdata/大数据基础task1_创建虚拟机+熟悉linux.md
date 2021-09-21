任务：

1. [创建三台虚拟机](https://mp.weixin.qq.com/s/WkjX8qz7nYvuX4k9vaCdZQ)
2. 在本机使用Xshell连接虚拟机
3. [CentOS7配置阿里云yum源和EPEL源](https://www.cnblogs.com/jimboi/p/8437788.html)
4. 安装jdk
5. 熟悉linux 常用命令
6. 熟悉，shell 变量/循环/条件判断/函数等

shell小练习1： 编写函数，实现将1-100追加到output.txt中，其中若模10等于0，则再追加输出一次。即10，20...100在这个文件中会出现两次。

注意：

- 电脑系统需要64位(4g+)
- 三台虚拟机的运行内存不能超过电脑的运行内存
- 三台虚拟机ip不能一样，否则会有冲突、

参考资料：

1. [安装ifconfig](https://jingyan.baidu.com/article/363872ec26bd0f6e4aa16f59.html)
2. [bash: wget: command not found的两种解决方法](https://www.cnblogs.com/areyouready/p/8909665.html)
3. linux系统下载ssh服务
4. [关闭windows的防火墙！如果不关闭防火墙的话，可能和虚拟机直接无法ping通！](https://www.linuxidc.com/Linux/2017-11/148427.htm)
5. 大数据软件 ：[链接](https://pan.baidu.com/s/17fEq3IPVoeE29cWCrSpO8Q) 提取码：finf

时间：2天



好的，任务开始！我的电脑是win10系统，64G，内存8G，满足条件。

1. [创建三台虚拟机](https://mp.weixin.qq.com/s/WkjX8qz7nYvuX4k9vaCdZQ) 

先安装virtualBox，一直点击下一步即可，可以修改想要安装的位置。之后按照上述链接里的基本步骤执行，不过对于我的电脑有几个不同之处：

（1）在虚拟机的设置——网络——界面名称处，教程里选择的是Realtek选项，而我的电脑网卡型号是Intel，因此需要改成自己电脑的网卡型号。查看电脑网卡型号的方式是：所有设置——网络和Internet——状态——更改适配器选项——右键查看自己正在连接的网络的属性即可。如果这一步设置不正确，后续无法连接网络。

（2）在安装centos系统时，教程中有一步设置网络的configure，需要更改IPv4 Settings和General中的一些默认设置，不知道为什么我安装时无法编辑configure中的内容，即进入configure后鼠标失效，键盘的上下左右键可用，但仍旧无法更改和保存。这一步还不清楚其左右，先跳过，后续如果出问题了再看。

（上述过程花了约1.5小时——第一天晚上）



\2. 在本机使用Xshell连接虚拟机

因为在第一步没有设置网络的configure选项，意味着虚拟机的IP是默认动态的，XShell远程操作虚拟机时，需要虚拟机的IP地址，动态的IP地址显然很不方便，因此我们需要把虚拟机的IP设置为静态的。具体过程如下：

1）查看网卡文件名：

```
ll /etc/sysconfig/network-scripts/|grep ifcfg-en
```

可以看到下方出现了一个ifcfg开头的文件名，就是你的网卡文件名。

2）打开网卡文件，比如我的网卡文件名是ifcfg-enp0s3，因此输入：

```
vi /etc/sysconfig/network-scripts/ifcfg-enp0s3
```

就打开了文件，在键盘上按"i"就可以进行文件的修改，修改后的内容大致是这样的：

```
TYPE=Ethernet
DEFROUTE=yes
PEERDNS=yes
PEERROUTES=yes
IPV4_FAILURE_FATAL=no
IPV6INIT=yes
IPV6_AUTOCONF=yes
IPV6_DEFROUTE=yes
IPV6_PEERDNS=yes
IPV6_PEERROUTES=yes
IPV6_FAILURE_FATAL=no
NAME=enp0s3
UUID=23b2b3e7-e6d5-4a6d-83b4-f6949392a486
DEVICE=enp0s3

#static assignment
ONBOOT=yes #开机启动
BOOTPROTO=static #静态IP
IPADDR=192.168.1.151 #本机地址，注意这里的地址的192.168.1是与你的本机(windows)的IP地址前三个字段保持一致的，可以在本机进入cmd界面输入ipconfig查看
NETMASK=255.255.255.0 #子网掩码
GATEWAY=192.168.1.1 #默认网关
```

最重要的是最后面五个，后三个都是文件中本来没有的，需要添加，前两项在文件中更改即可。

我还设置了下面两个字段，虽然不知道有什么用(⊙o⊙)…：

```
DNS1=192.168.1.1
DNS2=8.8.8.8
```

设置好之后。按esc键退出编辑模式，再输入":wq"即可退出vim界面。

3）重启网络服务，输入命令

```
systemctl restart network
```

即可。

4）开启sshd服务。

```
service sshd start
```

5）静态IP设置好了，可以通过ip addr命令查看自己的虚拟机IP地址，也就是刚刚设置好的地址。

可以"ping www.baidu.com"命令来检查能否顺利连接网络（ping的过程较长，可按ctrl+c停止）。

尝试虚拟机ping主机（主机IP通过ipconfig命令获取），以及主机ping虚拟机

```
ping 你的虚拟机IP地址或主机IP地址
```

![img](https://pic3.zhimg.com/v2-47279fa81ed018379849e86dadec2fae_b.png)

上面是我的主机ping虚拟机的结果。

如果虚拟机无法ping通主机的话（比如输入命令后长时间没有进展），可以尝试关闭主机的防火墙（所有设置——更新和安全——Windows安全——防火墙和网络保护——关闭防火墙）。但是一直关闭防火墙也不好呀，可以按照如下方法设置入站规则：

所有设置——更新和安全——Windows安全——防火墙和网络保护——高级设置——点击入站规则——找到“文件和打印共享（回显请求 – ICMPv4-In）”——右键启用规则



6）Xshell的安装比较简单，从[官网](https://www.netsarang.com/zh/xshell-download/)下载好软件，正常安装即可。

安装好后在Xshell中新建会话，输入虚拟机IP。在用户身份验证处，输入用户名和密码，点击确定。选择想要的会话，点击连接即可。

![img](https://pic1.zhimg.com/v2-fab072b6cbddfda67fbfacce5c46d50c_b.png)

![img](https://pic1.zhimg.com/v2-215e268dacdc9a75d7b7d76b82e6fc94_b.png)

参考资料：

<https://blog.csdn.net/ZZY1078689276/article/details/77280814>

<https://blog.csdn.net/qq_25908611/article/details/79077532> 



\3. [CentOS7配置阿里云yum源和EPEL源](https://www.cnblogs.com/jimboi/p/8437788.html) 

其实安装好Xshell之后，之后的操作都可以通过Xshell远程操作虚拟机，这样比较方便，而且可以直接复制粘贴各种命令，哈哈哈！

1）首先要安装wget，即输入命令yum -y install wget。

2）然后按照教程来就可以，没有遇到什么问题。

注意：如果你在执行“mv *.repo repo_bak/”命令之后才install wget，会出现“There are no enabled repos”错误，只需要把刚刚move过去的文件再move回来即可。



\4. 安装jdk

我是在windows中下载好jdk文件，通过xshell传输文件到虚拟机。

1）在windows下通过Xshell连接虚拟机，输入下列命令，在linux虚拟机中安装上传下载工具包rz及sz：

```
yum install -y lrzsz
```

2）上传文件到虚拟机。输入命令

```
rz
```

后，会弹出一个对话框，选择需要上传到虚拟机的文件即可，默认保存在当前目录下。

3）下面开始安装jdk。

在usr/local下创建目录java并进入目录下：

```
cd /usr/local/
mkdir java
cd java
```

将jdk文件移动(mv)或复制(cp)到创建的目录下：

```
mv /root/jdk-8u131-linux-x64.tar.gz /usr/local/java/
```

其中/root/是我一开始传输文件到虚拟机所保存的位置，jdk-8u131-linux-x64.tar.gz是文件名。

解压文件到当前目录

```
tar -zxvf jdk-8u131-linux-x64.tar.gz
```

编辑配置文件：

```
vi /etc/profile
```

设置好环境变量（注意不要有空格之类的）：

```
JAVA_HOME=/usr/local/java/jdk1.8.0_131
CLASSPATH=$JAVA_HOME/lib/
PATH=$PATH:$JAVA_HOME/bin
export PATH JAVA_HOME CLASSPATH
```

执行命令使得更改生效：

```
source /etc/profile
```

检查是否安装成功

```
java -version
```

![img](https://pic3.zhimg.com/v2-5316bcae18a53ff371395ba055af12ea_b.png)

参考资料：

通过xshell传输文件的教程：<https://blog.csdn.net/love666666shen/article/details/75742077> 

安装jdk的教程：<https://blog.csdn.net/szxiaohe/article/details/76650266> 



\5. 熟悉linux 常用命令

其实通过前面几个步骤，已经差不多熟悉linux的常用命令了。

```
ls   # 列出当前目录下的文件
cd A  # 切换到目录A
mv A B  # 移动文件A到B目录下
tar -zxvf *.tar.gz     # 解压压缩包
mkdir    # 创建目录
...
```



6.熟悉shell 变量/循环/条件判断/函数等

```
vi test_func.sh
```

在文件里写入（按“i”进入编辑模式）：

```
#! /bin/bash

echo "测试写入output.txt文件"
for((i=1;i<=100;i++));
do
echo $i >> output.txt
b=$(( $i % 10 ))
if [ $b = 0 ];then
echo $i >> output.txt
fi
done
```

注意if语句后面的"["后是有空格的，光这个就折腾了我好长时间.....对shell脚本还是不熟啊！

按esc键退出编辑模式，输入:wq保存文件，然后执行bash test_func.sh即可。

ls后会发现当前目录下多了一个output.txt文件，vi output.txt可以查看文件内容。

参考资料：

<https://blog.csdn.net/liuxizhen2009/article/details/22472297> 

[写入文件，追加内容，修改内容；shell,sed - 陳聽溪 - 博客园](https://www.cnblogs.com/taosim/articles/3761007.html) 





以上是第二天的学习内容。其中2、3、4是需要在三台虚拟机上运行的。

虽然现在回看每个步骤都挺简单的，但在做的过程中还是会遇到各种问题，比如我在安装jdk的时候，环境变量的写法一开始是类似于这样的：

```
export JAVA_HOME=/usr/java/jdk1.8.0_131/
export JRE_HOME=${JAVA_HOME}/jre
export CLASSPATH=.:${JAVA_HOME}/lib/dt.jar:${JAVA_HOME}/lib/tools.jar
export PATH=${JAVA_HOME}/bin:$PATH
```

但是怎么都不成功，找不到java。网上找了别的写法，一改，居然就好了。

另外就是shell脚本真的写了很久。。。我连for循环都要去查一下怎么写(⊙o⊙)…

从晚上10点做到第二天2点。。。emmm希望明天不要瞌睡！