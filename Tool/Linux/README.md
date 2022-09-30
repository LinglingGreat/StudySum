# Linux

## 忘记密码

Ubuntu的默认root密码是随机的，即每次开机都有一个新的 root密码。我们可以在终端输入命令 sudo passwd，然后输入当前用户的密码，enter，终端会提示我们输入新的密码并确认，此时的密码就是root新密码。修改成功后，输入命令 su root，再输入新的密码就ok了。

忘记密码，或者输入密码登录不了：

[https://m.php.cn/centos/445366.html](https://m.php.cn/centos/445366.html "https://m.php.cn/centos/445366.html")

## 环境配置

vim /etc/profile

环境变量配置全攻略：[https://www.cnblogs.com/youyoui/p/10680329.html](https://www.cnblogs.com/youyoui/p/10680329.html "https://www.cnblogs.com/youyoui/p/10680329.html")

## 发送请求

[Linux命令发送Http的get或post请求(curl和wget两种方法)](https://blog.csdn.net/cyl937/article/details/52850304 "Linux命令发送Http的get或post请求(curl和wget两种方法)")

## 问题解决

### Linux运行文件时报错：bash: \$'\r': command not found

问题解决

这是因为Windows系统的文件换行使用的是\r\n，而Unix系统是\n

方式一

安装dos2unix来进行文件转换

`yum install -y dos2unix`

`dos2unix aaa.sh`

方式二

使用vim打开文件，然后使用命令`:set ff=unix`，保存文件

使用vim打开文件`vim aaa.sh`

转换格式`:set ff=unix`

保存文件`:wq`

### 如何实现本地代码和远程的实时同步

<https://cloud.tencent.com/developer/article/1607185?from=information.detail.服务器>

### 服务器运行程序Out of memory

1.使用命令nvidia-smi，看到GPU显存被占满：

![](https://img2018.cnblogs.com/blog/1456376/201903/1456376-20190310203743501-1838554376.png)

2.尝试使用 ps aux|grep PID命令查看占用GPU内存的线程的使用情况。如下

![](https://img2018.cnblogs.com/blog/1456376/201903/1456376-20190310203801474-1961502808.png)

如果是自己的程序，可以直接kill掉，`kill -9 PID`

[服务器上运行程序Out of memory 解决办法](https://www.cnblogs.com/E-Dreamer-Blogs/p/10507015.html "服务器上运行程序Out of memory 解决办法")

## 常用命令

### 机器自身情况

查看显卡信息

*   `lspci -vnn | grep VGA -A 12`

*   `lspci | grep -i nvidia`

查看gpu使用情况

*   `nvidia-smi`

*   `watch -n 0.1 nvidia-smi`

查看安装的显卡的驱动信息`cat /proc/driver/nvidia/version`

查看CUDA版本： `cat /usr/local/cuda/version.txt`或者`nvcc -V`

查看cuDNN版本：`cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2`

查看系统版本：`lsb_release -a 或 cat /etc/redhat-release`或cat /etc/issue或者是 cat /etc/lsb-release

修改登录后显示信息：`sudo vim /etc/motd`

查看安装了哪些cuda版本：`ls -l /usr/local | grep cuda`

查询ip：`curl cip.cc`

### 文件系统

[Linux磁盘空间分析及清理（df、du、rm）](https://www.cnblogs.com/jing99/p/10487174.html "Linux磁盘空间分析及清理（df、du、rm）")

*   df可以查看一级文件夹大小、使用比例、档案系统及其挂入点。

*   查看当前目录总共占的容量，而不单独列出各子项占用的容量`du -sh`

*   查看当前目录下一级子文件和子目录占用的磁盘容量`du -lh --max-depth=1`

*   统计当前文件夹(目录)大小：`du -sh *`，包括隐藏文件：`du -sh * .[^.]*`

*   查看当前目录以及所有下级目录、文件占用的磁盘容量`du -h`

*   [Linux查看文件或文件夹大小: du命令](https://blog.csdn.net/duan19920101/article/details/104823301 "Linux查看文件或文件夹大小: du命令")

查看目录挂载路径 `df -h 目录名`

（1）查看已经挂载的硬盘大小：`df -h`

（2）查看详细的硬盘分区情况（包括挂载和未挂载两种的硬盘大小）：`fdisk -l`

查看文件大小`ls -alh test/`

磁盘挂载：先`fdisk -l`查看分区情况，找到需要挂载的分区，比如`/dev/nvme0n1p1`，然后`mount 分区 挂载目录`，比如mount /dev/nvme0n1p1 /data

### 文件的增删改查等

`ncdu`可以看文件夹大小，而且可以删除

删除文件或目录`rm -rf /var/log/httpd`

复制文件用`cp`，移动文件用`mv`

| 命令格式                                         | 运行结果                                                                                                                                   |               |
| -------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | ------------- |
| mv source\_file(文件) dest\_file(文件)           | 将源文件名 source\_file 改为目标文件名 dest\_file                                                                                                  |               |
| mv source\_file(文件) dest\_directory(目录)      | 将文件 source\_file 移动到目标目录 dest\_directory 中                                                                                             |               |
| mv source\_directory(目录) dest\_directory(目录) | 目录名 dest\_directory 已存在，将 source\_directory 移动到目录名 dest\_directory 中；目录名 dest\_directory 不存在则 source\_directory 改名为目录名 dest\_directory | mv info/ logs |

`mv /usr/runoob/*  . `：将 **/usr/runoob** 下的所有文件和目录移到当前目录下

`cp * hist`：将当前目录下的所有文件（不包括文件夹）复制到hist文件夹

用指令 cp 将当前目录 **test/** 下的所有文件复制到新目录 **newtest** 下，输入如下命令：`cp –r test/ newtest`

`cp源文件名 新文件名` 复制并重命名

`cp -r !(file3) /sahil`复制所有目录并跳过单个目录

`cp -r !(dir2) /sahil`复制所有目录并跳过单个文件夹

删除除XYZ以外的文件`rm !(X|Y|Z)`（可以先echo打印出来看对不对）

[https://qastack.cn/unix/7215/deleting-all-files-in-a-folder-except-files-x-y-and-z](https://qastack.cn/unix/7215/deleting-all-files-in-a-folder-except-files-x-y-and-z "https://qastack.cn/unix/7215/deleting-all-files-in-a-folder-except-files-x-y-and-z")

寻找文件夹

`find / -type d -iname "deepspeed"`

`find / -type d -iname "liling"`

**查看/删除某个日期之前/之后的文件**

查看用echo，删除改成rm

时间可以改。比如%H%M是时+分

```bash
for filename in `ls`; do if [ `date -r $filename +%m%d` -gt "1125" ];then echo $filename; fi done 

for filename in `ls`; do if [ `date -r $filename +%m%d` -lt "1125" ];then echo $filename; fi done 


for filename in `ls`; do if [ `date -r $filename +%m%d` -eq "1125" ];then echo $filename; fi done 


```

### scp命令

**scp命令在工作中是比较常用的，所以就总结如下：**

1、拷贝本机/home/administrator/test整个目录（test整个文件夹复制过去）至远程主机192.168.1.100的/root目录下

`scp -r /home/administrator/test/ `[root@192.168.1.100](mailto:root@192.168.1.100 "root@192.168.1.100")`:/root/`

2、拷贝单个文件至远程主机

`scp /home/administrator/Desktop/old/driver/test/test.txt `[root@192.168.1.100](mailto:root@192.168.1.100 "root@192.168.1.100")`:/root/`
其实上传文件和文件夹区别就在参数 -r， 跟cp, rm的参数使用差不多， 文加价多个 -r

3、远程文件/文件夹下载

举例，把192.168.62.10上面的/root/文件夹，下载到本地的/home/administrator/Desktop/new/下，使用远程端的root登陆

`scp -r `[root@192.168.62.10](mailto:root@192.168.62.10 "root@192.168.62.10")`:/root/ /home/administrator/Desktop/new/`

比如：`scp -r `[xinchenTest@172.16.75.144](mailto:xinchenTest@172.16.75.144 "xinchenTest@172.16.75.144")`:/data/liling/EVA/eva-xinling/ ./`

从当前工作目录中复制所有文件，除了名为file4的文件。`scp -rp !(file4) 192.168.19.142:/sahilfile1`

本地传文件到远程服务器（需要通过跳板机）

`ssh -N -f -L 6000:`[172.16.75.144:22](http://172.16.75.144:22 "172.16.75.144:22")` -p 6002 xinchenTest@dll.cenbrain.club -o TCPKeepAlive=yes`

`scp -P 6000 本地文件路径 xinchenTest@localhost:服务器文件路径`

端口号

### rsync命令

将tmp目录下的文件复制到 /home/xx/，排除其中的dirc目录 &#x20;

`rsync -avP --exclude=dirc/tmp  /home/xx/ `

注意 dirc/ 后面的 / 一定要，指名是目录，如果不加的话 dirc文件也会被排除

如果想排除多个目录或文件的话 使用

`rsync -avP  --exclude-from=/usr/exclude.list`

exclude.list 必须是绝对路径，里面保存了各种要排除的文件或目录，以换行隔开

\-a, --archive                   归档模式，表示以递归方式传输文件，并保持所有文件属性，等于-rlptgoD 
\-v, --verbose                  详细模式输出
\-P 等同于 --partial            保留那些因故没有完全传输的文件，以是加快随后的再次传输

直接在后面加多个排除的目录

PythonAPI目录下的文件以及文件夹复制到局域网219的/usr目录下，其中排除.svn，build,dist,.vscode等文件夹 

`rsync -av --exclude .svn/ --exclude build/ --exclude dist/  --exclude .vscode/ --exclude `​

远程

`rsync -av --progress --exclude build/ 192.168.19.142:/sahilsending . `

### 文件中查找字符串

`grep -rn "hello,world!" *`

`*` : 表示当前目录所有文件，也可以是某个文件名

`-r` 是递归查找

`-n` 是显示行号

`-R` 查找所有文件包含子目录

`-i` 忽略大小写

下面是一些有意思的命令行参数：

`grep -i pattern files` ：不区分大小写地搜索。默认情况区分大小写

`grep -l pattern files` ：只列出匹配的文件名

`grep -L pattern files` ：列出不匹配的文件名

`grep -w pattern files` ：只匹配整个单词，而不是字符串的一部分（如匹配‘magic’，而不是‘magical’）

`grep -C number pattern files` ：匹配的上下文分别显示\[number]行，

`grep pattern1 | pattern2 files` ：显示匹配 pattern1 或 pattern2 的行，

`grep pattern1 files | grep pattern2` ：显示既匹配 pattern1 又匹配 pattern2 的行。

这里还有些用于搜索的特殊符号：\\< 和 \\> 分别标注单词的开始与结尾。

例如： 

`grep man *` 会匹配 ‘Batman’、‘manic’、‘man’等，

`grep '\<man' *` 匹配‘manic’和‘man’，但不是‘Batman’，

`grep '\<man\>'` 只匹配‘man’，而不是‘Batman’或‘manic’等其他的字符串。

`'^'`：指匹配的字符串在行首，

`'$'`：指匹配的字符串在行尾， 

2，xargs配合grep查找

`find -type f -name '*.php'|xargs grep 'GroupRecord'`

### vim

**1.跳转到文件头**

输入冒号(:)，打开命令输入框

输入命令1

**2.跳转到文件尾**

输入冒号(:)，打开命令输入框

输入命令：\$

### 进程相关

简单看`ps pid`

看进程的详细信息`cd /proc/pid`

看显卡的进程占用`fuser -v /dev/nvidia*`

查找特定进程`ps -ef|grep ssh`，其中ssh是keyword，用ef信息会更详细

批量kill进程`ps -ef|grep pretrain_gpt2|grep -v grep|cut -c 9-15|xargs kill -9`

[nvidia-smi 无进程占用GPU，但GPU显存却被占用了很多](https://blog.csdn.net/m0_38007695/article/details/88954699 "nvidia-smi 无进程占用GPU，但GPU显存却被占用了很多")

`nohup python .py > run.out &`

### screen

screen的用法：[https://blog.51cto.com/zz6547/1829625](https://blog.51cto.com/zz6547/1829625 "https://blog.51cto.com/zz6547/1829625")

1可以实现多个“屏幕”的效果。

2可以实现类似“后台执行”的效果，避免远程终端窗口中执行长时间任务时意外断开

3可以远程共享字符界面会话，就像远程桌面一样，两个人看到的画面一样，只不过screen是字符界面,而且有一个前提是，两个人必须登录同一台主机的同一个用户。

`screen -S` 会话名称，可以创建一个指定名称的会话，不指定名称的情况下，会话会有ID编号。

`screen -ls` 查看当前机器已经建立的screen会话

`screen -x ID`号  ，可以直接加入某个screen会话，不管这个会话是处于Attached状态还是Detached状态，都可以使用此命令加入。

`screen -r ID`号  ，这个命令可以还原到某个跳出（剥离）状态的会话，不能还原到处于Attached状态的会话，如果使用此命令还原某个处于Attached状态的会话，screen会提示你，这个会话者处于Attached状态，意思就是说，这个会话里面有人用，你自己考虑是否加进来，加入会话以后这个会话就由你俩共同控制了，处于Attached状态的会话只能加入，不能还原。

退出

*   如果已经处于某个screen会话中，使用exit命令 或者 使用ctrl+d 快捷键，表示关闭当前会话，同时这个会话中运行的程序也会关闭

*   如果已经处于某个screen会话中，使用Ctrl+a+d 快捷键，表示跳出（剥离）当前会话，这个会话以及会话中的程序不会被停止或关闭，它们一直在后台运行。

*   不管是否处于screen会话中，都可以使用screen-d ID号 ，剥离指定screen的会话，如果指定跳出的会话中已经有人在操作，那么这个人会被强行剔出会话，但是会话以及其中的程序都会在后台正常运行，也就是说这个会话会从Attached状态变成Detached状态，会话中的人也会被跳出。

翻页

*   linux在进入screen模式下之后，发现是无法在终端使用鼠标滚轮进行上下翻页拉动的，无法查看上面的终端输出内容了

*   先按Ctrl+a键，然后释放，然后再按\[键即可进入翻页模式。

*   Ctrl+c切换回之前模式

### nohup

```
nohup python graph_api.py > logs/graph.log & echo $! > graph.pid
```

### 统计代码行数

Linux有一个统计文件行数的命令wc。使用wc可以打印出每个文件和总文件的行数、字数和字节数，如果没有指定文件，则会读取标准输入(一般是终端)做统计。

1.统计当前目录下，java文件数量：

```
find . -name "*.java" |wc -l
```

2.统计当前目录下，所有java文件行数：

```
find . -name "*.java" |xargs cat|wc -l
```

```
find . "(" -name "*.cpp" -or -name "*.h" ")"  -print | xargs wc -l
```

3.统计当前目录下，所有java文件行数，并过滤空行：

```
find . -name "*.java" |xargs cat|grep -v ^$|wc -l
```



## 解压

解压tar文件`tar -xvf xxx -C 解压路径`

解压tar.bz2文件`tar -jxvf xxx`

解压tar.gz和tgz文件`tar -zxvf xxx`

把/home目录下包括它的子目录全部做备份文件，并进行压缩，备份文件名为usr.tar.gz 。　　
`tar czvf usr.tar.gz /home`

`tar -cf all.tar *.jpg` 这条命令是将所有.jpg的文件打成一个名为all.tar的包。-c是表示产生新的包 ，-f指定包的文件名。

 `tar -rf all.tar *.gif` 这条命令是将所有.gif的文件增加到all.tar的包里面去。-r是表示增加文件的意思。

`tar -tf all.tar` 这条命令是列出all.tar包中所有文件，-t是列出文件的意思

`tar -xf all.tar` 这条命令是解出all.tar包中所有文件，-x是解开的意思
 
特别注意，在参数 f 之后的文件档名是自己取的，我们习惯上都用 .tar 来作为辨识。
如果加 z 参数，则以 .tar.gz 或 .tgz 来代表 gzip 压缩过的 tar file ～
如果加 j 参数，则以 .tar.bz2 来作为附档名啊～

## 端口

[https://github.com/NVIDIA/tacotron2/issues/181](https://github.com/NVIDIA/tacotron2/issues/181 "https://github.com/NVIDIA/tacotron2/issues/181")

查看监听端口的进程：`netstat -nltp`

我要使用4040端口，但是被其他的程序占用了，查找占用的程序

`netstat -apn | grep 4040`

最后一项显示的是pid和对应的名称

## 下载
 后台下载：`-b`

wget：[https://www.cnblogs.com/peida/archive/2013/03/18/2965369.html](https://www.cnblogs.com/peida/archive/2013/03/18/2965369.html "https://www.cnblogs.com/peida/archive/2013/03/18/2965369.html")

## 数字

1-0.000005=0.000005，

2-4=4

[https://www.cnblogs.com/mengzhongshi/p/3319407.html](https://www.cnblogs.com/mengzhongshi/p/3319407.html "https://www.cnblogs.com/mengzhongshi/p/3319407.html")

## gcc

更新gcc和g++
- `yum install centos-release-scl
- 查看能安装的版本`yum list dev\*gcc //用于查看可以安装的版本
- 安装 `yum install devtoolset-8-gcc devtoolset-8-gcc-c++
- 临时生效`source /opt/rh/devtoolset-8/enable
- 一直生效的方法：
`echo "source /opt/rh/devtoolset-8/enable" >> /etc/bashrc

`source /etc/bashrc 
- 或者（临时生效）：`scl enable devtoolset-7 bash`



## 安装re2c和ninja

参考资料：http://www.manongjc.com/detail/20-nfwuuxzbajoxqos.html

如果有报错autoreconf命令找不到，安装：`dnf install autoconf automake libtool`

## Linux环境下conda虚拟环境的迁移

[Linux环境下conda虚拟环境的迁移](https://blog.csdn.net/qq_42730750/article/details/125413470)

[修改conda环境和缓存默认路径](https://blog.csdn.net/javastart/article/details/102563461)

