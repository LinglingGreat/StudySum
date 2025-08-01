
[Linux 修改默认 shell - 知乎](https://zhuanlan.zhihu.com/p/504677740)

[查看当前使用的shell - perlman - 博客园](https://www.cnblogs.com/softwaretesting/archive/2012/02/14/2350688.html)
## 环境配置

vim /etc/profile

环境变量配置全攻略：[https://www.cnblogs.com/youyoui/p/10680329.html](https://www.cnblogs.com/youyoui/p/10680329.html "https://www.cnblogs.com/youyoui/p/10680329.html")

## 忘记密码

Ubuntu的默认root密码是随机的，即每次开机都有一个新的 root密码。我们可以在终端输入命令 sudo passwd，然后输入当前用户的密码，enter，终端会提示我们输入新的密码并确认，此时的密码就是root新密码。修改成功后，输入命令 su root，再输入新的密码就ok了。

忘记密码，或者输入密码登录不了：

[https://m.php.cn/centos/445366.html](https://m.php.cn/centos/445366.html "https://m.php.cn/centos/445366.html")

## 发送请求

[Linux命令发送Http的get或post请求(curl和wget两种方法)](https://blog.csdn.net/cyl937/article/details/52850304 "Linux命令发送Http的get或post请求(curl和wget两种方法)")

# 问题解决

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

# 常用命令

### 机器自身情况

查看本机的操作系统和位数信息：`uname -m && cat /etc/*release`

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

查询公网ip：`curl cip.cc`或者` `

查询内网ip: `ifconfig -a`

cpu信息. `cat /proc/cpuinfo`

### 文件系统

[Linux磁盘空间分析及清理（df、du、rm）](https://www.cnblogs.com/jing99/p/10487174.html "Linux磁盘空间分析及清理（df、du、rm）")

*   df可以查看一级文件夹大小、使用比例、档案系统及其挂入点。

*   查看当前目录总共占的容量，而不单独列出各子项占用的容量`du -sh`, 排序 `du -sh * |sort -hr`或者 `du -sh * |sort -h`

*   查看当前目录下一级子文件和子目录占用的磁盘容量`du -lh --max-depth=1`
*  查看test目录下一级子文件和子目录的大小 `ls -alh test/`
*   统计当前文件夹(目录)大小：`du -sh *`，包括隐藏文件：`du -sh * .[^.]*`

*   查看当前目录以及所有下级目录、文件占用的磁盘容量`du -h`
*   [Linux查看文件或文件夹大小: du命令](https://blog.csdn.net/duan19920101/article/details/104823301 "Linux查看文件或文件夹大小: du命令")
* 指定显示的单位，加上`--block-size=g`或者m, k

统计当前目录下文件的个数（不包括目录）`ls -l | grep "^-" | wc -l`

统计当前目录下文件的个数（包括子目录）  `ls -lR| grep "^-" | wc -l`

查看某目录下文件夹(目录)的个数（包括子目录）  `ls -lR | grep "^d" | wc -l`

查找指定后缀的文件的个数   `find ./ -name "*.jpg" | wc -l`

查看目录挂载路径 `df -h 目录名`

（1）查看已经挂载的硬盘大小：`df -h`

（2）查看详细的硬盘分区情况（包括挂载和未挂载两种的硬盘大小）：`fdisk -l`

磁盘挂载：先`fdisk -l`查看分区情况，找到需要挂载的分区，比如`/dev/nvme0n1p1`，然后`mount 分区 挂载目录`，比如mount /dev/nvme0n1p1 /data

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

删除某个文件以外的文件`find . ! -name epoch=000184.ckpt -exec rm -f {} \;`

删除匹配文件名的文件`find . -name 'info.log.2020-06*' -exec rm {} \;`


**linux删除/tmp文件夹中前缀为mobi且最新修改时间在6小时以前的文件和文件夹**

要删除`/tmp`文件夹中前缀为`mobi`且最新修改时间在6小时以前的文件和文件夹，你可以使用以下命令：

```bash
find /tmp -name 'mobi*' -type f -mmin +360 -o -name 'mobi*' -type d -mmin +360 -exec rm -rf {} +
```

这个命令会在`/tmp`目录下查找以`mobi`开头的文件和文件夹，并且最新修改时间在6小时（360分钟）以前的。然后，使用`-exec rm -rf {} +`来删除匹配的文件和文件夹。

请注意，这个命令会直接删除符合条件的文件和文件夹，所以请谨慎使用。在执行之前，建议先使用`-print`选项来检查将要删除的文件和文件夹列表，以确保不会误删重要内容。例如：

```bash
find /tmp -name 'mobi*' -type f -mmin +360 -o -name 'mobi*' -type d -mmin +360 -print
```

这个命令会列出符合条件的文件和文件夹列表，你可以先检查这个列表，确认没有问题后再执行删除操作。同样地，请谨慎操作，以免意外删除重要文件和文件夹。

### 批量修改文件名(find & rename & sed)

[批量修改文件名(find & rename & sed)\_wx608b59dedd9e5的技术博客\_51CTO博客](https://blog.51cto.com/u_15187242/3773168)

```
字母的替换
rename "s/AA/aa/" *             //把文件名中的AA替换成aa
修改文件的后缀
rename "s/.html/.php/" *     //把.html 后缀的改成 .php后缀
批量添加文件后缀
rename "s/$/.txt/" *             //把所有的文件名都以txt结尾
批量删除文件名
rename "s/.txt//" *               //把所有以.txt结尾的文件名的.txt删掉


```

分割文件，按照行
`split -l 4000000 -d -a 2 --additional-suffix=".json" RedPajamaC4_train.json RedPajamaC4_train_
`

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

举例：`rsync -avh --progress /home/lldata/* .`

`rsync -avP --partial /path/to/source /path/to/destination`

`rsync -P -r --rsh=ssh user@ip:file path`,  `rsync -av user@ip:file path`

`-a, --archive`                   归档模式，表示以递归方式传输文件，并保持所有文件属性，等于-rlptgoD 

`-v, --verbose`                  详细模式输出。`-v`参数表示输出细节。`-vv`表示输出更详细的信息，`-vvv`表示输出最详细的信息。

`--include='*/' --include='.*'    同步隐藏目录以及隐藏文件

`-P`参数是`--progress`和`--partial`这两个参数的结合。

`--partial`参数允许恢复中断的传输。不使用该参数时，`rsync`会删除传输到一半被打断的文件；使用该参数后，传输到一半的文件也会同步到目标目录，下次同步时再恢复中断的传输。一般需要与`--append`或`--append-verify`配合使用。

`--partial-dir`参数指定将传输到一半的文件保存到一个临时目录，比如`--partial-dir=.rsync-partial`。一般需要与`--append`或`--append-verify`配合使用。

`--progress`参数表示显示进展。

`--exclude=dirc/tmp`  排除某个目录。
- 将tmp目录下的文件复制到 /home/xx/，排除其中的dirc目录 `rsync -avP --exclude=dirc/tmp  /home/xx/ `
- `rsync -av --exclude .svn/ --exclude build/ --exclude dist/  --exclude .vscode/ --exclude `

`--exclude-from`  参数指定一个本地文件，里面是需要排除的文件模式，每个模式一行。
- 比如`rsync -avP  --exclude-from=/usr/exclude.list`，exclude.list 必须是绝对路径

`--delete`参数删除只存在于目标目录、不存在于源目标的文件，即保证目标目录是源目标的镜像。

`--existing`、`--ignore-non-existing`参数表示不同步目标目录中不存在的文件和目录。

`--ignore-existing`参数表示只要该文件在目标目录中已经存在，就跳过去，不再同步这些文件。
- 举例：`rsync -a --ignore-existing root@ip:/home/folder .`

`-u`、`--update`参数表示同步时跳过目标目录中修改时间更新的文件，即不同步这些有更新的时间戳的文件。

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

或者 pkill -f "process_name"

[nvidia-smi 无进程占用GPU，但GPU显存却被占用了很多](https://blog.csdn.net/m0_38007695/article/details/88954699 "nvidia-smi 无进程占用GPU，但GPU显存却被占用了很多")

`nohup python .py > run.out &`

全部kill
```
import os
pid = list(set(os.popen('fuser -v /dev/nvidia*').read().split()))
kill_cmd = 'kill -9 ' + ' '.join(pid)
print(kill_cmd)
os.popen(kill_cmd)

```

### screen

screen的用法：[https://blog.51cto.com/zz6547/1829625](https://blog.51cto.com/zz6547/1829625 "https://blog.51cto.com/zz6547/1829625")

1可以实现多个“屏幕”的效果。

2可以实现类似“后台执行”的效果，避免远程终端窗口中执行长时间任务时意外断开

3可以远程共享字符界面会话，就像远程桌面一样，两个人看到的画面一样，只不过screen是字符界面,而且有一个前提是，两个人必须登录同一台主机的同一个用户。

`screen -S` 会话名称，可以创建一个指定名称的会话，不指定名称的情况下，会话会有ID编号。

`screen -ls` 查看当前机器已经建立的screen会话

`screen -x ID`号  ，可以直接加入某个screen会话，不管这个会话是处于Attached状态还是Detached状态，都可以使用此命令加入。

`screen -r ID`号  ，这个命令可以还原到某个跳出（剥离）状态的会话，不能还原到处于Attached状态的会话，如果使用此命令还原某个处于Attached状态的会话，screen会提示你，这个会话者处于Attached状态，意思就是说，这个会话里面有人用，你自己考虑是否加进来，加入会话以后这个会话就由你俩共同控制了，处于Attached状态的会话只能加入，不能还原。

`screen -S 会话名称 -X quit`删除会话

退出

*   如果已经处于某个screen会话中，使用exit命令 或者 使用ctrl+d 快捷键，表示关闭当前会话，同时这个会话中运行的程序也会关闭

*   如果已经处于某个screen会话中，使用Ctrl+a+d 快捷键，表示跳出（剥离）当前会话，这个会话以及会话中的程序不会被停止或关闭，它们一直在后台运行。

*   不管是否处于screen会话中，都可以使用screen-d ID号 ，剥离指定screen的会话，如果指定跳出的会话中已经有人在操作，那么这个人会被强行剔出会话，但是会话以及其中的程序都会在后台正常运行，也就是说这个会话会从Attached状态变成Detached状态，会话中的人也会被跳出。

翻页

*   linux在进入screen模式下之后，发现是无法在终端使用鼠标滚轮进行上下翻页拉动的，无法查看上面的终端输出内容了

*   先按Ctrl+a键，然后释放，然后再按\[键即可进入翻页模式。

*   Ctrl+c切换回之前模式

安装方式：yum install screen

如果没有权限

https://www.cnblogs.com/GoubuLi/p/12679471.html

https://blog.csdn.net/qq_36441393/article/details/107123645

```
#!/bin/bash

prefix=$HOME/pkgs
mkdir -p $prefix
cd $prefix
wget https://ftp.gnu.org/gnu/ncurses/ncurses-6.0.tar.gz
tar -xzf ncurses-6.0.tar.gz && rm ncurses-6.0.tar.gz
cd ncurses-6.0
./configure --prefix=$prefix
make && make install
cd .. && rm -rf ncurses-6.0

wget https://ftp.gnu.org/gnu/screen/screen-4.6.2.tar.gz
tar -xzf screen-4.6.2.tar.gz && rm screen-4.6.2.tar.gz
cd screen-4.6.2
export LDFLAGS="-L$prefix/lib"
export CPPFLAGS="-I$prefix/include"
./configure --prefix=$prefix
make && make install
cd .. && rm -rf screen-4.6.2

echo "PATH=$prefix/bin:"'$PATH' >> $HOME/.bashrc
source $HOME/.bashrc
```

### nohup

```
nohup python graph_api.py > logs/graph.log & echo $! > graph.pid
```

### 流量分析

安装nethogs工具: centos系统 yum install 或者`rpm -ivh nethogs-0.8.5-9.el8.x86_64.rpm`

执行nethogs命令即可。

## 压缩解压
压缩成zip文件`zip -r xxx.zip 目录或文件`

解压`unzip -o -d 输出目录 xxx.zip`

打包成tar文件`tar -cvf xxx.tar 目录或文件`.  正常打包情况下，打包后的目录下还有个原目录。 怎么去掉这个多余的文件夹呢？如下`tar -cvf config.tar -C config/ .`      打包包含隐藏目录的文件夹：`tar -cvf config.tar -C .[!.]*`

解压tar文件`tar -xvf xxx.tar -C 解压路径`

打包压缩成tar.gz文件`tar -zcvf xxx.tar.gz 目录或文件`

解压tar.gz和tgz文件`tar -zxvf xxx`

打包压缩成tar.bz2文件`tar -jcvf xxx.tar.gz 目录或文件`

解压tar.bz2文件`tar -jxvf xxx`

解压tar.zst文件`tar -I zstd -xvf xxxx.tar.zst`，需要install zstd

把/home目录下包括它的子目录全部做备份文件，并进行压缩，备份文件名为usr.tar.gz 。　　
`tar czvf usr.tar.gz /home`

`tar -cf all.tar *.jpg` 这条命令是将所有.jpg的文件打成一个名为all.tar的包。-c是表示产生新的包 ，-f指定包的文件名。

 `tar -rf all.tar *.gif` 这条命令是将所有.gif的文件增加到all.tar的包里面去。-r是表示增加文件的意思。

`tar -tf all.tar` 这条命令是列出all.tar包中所有文件，-t是列出文件的意思

`tar -xf all.tar` 这条命令是解出all.tar包中所有文件，-x是解开的意思
 
特别注意，在参数 f 之后的文件档名是自己取的，我们习惯上都用 .tar 来作为辨识。
如果加 z 参数，则以 .tar.gz 或 .tgz 来代表 gzip 压缩过的 tar file ～
如果加 j 参数，则以 .tar.bz2 来作为附档名啊～

**解压所有tar文件**

```
#!/bin/bash

TARDIR="xxx"

UNTARDIR="xxx"

  

printf "Entered path: $TARDIR.\n\n"

cd "$TARDIR"

  

for tar in *.tar

do

dirname=`echo $tar | sed 's/\.tar$//'`

printf "Directory name to extract this file is: %s.\n" $dirname

dirfullpath="$UNTARDIR/$dirname"

printf "Directory full path to extract this file is：%s.\n" $dirfullpath

mkdir "$dirfullpath"

tar -xvf $tar -C $dirfullpath

  

printf "\n\n"

done
```

## 端口

[https://github.com/NVIDIA/tacotron2/issues/181](https://github.com/NVIDIA/tacotron2/issues/181 "https://github.com/NVIDIA/tacotron2/issues/181")

查看监听端口的进程：`netstat -nltp`

我要使用4040端口，但是被其他的程序占用了，查找占用的程序

`netstat -apn | grep 4040`

```bash
# 使用 lsof 查看端口占用
lsof -i :12222

# 或者使用 netstat
netstat -tuln | grep 12222
```

[mp.weixin.qq.com/s/l9k49nf93nj1o3oEx3ya6w](https://mp.weixin.qq.com/s/l9k49nf93nj1o3oEx3ya6w)

最后一项显示的是pid和对应的名称

`kill -9 $(lsof -i tcp:34567 -t)`

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

source /ssdwork/miniconda3/etc/profile.d/conda.sh

[Linux环境下conda虚拟环境的迁移](https://blog.csdn.net/qq_42730750/article/details/125413470)

[修改conda环境和缓存默认路径](https://blog.csdn.net/javastart/article/details/102563461)



# 定时任务

在 Linux 系统中，有几种方式可以查询定时任务：

## 1. 查看 crontab 定时任务

**查看当前用户的 crontab 任务**：

`crontab -l`

**查看特定用户的 crontab 任务**（需要 root 权限）：

`crontab -u username -l`

## 2. 查看系统级定时任务

**查看系统 crontab 配置文件**：

`cat /etc/crontab`

**查看 cron 目录下的定时任务**：

`ls -l /etc/cron.d/ ls -l /etc/cron.daily/ ls -l /etc/cron.hourly/ ls -l /etc/cron.weekly/ ls -l /etc/cron.monthly/`

## 3. 查看 systemd 定时器（现代 Linux 系统）

**列出所有定时器**：

`systemctl list-timers --all`

## 4. 查看 at 任务（一次性定时任务）

`atq`

或

`at -l`

## 5. 查看 anacron 任务

`cat /etc/anacrontab`

## 6. 查看 cron 服务状态

`systemctl status cron    # 对于使用 systemd 的系统 service cron status      # 对于使用 SysV init 的系统`

这些命令可以帮助您全面了解系统上配置的各种定时任务。


# slurm

查看节点信息`sinfo -N`或者`sinfo`
- alloc——节点在用
- idle——节点可用
- mix——部分占用
- down——节点下线
- drain——节点故障

查看节点的内存，gpu等信息 `scontrol show nodes`。scontrol 可以用于查看分区、节点和作业的详细信息，也可以修改等待中的作业属性。可以使用 **man scontrol** 查看详细用法。

查询作业状态 `squeue`, 如果名字太长可以用这个：

```bash
squeue --format="%.18i %.9P %.30j %.8u %.8T %.10M %.9l %.6D %R"
```

`scontrol show job <jobid>`。这会显示该作业的所有设置和状态信息，包括作业被分配到的具体节点。

删除作业：`scancel`
- -u, --user=user_name 删除特定用户的作业
- -i, --interactive交互模式. 对每一个作业进行确认
- -n, --jobname=job_name 删除特定名称的作业
- -p, --partition=partition_name 删除特定分区的作业

1.  scancel后接作业id,删除对应作业
2.  scancel后接参数，将删除所有满足参数的作业

### 提交作业

提交作业 `sbatch job_script`, 比如`sbatch train.sh`, 注意sh文件中的python执行命令不要加nohup

可以使用 **man sbatch** 命令查看sbatch详细用法。常用的：

```bash
#SBATCH --job-name=test     # create a short name for your job
#SBATCH --nodes=2                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=32G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:8             # number of gpus per node
#SBATCH -w xxx,xxx               # specified nodes
#SBATCH --output=file_name       # output file, default is slurm_number.out
```

这里以多机多卡为例，单机多卡，单机单卡是类似的，改一下相应的sbatch参数以及torchrun或accelerate launch参数即可。

因为slurm会运行`nodes*ntasks-per-node`次程序，所以**根据运行参数中是否存在当前节点不同值也不同的变量（比如torch run中的node_rank）**，分为两种情况：

1.如果你的启动命令中没有随着当前节点不同值也不同的变量（比如torch run中的node_rank），那么可以直接写一个slurm启动脚本slurm_start.sh:

可以参考 https://github.com/PrincetonUniversity/multi_gpu_training/tree/main/02_pytorch_ddp

```bash
#!/bin/bash
#SBATCH --job-name=ddp-torch     # create a short name for your job
#SBATCH --nodes=2                # node count
#SBATCH --ntasks-per-node=2      # total number of tasks per node
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=32G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:2             # number of gpus per node
#SBATCH --output=%x-%j.out           # output file name, %x is jobname, %y is jobid

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE
#SBATCH --time=00:05:00          # total run time limit (HH:MM:SS)
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

# module purge
# module load anaconda3/2021.11
# conda activate torch-env

srun python main.py --epochs=100
```

2.否则的话，需要分两步。

先写一个启动你的程序的脚本slurm_main.sh，这里是每个机器使用8张卡：
```bash
#!/bin/bash
echo COUNT_NODE=$COUNT_NODE
echo HOSTNAMES = $HOSTNAMES
echo hostname = `hostname`
echo MASTER_ADDR= $MASTER_ADDR
echo MASTER_PORT= $MASTER_PORT

H=`hostname`
# 这里根据你的hostname规则来定，比如我的hostname是xxx.cluster.com, HOSTNAMES是xxx yyy
THEID=`echo -e $HOSTNAMES  | python -c "import sys;[sys.stdout.write(str(i)) for i,line in enumerate(next(sys.stdin).split(' ')) if line.strip() == '$H'.strip().split('.')[0]]"`
echo THEID=$THEID

# 如果你使用的是torchrun
torchrun --nproc_per_node=8 --nnodes=$COUNT_NODE --node_rank=$THEID  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT main.py

# 如果你使用的是accelerate launch
accelerate launch --num_processes $(( 8 * $COUNT_NODE )) --num_machines $COUNT_NODE --multi_gpu --mixed_precision fp16 --machine_rank $THEID --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT main.py
```


然后写一个slurm启动脚本slurm_start.sh, 使用2台机器
```bash
#!/bin/bash
#SBATCH --job-name=test     # create a short name for your job
#SBATCH --nodes=2                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks per node, crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=32G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:8             # number of gpus per node
#SBATCH -w xxx,xxx         # specified nodes

# sent to sub script
# ******************* These are read internally it seems ***********************************
# ******** Master port, address and world size MUST be passed as variables for DDP to work
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
master_port=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_PORT=$master_port
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
# ******************************************************************************************
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Nodelist:= " $SLURM_JOB_NODELIST
echo "Number of nodes:= " $SLURM_JOB_NUM_NODES
echo "Ntasks per node:= "  $SLURM_NTASKS_PER_NODE
echo "MASTER_PORT:= " $MASTER_PORT
echo "WORLD_SIZE:= " $WORLD_SIZE
echo "MASTER_ADDR:= " $MASTER_ADDR
echo "COUNT_NODE:= " $COUNT_NODE
echo "HOSTNAMES:= " $HOSTNAMES
echo "GPUS:= " $SLURM_GPUS_ON_NODE
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "

srun slurm_main.sh


```

上述两种情况都是直接在登录节点切换到你的对应环境，然后运行`sbatch slurm_start.sh`

### 可能的报错
1. 脚本运行后卡住不动
- 参数设置少了，比如torch run少了个`--nproc_per_node`参数

2. oom：报错信息Some of your processes may have been killed by the cgroup out-of-memory handler
- 调低sbatch的参数`cpus-per-task`和`mem`

3. 权限问题：运行脚本后出现error: execve():xxx.sh: Permission denied的提示。
- 将xxx.sh的文件权限修改为755(`chmod 755 xxx.sh`)

4. RuntimeError: Socket Timeout
- 网络问题，重试，换节点重试

5. Address already in use
- ntasks-per-node参数改成1，slurm会运行`nodes*ntasks-per-node`次程序，如果一个node上有多个task，那么会运行多次，这样就会导致地址冲突，通常在torch run中出现

6. NCCL error
- 可能是代码有问题，检查下rank, world_size等

参考资料：
https://cloud.tencent.com/developer/article/2135660?shareByChannel=link

https://gist.github.com/rom1504/474f97a95a526d40ae44a3fc3c657a2e


## 修改时间

cp /usr/share/zoneinfo/GMT /etc/localtime

sudo date 1031145622   月日时分年

hwclock --set --date="09/17/2003 13:26:00"。 月日年时分秒.格林尼治时间

## 一键部署
```#!/bin/bash
for ((machine_id=11; machine_id<=25; machine_id++))
do
if [ $machine_id -ge 10 ]
then
    machine="xxx"${machine_id}
else
    machine="xxx0"${machine_id}
fi
echo $machine
ssh $machine > /dev/null 2>&1 << eeooff
cd path/
conda activate env
bash stop.sh && bash server_prod.sh
exit
eeooff
echo done!
sleep 2m
done
```

# 权限

### 修改账号创建文件默认权限

针对未来可能会创建的文件，需要修改每个账号的创建文件的默认权限：

修改~/.bashrc（或者/etc/profile，如果有权限的话），加入以下代码：

```
if [ $UID -gt 199 ] && [ "`/usr/bin/id -gn`" = "`/usr/bin/id -un`" ]; then
	umask 002
else
	umask 002
fi
```

修改后保存，source ~/.bashrc

执行成功后，新创建的文件/文件夹有群组可读可写权限

### 修改已有文件权限

`chmod -R 777 文件夹` 赋予所有人（包括群组、其他用户）可读可写权限

### 多用户共享目录conda安装

-   登录自己的账号
    
-   运行source xxx/miniconda3/bin/activate
    
-   运行conda init
    
-   修改 ~/.bash_profile，然后source ~/.bash_profile
```
# if running bash
if [ -n "$BASH_VERSION" ]; then
    # include .bashrc if it exists
    if [ -f "$HOME/.bashrc" ]; then
        . "$HOME/.bashrc"
    fi
fi
```
    
-   即可使用conda命令

### 文件夹权限指定给用户

没有root权限的情况下可以使用setfacl命令。

1. **基本语法**:

`setfacl [选项] 文件名`

2. **常见选项**:
    
    - `-m`：添加 ACL 条目。
    - `-x`：移除 ACL 条目。
    - `-b`：移除所有 ACL 条目。
    - `-R`：递归地应用 ACL 到目录及其子目录。
    - `-d`：设置默认 ACL。

如果您想指定一个文件夹，并且只允许特定的用户访问它，您可以使用 `setfacl` 命令来设置访问控制列表（ACL）规则。这里是一个如何操作的步骤指南： 
1. **移除其他所有用户的访问权限**: 首先，为了确保只有特定的用户可以访问该文件夹，您需要移除除了文件所有者之外的所有其他用户的访问权限。使用 `chmod` 命令可以进行基础权限设置： ```chmod 700 /path/to/directory ``` 这个命令将设置文件夹的权限，使得只有所有者可以读、写以及执行（访问）这个文件x夹。
2. **添加特定用户的访问权限**: 之后，您可以使用 `setfacl` 来为一个特定用户添加访问权限。假设我们要给用户名为 `username` 的用户添加这个权限： ```setfacl -m u:username:rwx /path/to/directory ``` 这条命令实际上为用户名为 `username` 的用户设置了读（r）、写（w）和执行（x）权限。 移除权限`setfacl -x u:username file.txt
3. **设置默认 ACL 权限**: 如果您还想要确保未来在此目录中创建的所有文件和子目录都继承同样的访问权限，您可以设置默认 ACL 权限： ```setfacl -d -m u:username:rwx /path/to/directory ``` 这条命令设置了默认的 ACL 权限，以便任何在该目录中新创建的文件和子目录都会自动给 `username` 用户分配读写执行权限。 
4. **查看 ACL 权限**: 为了确认您的权限已正确设定，可以查看这个目录的当前 ACL 配置： ```getfacl /path/to/directory ``` 通过以上步骤，您将能设置一个文件夹让特定用户访问。请注意，以上操作可能需要您具有管理权限，或至少对文件夹有适当的读写权限。

[setfacl 命令，Linux setfacl 命令详解：设置文件访问控制列表 - Linux 命令搜索引擎](https://wangchujiang.com/linux-command/c/setfacl.html)

# nginx

```
$HOME/nginx/sbin/nginx -s stop  # 先停止Nginx
$HOME/nginx/sbin/nginx -c $HOME/nginx/conf/nginx.conf  # 启动Nginx
$HOME/nginx/sbin/nginx -s reload # 重新启动
nginx -t #测试配置是否正确
```

#### a. 下载并编译 Nginx

首先，从 Nginx 官方网站 下载最新的源码包。然后编译并安装到你的用户目录下。

```
# 下载 Nginx 源码
wget http://nginx.org/download/nginx-1.25.2.tar.gz
tar -zxvf nginx-1.25.2.tar.gz
cd nginx-1.25.2

# 配置安装路径到用户目录
./configure --prefix=$HOME/nginx --without-http_rewrite_module --without-http_gzip_module

# 编译并安装
make && make install
```

安装后，Nginx 会被放在 `$HOME/nginx` 目录中。

#### b. 运行 Nginx

使用编译好的 Nginx 可以在用户权限下运行。首先，进入 Nginx 安装目录并启动：

`cd $HOME/nginx ./sbin/nginx`

要停止 Nginx，可以使用以下命令：

`./sbin/nginx -s stop`

### 2. 配置 Nginx 负载均衡

现在可以编辑 Nginx 的配置文件，配置路径通常在 `$HOME/nginx/conf/nginx.conf`。

### 配置示例

```nginx
http {
    upstream backend_servers {
        server internal-service1:8080;
        server internal-service2:8080;
        server internal-service3:8080;
    }

    server {
        listen 8081;  # 使用非特权端口（1024以上端口），因为没有root权限
        server_name localhost;

        location / {
            proxy_pass http://backend_servers;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}

```

你可以通过 `localhost:8081` 访问 Nginx，它会将请求负载均衡到内部的服务。



### 3. 调整 Nginx 监听的端口

由于你没有 `root` 权限，Nginx 不能监听 1024 以下的端口（如 80 或 443）。你可以让 Nginx 监听高端口。

### 4. 运行和测试

完成配置之后，可以通过以下命令来启动 Nginx：

`$HOME/nginx/sbin/nginx -c $HOME/nginx/conf/nginx.conf`

测试是否连通. `curl http://localhost:8081

### 5. 设置开机自启动（可选）

如果你想让 Nginx 自动在登录时启动，可以将启动命令添加到 `.bashrc` 或 `.bash_profile` 中：

`echo "$HOME/nginx/sbin/nginx -c $HOME/nginx/conf/nginx.conf" >> ~/.bashrc`

通过这种方式，即使你没有 `root` 权限，依然可以运行 Nginx 并实现负载均衡。

# k8s

### 查看事件

#### 检查集群事件

```Bash
kubectl get events --sort-by='.lastTimestamp'
kubectl get events --sort-by=.metadata.creationTimestamp | grep vip
```

#### 查看可用gpu

```bash
kubectl describe nodes | grep -E "nvidia.com/gpu|HolderIdentity:" 
kubectl describe nodes | grep -E "nvidia.com/gpu|name:"
```

kubectl describe nodes | grep -A 10 "Allocated resources"

### 服务启动

```Markdown
kubectl apply -f model.yaml
```

查看部署了哪些服务deployment/pod：

```Markdown
kubectl get deployment
kubectl get pods -o wide
```

```
# 显示 Pod 的详细信息，包括事件历史、容器状态等
kubectl describe pod your_pod_name

# scaledobject名字是config里面的名字 
kubectl get scaledobject 
kubectl describe scaledobject <name> 

# hpa的名字是自动启的，每次查看一下 
kubectl get hpa 
kubectl describe hpa <your hpa name>
```

查看pod启动后的log

```Markdown
kubectl logs -f your_pod_name
```

进入pod内部

```Plain
kubectl exec -it <pod-name> -- /bin/bash
```

删除服务，谨慎使用：
- 注意，使用deployment启动的服务，不要直接删除pod，删除了会自动重启。**需要用`delete deployment <name>`进行删除

```Bash
kubectl delete deployment  <name>
```

### 滚动重启

1. 滚动重启 Deployment:（**deployment**名字叫做：xx-name）
    

```Bash
kubectl rollout restart deployment/xx-name
```

2. 扩展 Deployment 到 5 个副本:
    

```Bash
kubectl scale deployment/xx-name --replicas=5
```

4. 查看 Deployment 的滚动更新状态:
    

```Bash
kubectl rollout status deployment/xx-name
```

5. 如果需要，回滚到之前的版本:
    

```Bash
kubectl rollout undo deployment/xx-name
```

6. 暂停滚动更新（如果你需要暂停正在进行的更新）:
    

```Bash
kubectl rollout pause deployment/xx-name
```

7. 恢复已暂停的滚动更新:
    

```Bash
kubectl rollout resume deployment/xx-name
```

Deployment 的详细信息:

```Bash
kubectl describe deployment xx-name
```

```bash
# 使用下面的命令，可以在本地访问promethus里面的监控数据 
kubectl port-forward -n monitoring svc/prometheus-k8s 9090:9090 

# grafana监控转发 
kubectl port-forward svc/grafana -n monitoring 8082:3000
```


# 安装网络监控系统指南

为了实时监控您的网络连接和性能，特别是与远程 Redis 服务器的连接，我推荐几种可在线查看的网络监控解决方案。以下是详细的安装和配置指南：

## 1. Netdata - 轻量级实时监控系统

Netdata 是一个非常轻量级但功能强大的实时性能监控工具，特别适合监控网络连接。

### 安装步骤

```bash
# 一键安装脚本
bash <(curl -Ss https://my-netdata.io/kickstart.sh)

# 或者通过包管理器安装
# 对于基于 Debian/Ubuntu 的系统
apt update
apt install netdata

# 对于基于 RHEL/CentOS 的系统
yum install epel-release
yum install netdata
```

### 配置网络监控

```bash
# 编辑配置文件
nano /etc/netdata/netdata.conf

# 确保以下部分存在并正确配置
[web]
    bind to = 0.0.0.0
    allow connections from = *
    allow dashboard from = *

# 重启服务
systemctl restart netdata
```

### 访问监控面板
- 打开浏览器访问 `http://服务器IP:19999`
- 查看 "Network Interfaces" 和 "Network Stack" 部分

### 添加 Redis 连接监控
```bash
# 创建自定义监控脚本
mkdir -p /usr/local/bin/netdata-custom
nano /usr/local/bin/netdata-custom/redis_connection_monitor.sh

# 添加以下内容
#!/bin/bash
REDIS_HOST="your_redis_server_ip"
REDIS_PORT="6379"

# 获取连接状态
ESTABLISHED=$(netstat -an | grep $REDIS_HOST:$REDIS_PORT | grep ESTABLISHED | wc -l)
TIME_WAIT=$(netstat -an | grep $REDIS_HOST:$REDIS_PORT | grep TIME_WAIT | wc -l)
CLOSE_WAIT=$(netstat -an | grep $REDIS_HOST:$REDIS_PORT | grep CLOSE_WAIT | wc -l)

# 输出结果给Netdata
echo "BEGIN redis_connections.connections $1"
echo "SET established $ESTABLISHED"
echo "SET time_wait $TIME_WAIT"
echo "SET close_wait $CLOSE_WAIT"
echo "END"

# 设置执行权限
chmod +x /usr/local/bin/netdata-custom/redis_connection_monitor.sh

# 配置Netdata使用此脚本
nano /etc/netdata/charts.d.conf
# 添加: redis_connection_monitor=yes
```

## 2. Prometheus + Grafana - 企业级监控方案

对于更强大的监控需求，Prometheus 配合 Grafana 是理想选择。

### 安装 Prometheus

```bash
# 下载Prometheus
cd /tmp
wget https://github.com/prometheus/prometheus/releases/download/v2.37.0/prometheus-2.37.0.linux-amd64.tar.gz
tar xvfz prometheus-*.tar.gz
cd prometheus-*

# 创建目录
sudo mkdir -p /etc/prometheus /var/lib/prometheus

# 复制文件
sudo cp prometheus /usr/local/bin/
sudo cp promtool /usr/local/bin/
sudo cp -r consoles/ console_libraries/ /etc/prometheus/

# 创建配置文件
sudo nano /etc/prometheus/prometheus.yml

# 添加基本配置
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
  
  - job_name: 'node'
    static_configs:
      - targets: ['localhost:9100']
```

### 安装 Node Exporter（系统监控）

```bash
# 下载Node Exporter
cd /tmp
wget https://github.com/prometheus/node_exporter/releases/download/v1.3.1/node_exporter-1.3.1.linux-amd64.tar.gz
tar xvfz node_exporter-*.tar.gz
cd node_exporter-*

# 复制文件
sudo cp node_exporter /usr/local/bin/

# 创建systemd服务
sudo nano /etc/systemd/system/node_exporter.service

# 添加以下内容
[Unit]
Description=Node Exporter
After=network.target

[Service]
User=node_exporter
Group=node_exporter
Type=simple
ExecStart=/usr/local/bin/node_exporter

[Install]
WantedBy=multi-user.target

# 创建用户
sudo useradd -rs /bin/false node_exporter

# 启动服务
sudo systemctl daemon-reload
sudo systemctl start node_exporter
sudo systemctl enable node_exporter
```

### 创建 Prometheus 服务

```bash
# 创建systemd服务
sudo nano /etc/systemd/system/prometheus.service

# 添加以下内容
[Unit]
Description=Prometheus
Wants=network-online.target
After=network-online.target

[Service]
User=prometheus
Group=prometheus
Type=simple
ExecStart=/usr/local/bin/prometheus \
    --config.file /etc/prometheus/prometheus.yml \
    --storage.tsdb.path /var/lib/prometheus/ \
    --web.console.templates=/etc/prometheus/consoles \
    --web.console.libraries=/etc/prometheus/console_libraries

[Install]
WantedBy=multi-user.target

# 创建用户
sudo useradd -rs /bin/false prometheus

# 设置权限
sudo chown prometheus:prometheus /etc/prometheus /var/lib/prometheus -R
sudo chown prometheus:prometheus /usr/local/bin/prometheus /usr/local/bin/promtool

# 启动服务
sudo systemctl daemon-reload
sudo systemctl start prometheus
sudo systemctl enable prometheus
```

### 安装 Grafana

```bash
# 添加Grafana仓库
# 对于Debian/Ubuntu
sudo apt-get install -y apt-transport-https software-properties-common
sudo wget -q -O /usr/share/keyrings/grafana.key https://packages.grafana.com/gpg.key
echo "deb [signed-by=/usr/share/keyrings/grafana.key] https://packages.grafana.com/oss/deb stable main" | sudo tee -a /etc/apt/sources.list.d/grafana.list
sudo apt-get update
sudo apt-get install grafana

# 对于RHEL/CentOS
sudo nano /etc/yum.repos.d/grafana.repo
# 添加以下内容
[grafana]
name=grafana
baseurl=https://packages.grafana.com/oss/rpm
repo_gpgcheck=1
enabled=1
gpgcheck=1
gpgkey=https://packages.grafana.com/gpg.key
sslverify=1
sslcacert=/etc/pki/tls/certs/ca-bundle.crt

# 安装
sudo yum install grafana

# 启动服务
sudo systemctl daemon-reload
sudo systemctl start grafana-server
sudo systemctl enable grafana-server
```

### 配置 Grafana

1. 访问 `http://服务器IP:3000`
2. 默认用户名/密码：admin/admin
3. 添加 Prometheus 数据源
4. 导入网络监控仪表板（ID: 1860）

## 3. 简单的网络监控脚本（带Web界面）

如果您需要一个更简单的解决方案，可以使用以下自定义脚本：

```bash
# 安装必要的包
sudo apt-get install apache2 php php-json

# 创建监控脚本
sudo mkdir -p /opt/network-monitor
sudo nano /opt/network-monitor/monitor.sh

# 添加以下内容
#!/bin/bash
REDIS_HOST="your_redis_server_ip"
REDIS_PORT="6379"
OUTPUT_DIR="/var/www/html/network-stats"
mkdir -p $OUTPUT_DIR

while true; do
    # 获取时间戳
    TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
    
    # 网络连接状态
    CONNECTIONS=$(netstat -an | grep $REDIS_HOST:$REDIS_PORT | awk '{print $6}' | sort | uniq -c)
    
    # Ping延迟
    PING_RESULT=$(ping -c 1 $REDIS_HOST | grep "time=" | awk -F "time=" '{print $2}' | awk '{print $1}')
    
    # 保存数据
    echo "{\"timestamp\":\"$TIMESTAMP\",\"ping\":\"$PING_RESULT\",\"connections\":\"$CONNECTIONS\"}" >> $OUTPUT_DIR/data.json
    
    # 保持最新的1000条记录
    tail -1000 $OUTPUT_DIR/data.json > $OUTPUT_DIR/data.json.tmp
    mv $OUTPUT_DIR/data.json.tmp $OUTPUT_DIR/data.json
    
    # 每10秒采集一次
    sleep 10
done

# 设置执行权限
sudo chmod +x /opt/network-monitor/monitor.sh

# 创建systemd服务
sudo nano /etc/systemd/system/network-monitor.service

# 添加以下内容
[Unit]
Description=Network Monitor Service
After=network.target

[Service]
ExecStart=/opt/network-monitor/monitor.sh
Restart=always

[Install]
WantedBy=multi-user.target

# 启动服务
sudo systemctl daemon-reload
sudo systemctl start network-monitor
sudo systemctl enable network-monitor
```

### 创建Web界面

```bash
# 创建HTML文件
sudo nano /var/www/html/network-monitor.html

# 添加以下内容
<!DOCTYPE html>
<html>
<head>
    <title>Redis Network Monitor</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .container { width: 80%; margin: 0 auto; }
        .chart-container { height: 400px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Redis Network Monitor</h1>
        <div class="chart-container">
            <canvas id="pingChart"></canvas>
        </div>
        <div class="chart-container">
            <canvas id="connectionsChart"></canvas>
        </div>
    </div>

    <script>
        // 加载数据
        async function loadData() {
            const response = await fetch('network-stats/data.json');
            const text = await response.text();
            const lines = text.trim().split('\n');
            return lines.map(line => JSON.parse(line));
        }

        // 更新图表
        async function updateCharts() {
            const data = await loadData();
            
            // 准备数据
            const labels = data.map(item => item.timestamp);
            const pingData = data.map(item => parseFloat(item.ping));
            
            // Ping图表
            const pingCtx = document.getElementById('pingChart').getContext('2d');
            new Chart(pingCtx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Ping Latency (ms)',
                        data: pingData,
                        borderColor: 'rgb(75, 192, 192)',
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false
                }
            });
            
            // 连接状态图表
            // 这里需要解析连接数据，根据实际格式调整
        }

        // 初始加载
        updateCharts();
        
        // 每30秒刷新一次
        setInterval(updateCharts, 30000);
    </script>
</body>
</html>
```

## 4. 使用 Smokeping 监控网络延迟

Smokeping 专门用于监控网络延迟和丢包率，非常适合监控到 Redis 服务器的网络质量。

```bash
# 安装Smokeping
# 对于Debian/Ubuntu
sudo apt-get install smokeping

# 对于RHEL/CentOS
sudo yum install epel-release
sudo yum install smokeping

# 配置Smokeping
sudo nano /etc/smokeping/config.d/Targets

# 添加Redis服务器监控
+ Redis
menu = Redis Server
title = Redis Server Connection
++ RedisServer
menu = Redis Main Server
title = Redis Server Latency
host = your_redis_server_ip

# 重启服务
sudo systemctl restart smokeping

# 访问Web界面
# 通常在 http://服务器IP/smokeping/smokeping.cgi
```

## 推荐选择

1. **简单快速方案**：Netdata - 安装简单，几分钟内即可运行，提供丰富的实时数据
2. **企业级方案**：Prometheus + Grafana - 更强大、可扩展，但设置较复杂
3. **专注网络延迟**：Smokeping - 专门针对网络延迟监控，图表直观
4. **轻量级自定义**：自定义脚本 - 最轻量，但功能有限

无论选择哪种方案，建议配置告警功能，在网络连接出现异常时及时通知您。这样您就可以实时监控与 Redis 服务器的连接情况，更容易发现凌晨超时问题的根本原因。