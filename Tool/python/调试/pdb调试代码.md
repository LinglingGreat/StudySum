# pdb调试代码

使用pdb来调试代码：

pdb.set\_trace();

[https://www.jianshu.com/p/fb5f791fcb18](https://www.jianshu.com/p/fb5f791fcb18 "https://www.jianshu.com/p/fb5f791fcb18")

| 命令             | 解释            |
| -------------- | ------------- |
| break 或 b 设置断点 | 设置断点          |
| continue 或 c   | 继续执行程序        |
| list 或 l       | 查看当前行的代码段     |
| step 或 s       | 进入函数          |
| return 或 r     | 执行代码直到从当前函数返回 |
| exit 或 q       | 中止并退出         |
| **next 或 n**   | 执行下一行         |
| pp             | 打印变量的值        |
| help           | 帮助            |

在第一次按下了 n+enter 之后可以直接按 enter 表示重复执行上一条 debug 命令。

在调试的时候可以动态改变变量的值，具体如下实例。需要注意的是下面有个错误，原因是 b 已经被赋值了，如果想重新改变 b 的赋值，则应该使用!b。

```python
[root@rcc-pok-idg-2255 ~]# python epdb2.py 
 > /root/epdb2.py(10)?() 
 -> b = "bbb"
 (Pdb) var = "1234"
 (Pdb) b = "avfe"
 *** The specified object '= "avfe"' is not a function 
 or was not found along sys.path. 
 (Pdb) !b="afdfd"
 (Pdb)
```
