## OS平台编程的需求
* 目录文件的操作
    * 对系统目录、文件的操作方法
* 程序定时执行
* 可执行程序的转换
    * python程序向可执行程序的转换


**os模块中关于文件/目录常用的函数使用方法**

| **函数名**                     | **使用方法**                                 |
| --------------------------- | ---------------------------------------- |
| getcwd()                    | 返回当前工作目录                                 |
| chdir(path)                 | 改变工作目录                                   |
| listdir(path='.')           | 列举指定目录中的文件名（'.'表示当前目录，'..'表示上一级目录）       |
| mkdir(path)                 | 创建单层目录，如该目录已存在抛出异常                       |
| makedirs(path)              | 递归创建多层目录，如该目录已存在抛出异常，注意：'E:\\a\\b'和'E:\\a\\c'并不会冲突 |
| remove(path)                | 删除文件                                     |
| rmdir(path)                 | 删除单层目录，如该目录非空则抛出异常                       |
| removedirs(path)            | 递归删除目录，从子目录到父目录逐层尝试删除，遇到目录非空则抛出异常        |
| rename(old, new)            | 将文件old重命名为new                            |
| system(command)             | 运行系统的shell命令                             |
| walk(top)                   | 遍历top路径以下所有的子目录，返回一个三元组：(路径, [包含目录], [包含文件])【具体实现方案请看：第30讲课后作业^_^】 |
| *以下是支持路径操作中常用到的一些定义，支持所有平台* |                                          |
| os.curdir                   | 指代当前目录（'.'）                              |
| os.pardir                   | 指代上一级目录（'..'）                            |
| os.sep                      | 输出操作系统特定的路径分隔符（Win下为'\\'，Linux下为'/'）     |
| os.linesep                  | 当前平台使用的行终止符（Win下为'\r\n'，Linux下为'\n'）     |
| os.name                     | 指代当前使用的操作系统（包括：'posix',  'nt', 'mac', 'os2', 'ce', 'java'） |

**os.path模块中关于路径常用的函数使用方法**

| **函数名**                     | **使用方法**                                 |
| --------------------------- | ---------------------------------------- |
| basename(path)              | 去掉目录路径，单独返回文件名                           |
| dirname(path)               | 去掉文件名，单独返回目录路径                           |
| join(path1[, path2[, ...]]) | 将path1, path2各部分组合成一个路径名                 |
| split(path)                 | 分割文件名与路径，返回(f_path, f_name)元组。如果完全使用目录，它也会将最后一个目录作为文件名分离，且不会判断文件或者目录是否存在 |
| splitext(path)              | 分离文件名与扩展名，返回(f_name, f_extension)元组      |
| getsize(file)               | 返回指定文件的尺寸，单位是字节                          |
| getatime(file)              | 返回指定文件最近的访问时间（浮点型秒数，可用time模块的gmtime()或localtime()函数换算） |
| getctime(file)              | 返回指定文件的创建时间（浮点型秒数，可用time模块的gmtime()或localtime()函数换算） |
| getmtime(file)              | 返回指定文件最新的修改时间（浮点型秒数，可用time模块的gmtime()或localtime()函数换算） |
| *以下为函数返回 True 或 False*      |                                          |
| exists(path)                | 判断指定路径（目录或文件）是否存在                        |
| isabs(path)                 | 判断指定路径是否为绝对路径                            |
| isdir(path)                 | 判断指定路径是否存在且是一个目录                         |
| isfile(path)                | 判断指定路径是否存在且是一个文件                         |
| islink(path)                | 判断指定路径是否存在且是一个符号链接                       |
| ismount(path)               | 判断指定路径是否存在且是一个挂载点                        |
| samefile(path1, paht2)      | 判断path1和path2两个路径是否指向同一个文件               |

### 目录文件的操作 os库
python安装后自带的函数库，处理操作系统相关功能  
os.getcwd() 获得当前工作目录  
os.listdir(path) 返回指定目录下的所有文件和目录名  
os.remove() 删除一个文件  
os.removedirs(path) 删除多个目录  
os.chdir(path) 更改当前目录到指定目录  
os.mkdir(path) 新建一个目录  
os.rmdir(name) 删除一个目录  
os.rename(src, dst) 更改文件名  
os.path 处理操作系统目录的一个子库  
Os.path.isfile() 检验路径是否是一个文件  
Os.path.isdir()检验路径是否是一个路径    
Os.path.exists() 判断路径是否存在  
Os.path.split() 返回一个路径的目录名和文件名  
os.path.splitext() 分离扩展名  
Os.path.dirname 获得路径名  
Os.path.basename() 获得文件名  
Os.path.getsize() 获得文件大小  
Os.path.join(path, name) 返回绝对路径  
os.walk(path)用于遍历一个目录，返回一个三元组
root, dirs, files = os.walk(path)  
其中，root是字符串，dirs和files是列表类型，表示root
中的所有目录和所有文件  

例子：打印某一目录下的全部文件
```
import os
path = input("输入一个路径:")
for root, dirs, files in os.walk(path):
    for name in files:
        print(os.path.join(root, name))
```
Os.walk会自顶向下依次遍历目录信息，以三元组形式输出  
例子：将文件夹下所有文件名字后增加一个字符串_py  
```
import os
path = input("输入一个路径：")
for root, dirs, files in os.walk(path):
    for name in files:
        fname,fext = os.path.splitxt(name)
        os.rename(os.path.join(root, name), \
        os.path.join(root, fname+'_py'+fext))
```
### 程序定时执行 sched库
sched库用来进行任务调度  
sched.scheduler()用来创建一个调度任务  
当需要对一个任务进行时间调度时，用这个库  
scheduler.enter(delay, priority, action, argument=())  
创建一个调度事件，argument中是action()的参数部分  
scheduler.run() 运行调度任务中的全部调度事件  
scheduler.cancel(event)取消某个调度事件 

例子：函数定时执行func_sched.py
```
import sched, time
def print_time(msg='default'):
    print("当前时间",time.time(), msg)
s = sched.scheduler(time.time, time.sleep)
print(time.time())
s.enter(5, 1, print_time, argument=('延迟5秒，优先级1',))
s.enter(3, 2, print_time, argument=('延迟3秒，优先级2',))
s.enter(3, 1, print_time, argument=('延迟3秒，优先级1',))
s.run()
print(time.time())
```
### 可执行程序的转换 py2exe库
问题：在windows平台下，使用exe文件执行，如何将
python程序变成exe程序，并在没有python环境的情况下
运行呢？  
步骤1：确定python程序可执行， func_sched.py  
步骤2：写一个发布脚本setup.py  
```
from distutils.core import setup
import py2exe
setup(console=['func_sched.py'])
```
步骤3：在windows命令行cmd下运行  
python setup.py py2exe  
步骤4：运行结果  
生成两个目录：dist和__pycache__  
其中，dist中包含了发布的exe程序  
__pycache__是过程文件，可以删除  
注意：目录dist需要整体拷贝到其他系统使用，因为，其
中包含了exe运行的依赖库，不能只拷贝exe文件  