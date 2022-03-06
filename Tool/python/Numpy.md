## 维度数据
维度：一组数据的组织形式  
>一维数据：由对等关系的有序或无序数据构成，采用线性方式组织   
>>对应列表、数组和集合等概念
>>列表和数组是一组数据的有序结构  
>>列表：数据类型可以不同  
>>数组：数据类型相同  
>>python的列表和集合类型

>二维数据：由多个一维数据构成，是一维数据的组合形式  
>>表格是典型的二维数据，其中，表头是二维数据的一部分  
>>python的列表类型

>多维数据：由一维或二维数据在新维度上扩展形成   
>>python的列表类型 

>高维数据：仅利用最基本的二元关系展示数据间的复杂结构 
>>python的字典类型或数据表示格式(JSON、XML和YAML格式)   
>>举例：键值对  
``` 
{
   “firstName” : “Tian” ,
   “lastName” : “Song” ,
   “address” : {
                  “streetAddr” : “中关村南大街5号” ,
                  “city” : “北京市” ,
                  “zipcode” : “100081”
                } ,
   “prof” : [ “Computer System” , “Security” ]
}
```
## NumPy

NumPy是一个开源的Python科学计算基础库，包含：

• 一个强大的N维数组对象ndarray  
• 广播功能函数  
• 整合C/C++/Fortran代码的工具  
• 线性代数、傅里叶变换、随机数生成等功能

NumPy是SciPy、Pandas等数据处理或科学计算库的基础

### N维数组对象：ndarray
```
a = np.array([0,1,2,3,4],
             [9,8,7,6,5])
```
>ndarray对象的属性

.ndim: 秩，即轴的数量或维度的数量     
.shape: ndarray对象的尺度，对于矩阵，n行m列   
.size: ndarray对象元素的个数，相当于.shape中n*m的值   
.dtype: ndarray对象的元素类型   
.itemsize: ndarray对象中每个元素的大小，以字节为单位   

>ndarray的元素类型  

bool 布尔类型，True或False  
intc 与C语言中的int类型一致，一般是int32或int64  
intp 用于索引的整数，与C语言中ssize_t一致，int32或int64  
int8 字节长度的整数，取值：[‐128, 127]  
int16 16位长度的整数，取值：[‐32768, 32767]  
int32 32位长度的整数，取值：[‐231, 231‐1]  
int64 64位长度的整数，取值：[‐263, 263‐1] 
uint8 8位无符号整数，取值：[0, 255]  
uint16 16位无符号整数，取值：[0, 65535]  
uint32 32位无符号整数，取值：[0, 232‐1]  
uint64 32位无符号整数，取值：[0, 264‐1]  
((符号)尾数*10^指数)   
float16 16位半精度浮点数：1位符号位，5位指数，10位尾数  
float32 32位半精度浮点数：1位符号位，8位指数，23位尾数  
float64 64位半精度浮点数：1位符号位，11位指数，52位尾数  
实部(.real) + j虚部(.imag)   
complex64 复数类型，实部和虚部都是32位浮点数  
complex128 复数类型，实部和虚部都是64位浮点数   

>ndarray数组的创建方法  

(1)从Python中的列表、元组等类型创建ndarray数组  
```
x = np.array(list/tuple)
x = np.array(list/tuple, dtype=np.float32)

x = np.array([0,1,2,3])
x = np.array((4,5,6,7))
x = np.array([[1,2],[9,8],(0.1,0.2)])
```
(2)使用NumPy中函数创建ndarray数组，如：arange,ones,zeros等  

np.arange(n) 类似range()函数，返回ndarray类型，元素从0到n‐1  
np.ones(shape) 根据shape生成一个全1数组，shape是元组类型  
np.zeros(shape) 根据shape生成一个全0数组，shape是元组类型  
np.full(shape,val) 根据shape生成一个数组，每个元素值都是val  
np.eye(n) 创建一个正方的n*n单位矩阵，对角线为1，其余为0  
np.ones_like(a) 根据数组a的形状生成一个全1数组  
np.zeros_like(a) 根据数组a的形状生成一个全0数组  
np.full_like(a,val) 根据数组a的形状生成一个数组，每个元素值都是val  
```
x = np.ones((2,3,4))
print(x)
[[[1. 1. 1. 1.]
  [1. 1. 1. 1.]
  [1. 1. 1. 1.]]
  
 [[1. 1. 1. 1.]
  [1. 1. 1. 1.]
  [1. 1. 1. 1.]]]
x.shape
(2,3,4)
```
(3)使用NumPy中其他函数创建ndarray数组

np.linspace() 根据起止数据等间距地填充数据，形成数组  
np.concatenate() 将两个或多个数组合并成一个新的数组  

```
a = np.linspace(1,10,4)
a
array([1., 4., 7., 10.])
b = np.linspace(1,10,4,endpoint=False)
b
array([1., 3.25, 5.5, 7.75])
c = np.concatenate((a,b))
c
array([1., 4., 7., 10., 1., 3.25, 5.5, 7.75])
```
(4)从字节流(raw bytes)中创建ndarray数组  
(5)从文件中读取特定格式，创建ndarray数组  

>ndarray数组的变换

.reshape(shape) 不改变数组元素，返回一个shape形状的数组，原数组不变  a.reshape((3,8))
.resize(shape) 与.reshape()功能一致，但修改原数组  
.swapaxes(ax1,ax2) 将数组n个维度中两个维度进行调换  
.flatten() 对数组进行降维，返回折叠后的一维数组，原数组不变  

ndarray数组的类型变换:new_a = a.astype(new_type)  b = a.astype(np.float)  
astype()方法一定会创建新的数组(原始数据的一个拷贝)，即使两个类型一致

ndarray数组向列表的转换：ls = a.tolist()  

>ndarray数组的操作

>>数组的索引和切片  
一维数组的的索引和切片：与Python的列表类似  
a[2], a[1:4:2](起始编号：终止编号(不含)：步长)  
多维数组的索引：a[1,2,3]每个维度一个索引值，逗号分隔  
多维数组的切片:

```
a = np.arange(24) .reshape((2,3,4))
a
array([[[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]],
       [[12, 13, 14, 15],
        [16, 17, 18, 19],
        [20, 21, 22, 23]]])
a[:, 1, -3] #选取一个维度用： 
array([ 5, 17]) 
a[:, 1:3, :] #每个维度切片方法与一维数组相同  
array([[[ 4,  5,  6,  7],
        [ 8,  9, 10, 11]],
       [[16, 17, 18, 19],
        [20, 21, 22, 23]]])
a[:, :, ::2]每个维度可以使用步长跳跃切片
array([[[ 0,  2],
        [ 4,  6],
        [ 8, 10]],
       [[12, 14],
        [16, 18],
        [20, 22]]])
```

>ndarray数组的运算  
数组与标量之间的运算作用于数组的每一个元素

NumPy一元函数：对ndarray中的数据执行元素级运算的函数

np.abs(x) np.fabs(x) 计算数组各元素的绝对值  
np.sqrt(x) 计算数组各元素的平方根  
np.square(x) 计算数组各元素的平方  
np.log(x) np.log10(x) np.log2(x) 计算数组各元素的自然对数、10底对数和2底对数  
np.ceil(x) np.floor(x) 计算数组各元素的ceiling值或floor值  
np.rint(x) 计算数组各元素的四舍五入值  
np.modf(x) 将数组各元素的小数和整数部分以两个独立数组形式返回  
np.cos(x) np.cosh(x) np.sin(x) np.sinh(x)
np.tan(x) np.tanh(x)计算数组各元素的普通型和双曲型三角函数  
np.exp(x) 计算数组各元素的指数值  
np.sign(x) 计算数组各元素的符号值，1(+), 0, ‐1(‐)

>NumPy二元函数

+ ‐ * / ** 两个数组各元素进行对应运算  
np.maximum(x,y) np.fmax() np.minimum(x,y) np.fmin() 元素级的最大值/最小值计算  
np.mod(x,y) 元素级的模运算  
np.copysign(x,y) 将数组y中各元素值的符号赋值给数组x对应元素  
\> < >= <= == != 算术比较，产生布尔型数组

### 数据存取与函数
>数据的CSV文件存取  

np.savetxt(frame, array, fmt='%.18e', delimiter=None)  
• frame : 文件、字符串或产生器，可以是.gz或.bz2的压缩文件  
• array : 存入文件的数组  
• fmt : 写入文件的格式，例如：%d %.2f %.18e  
• delimiter : 分割字符串，默认是任何空格 
```
a = np.arange(100).reshape(5, 20) 
np.savetxt('a.csv', a, fmt='%d', delimiter=',')
```
np.loadtxt(frame, dtype=np.float, delimiter=None， unpack=False)  
• frame : 文件、字符串或产生器，可以是.gz或.bz2的压缩文件  
• dtype : 数据类型，可选  
• delimiter : 分割字符串，默认是任何空格  
• unpack : 如果True，读入属性将分别写入不同变量
```
b = np.loadtxt('a.csv', dtype=np.int, delimiter=',')
```
CSV只能有效存储一维和二维数组  
np.savetxt() np.loadtxt()只能有效存取一维和二维数组
>多维数据的存取

a.tofile(frame, sep='', format='%s')  
• frame : 文件、字符串  
• sep : 数据分割字符串，如果是空串，写入文件为二进制  
• format : 写入数据的格式  
```
a = np.arange(100).reshape(5, 10, 2)
a.tofile("b.dat", sep=",", format='%d')
```
np.fromfile(frame, dtype=float, count=‐1, sep='')  
• frame : 文件、字符串  
• dtype : 读取的数据类型  
• count : 读入元素个数，‐1表示读入整个文件  
• sep : 数据分割字符串，如果是空串，写入文件为二进制  
```
a = np.arange(100).reshape(5,10,2)
a.tofile("b.dat", sep=",", format='%d')
c = np.fromfile("b.dat", dtype=np.int,sep=",")
c
array([0, 1, 2, ..., 97, 98, 99])
c = np.fromfile("b.dat", dtype=np.int, sep=",").reshape(5,10,2)
```
该方法需要读取时知道存入文件时数组的维度和元素类型  
a.tofile()和np.fromfile()需要配合使用  
可以通过元数据文件来存储额外信息

>NumPy的便捷文件存取

np.save(fname, array) 或np.savez(fname, array)  
• fname : 文件名，以.npy为扩展名，压缩扩展名为.npz  
• array : 数组变量 

np.load(fname)  
• fname : 文件名，以.npy为扩展名，压缩扩展名为.npz  
```
a = np.arange(100).reshape(5,10,2)
np.save("a.npy",a)
b = np.load("a.npy")
```
### NumPy的随机数函数

NumPy的随机数函数子库np.random.*  
np.random.rand(), np.random.randn(), np.random.randint()

np.random的随机数函数

rand(d0,d1,..,dn) 根据d0‐dn创建随机数数组，浮点数，[0,1)，均匀分布  
randn(d0,d1,..,dn) 根据d0‐dn创建随机数数组，标准正态分布  
randint(low[,high,shape]) 根据shape创建随机整数或整数数组，范围是[low, high)  
seed(s) 随机数种子，s是给定的种子值
shuffle(a) 根据数组a的第1轴进行随排列，改变数组x
permutation(a) 根据数组a的第1轴产生一个新的乱序数组，不改变数组x  
choice(a[,size,replace,p]) 从一维数组a中以概率p抽取元素，形成size形状新数组,replace表示是否可以重用元素，默认为False  
uniform(low,high,size) 产生具有均匀分布的数组,low起始值,high结束值,size形状  
normal(loc,scale,size) 产生具有正态分布的数组,loc均值,scale标准差,size形状  
poisson(lam,size) 产生具有泊松分布的数组,lam随机事件发生率,size形状  
```
a = np.random.rand(3,4,5)
sn = np.random.randn(3,4,5)
b = np.random.randint(100,200,(3,4))
np.random.seed(10)
np.random.randint(100,200,(3,4))

b = np.random.randint(100,200,(8,))
np.random.choice(b,(3,2))
np.random.choice(b,(3,2), replace=False)
np.random.choice(b,(3,2), p=b/np.sum(b))

u = np.random.uniform(0, 10, (3,4))
n = np.random.normal(10, 5, (3,4))
```

### NumPy的统计函数

Numpy直接提供的统计类函数：np.*

sum(a, axis=None) 根据给定轴axis计算数组a相关元素之和，axis整数或元组  
mean(a, axis=None) 根据给定轴axis计算数组a相关元素的期望，axis整数或元组  
average(a,axis=None,weights=None) 根据给定轴axis计算数组a相关元素的加权平均值  
std(a, axis=None) 根据给定轴axis计算数组a相关元素的标准差  
var(a, axis=None) 根据给定轴axis计算数组a相关元素的方差  
axis=None是统计函数的标配参数  

min(a) max(a) 计算数组a中元素的最小值、最大值  
argmin(a) argmax(a) 计算数组a中元素最小值、最大值的降一维后下标  
unravel_index(index, shape) 根据shape将一维下标index转换成多维下标  
ptp(a) 计算数组a中元素最大值与最小值的差  
median(a) 计算数组a中元素的中位数（中值）  

```
a = np.arange(15).reshape(3,5)
np.average(a, axis=0, weights=[10, 5, 1])

b = np.arange(15,0,-1).reshape(3,5)
np.argmax(b)  # 扁平化后的下表
np.unravel_index(np.argmax(b), b.shape) # 重塑成多维下标
```

### NumPy的梯度函数

np.gradient(f) 计算数组f中元素的梯度，当f为多维时，返回每个维度梯度  
梯度：连续值之间的变化率，即斜率  
XY坐标轴连续三个X坐标对应的Y轴值：a, b, c，其中，b的梯度是： (c‐a)/2  
```
a = np.random.randint(0,20,(5))
a
array([15, 3, 12, 13, 14])
np.gradient(a)
array([12., -1.5, 5., 1., 1.])
# -1.5=(12-15)/2存在两侧值, 1.=(14-13)/1只有一侧值
c = np.random.randint(0,50,(3,5))
np.gradient(c)
```