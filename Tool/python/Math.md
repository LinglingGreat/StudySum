## math库中常用的数学函数
圆周率pi  π的近似值，15位小数  
自然常数e  e的近似值，15位小数  
ceil(x)  对浮点数向上取整  
floor(x)  对浮点数向下取整  
pow(x,y)  计算x的y次方  
log(x)  以e为基的对数，  
log10(x)  以10为基的对数，  
sqrt(x)  平方根    
exp(x) e的x次幂，  
degrees(x) 将弧度值转换成角度  
radians(x) 将角度值转换成弧度  
sin(x) sin x 正弦函数  
cos(x) cos x 余弦函数  
tan(x) tan x 正切函数  
asin(x) arcsin x 反正弦函数，x∈[-1.0,1.0]  
acos(x) arccos x 反余弦函数，x∈[-1.0,1.0]  
atan(x) arctan x 反正切函数，x∈[-1.0,1.0]  
## random库中常用的函数
seed(x) 给随机数一个种子值，默认随机种子是系
统时钟  
random() 生成一个[0, 1.0)之间的随机小数  
uniform(a,b) 生成一个a到b之间的随机小数  
randint(a,b) 生成一个a到b之间的随机整数  
randrange(a,b,c) 随机生成一个从a开始到b以c递增的数  
choice(<list&gt;) 从列表中随机返回一个元素  
shuffle(<list&gt;) 将列表中元素随机打乱  
sample(<list&gt;,k) 从指定列表随机获取k个元素  
## 用蒙特卡洛方法计算π
蒙特卡洛(Monte Carlo)方法，又称随机抽样或统计
试验方法。当所求解问题是某种事件出现的概率，或某
随机变量期望值时，可以通过某种“试验”的方法求解。

简单说，蒙特卡洛是利用随机试验求解问题的方法。  

首先构造一个单位正方形和1/4圆  
随机向单位正方形和圆结构抛洒大量点，对于每个点，
可能在圆内或者圆外，当随机抛点数量达到一定程度，
圆内点将构成圆的面积，全部抛点将构成矩形面积。圆
内点数除以圆外点数就是面积之比，即π/4。随机点数
量越大，得到的π值越精确。  

π计算问题的IPO表示如下：  
输入：抛点的数量  
处理：对于每个抛洒点，计算点到圆心的距离，通过距离判断该点在圆内或是圆外。统计在圆内点的数量  
输出：π值  

```
from random import random
from math import sqrt
from time import clock
# 增加DARTS数量，能够进一步增加精度
DARTS = 1200
hits = 0
clock()
# 代码主体是一个循环，模拟抛洒多个点的过程
# 对于一个抛点，通过random()函数给出随机的坐标值(x,y)，
# 然后利用开方函数sqrt()计算抛点到原点距离
# 然后通过if语句判断这个距离是否落在圆内
# 最终，根据总抛点落入圆内的数量，计算比值，从而得到π值
for i in range(1,DARTS):
    x, y = random(), random()
    dist = sqrt(x**2 + y**2)
    if dist <= 1.0:
        hits = hits + 1
pi = 4 * (hits/DARTS)
print("Pi的值是 %s" % pi)
print("程序运行时间是 %-5.5ss" % clock())
```
