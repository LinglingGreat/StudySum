## 魔法方法

| **魔法方法**                                 | **含义**                                   |
| ---------------------------------------- | ---------------------------------------- |
|                                          | **基本的魔法方法**                              |
| \__new__(cls[, ...])                     | 1. __new__ 是在一个对象实例化的时候所调用的第一个方法2. 它的第一个参数是这个类，其他的参数是用来直接传递给 __init__ 方法3. __new__ 决定是否要使用该 __init__ 方法，因为 __new__ 可以调用其他类的构造方法或者直接返回别的实例对象来作为本类的实例，如果 __new__ 没有返回实例对象，则 __init__ 不会被调用4. __new__ 主要是用于继承一个不可变的类型比如一个 tuple 或者 string |
| \__init__(self[, ...])                   | 构造器，当一个实例被创建的时候调用的初始化方法                  |
| \__del__(self)                           | 析构器，当一个实例被销毁的时候调用的方法                     |
| \__call__(self[, args...])               | 允许一个类的实例像函数一样被调用：x(a, b) 调用 x.__call__(a, b) |
| \__len__(self)                           | 定义当被 len() 调用时的行为                        |
| \__repr__(self)                          | 定义当被 repr() 调用时的行为                       |
| \__str__(self)                           | 定义当被 str() 调用时的行为                        |
| \__bytes__(self)                         | 定义当被 bytes() 调用时的行为                      |
| \__hash__(self)                          | 定义当被 hash() 调用时的行为                       |
| \__bool__(self)                          | 定义当被 bool() 调用时的行为，应该返回 True 或 False     |
| \__format__(self, format_spec)           | 定义当被 format() 调用时的行为                     |
|                                          | **有关属性**                                 |
| \__getattr__(self, name)                 | 定义当用户试图获取一个不存在的属性时的行为                    |
| \__getattribute__(self, name)            | 定义当该类的属性被访问时的行为                          |
| \__setattr__(self, name, value)          | 定义当一个属性被设置时的行为                           |
| \__delattr__(self, name)                 | 定义当一个属性被删除时的行为                           |
| \__dir__(self)                           | 定义当 dir() 被调用时的行为                        |
| \__get__(self, instance, owner)          | 定义当描述符的值被取得时的行为                          |
| \__set__(self, instance, value)          | 定义当描述符的值被改变时的行为                          |
| \__delete__(self, instance)              | 定义当描述符的值被删除时的行为                          |
|                                          | **比较操作符**                                |
| \__lt__(self, other)                     | 定义小于号的行为：x < y 调用 x.__lt__(y)            |
| \__le__(self, other)                     | 定义小于等于号的行为：x <= y 调用 x.__le__(y)         |
| \__eq__(self, other)                     | 定义等于号的行为：x == y 调用 x.__eq__(y)           |
| \__ne__(self, other)                     | 定义不等号的行为：x != y 调用 x.__ne__(y)           |
| \__gt__(self, other)                     | 定义大于号的行为：x > y 调用 x.__gt__(y)            |
| \__ge__(self, other)                     | 定义大于等于号的行为：x >= y 调用 x.__ge__(y)         |
|                                          | **算数运算符**                                |
| \__add__(self, other)                    | 定义加法的行为：+                                |
| \__sub__(self, other)                    | 定义减法的行为：-                                |
| \__mul__(self, other)                    | 定义乘法的行为：*                                |
| \__truediv__(self, other)                | 定义真除法的行为：/                               |
| \__floordiv__(self, other)               | 定义整数除法的行为：//                             |
| \__mod__(self, other)                    | 定义取模算法的行为：%                              |
| \__divmod__(self, other)                 | 定义当被 divmod() 调用时的行为                     |
| \__pow__(self, other[, modulo])          | 定义当被 power() 调用或 ** 运算时的行为               |
| \__lshift__(self, other)                 | 定义按位左移位的行为：<<                            |
| \__rshift__(self, other)                 | 定义按位右移位的行为：>>                            |
| \__and__(self, other)                    | 定义按位与操作的行为：&                             |
| \__xor__(self, other)                    | 定义按位异或操作的行为：^                            |
| \__or__(self, other)                     | 定义按位或操作的行为：\|                            |
|                                          | **反运算**                                  |
| \__radd__(self, other)                   | （与上方相同，当左操作数不支持相应的操作时被调用）                |
| \__rsub__(self, other)                   | （与上方相同，当左操作数不支持相应的操作时被调用）                |
| \__rmul__(self, other)                   | （与上方相同，当左操作数不支持相应的操作时被调用）                |
| \__rtruediv__(self, other)               | （与上方相同，当左操作数不支持相应的操作时被调用）                |
| \__rfloordiv__(self, other)              | （与上方相同，当左操作数不支持相应的操作时被调用）                |
| \__rmod__(self, other)                   | （与上方相同，当左操作数不支持相应的操作时被调用）                |
| \__rdivmod__(self, other)                | （与上方相同，当左操作数不支持相应的操作时被调用）                |
| \__rpow__(self, other)                   | （与上方相同，当左操作数不支持相应的操作时被调用）                |
| \__rlshift__(self, other)                | （与上方相同，当左操作数不支持相应的操作时被调用）                |
| \__rrshift__(self, other)                | （与上方相同，当左操作数不支持相应的操作时被调用）                |
| \__rand__(self, other)                   | （与上方相同，当左操作数不支持相应的操作时被调用）                |
| \__rxor__(self, other)                   | （与上方相同，当左操作数不支持相应的操作时被调用）                |
| \__ror__(self, other)                    | （与上方相同，当左操作数不支持相应的操作时被调用）                |
|                                          | **增量赋值运算**                               |
| \__iadd__(self, other)                   | 定义赋值加法的行为：+=                             |
| \__isub__(self, other)                   | 定义赋值减法的行为：-=                             |
| \__imul__(self, other)                   | 定义赋值乘法的行为：*=                             |
| \__itruediv__(self, other)               | 定义赋值真除法的行为：/=                            |
| \__ifloordiv__(self, other)              | 定义赋值整数除法的行为：//=                          |
| \__imod__(self, other)                   | 定义赋值取模算法的行为：%=                           |
| \__ipow__(self, other[, modulo])         | 定义赋值幂运算的行为：**=                           |
| \__ilshift__(self, other)                | 定义赋值按位左移位的行为：<<=                         |
| \__irshift__(self, other)                | 定义赋值按位右移位的行为：>>=                         |
| \__iand__(self, other)                   | 定义赋值按位与操作的行为：&=                          |
| \__ixor__(self, other)                   | 定义赋值按位异或操作的行为：^=                         |
| \__ior__(self, other)                    | 定义赋值按位或操作的行为：\|=                         |
|                                          | **一元操作符**                                |
| \__pos__(self)                           | 定义正号的行为：+x                               |
| \__neg__(self)                           | 定义负号的行为：-x                               |
| \__abs__(self)                           | 定义当被 abs() 调用时的行为                        |
| \__invert__(self)                        | 定义按位求反的行为：~x                             |
|                                          | **类型转换**                                 |
| \__complex__(self)                       | 定义当被 complex() 调用时的行为（需要返回恰当的值）          |
| \__int__(self)                           | 定义当被 int() 调用时的行为（需要返回恰当的值）              |
| \__float__(self)                         | 定义当被 float() 调用时的行为（需要返回恰当的值）            |
| \__round__(self[, n])                    | 定义当被 round() 调用时的行为（需要返回恰当的值）            |
| \__index__(self)                         | 1. 当对象是被应用在切片表达式中时，实现整形强制转换2. 如果你定义了一个可能在切片时用到的定制的数值型,你应该定义 __index__3. 如果 __index__ 被定义，则 __int__ 也需要被定义，且返回相同的值 |
|                                          | **上下文管理（with 语句）**                       |
| \__enter__(self)                         | 1. 定义当使用 with 语句时的初始化行为2. __enter__ 的返回值被 with 语句的目标或者 as 后的名字绑定 |
| \__exit__(self, exc_type, exc_value, traceback) | 1. 定义当一个代码块被执行或者终止后上下文管理器应该做什么2. 一般被用来处理异常，清除工作或者做一些代码块执行完毕之后的日常工作 |
|                                          | **容器类型**                                 |
| \__len__(self)                           | 定义当被 len() 调用时的行为（返回容器中元素的个数）            |
| \__getitem__(self, key)                  | 定义获取容器中指定元素的行为，相当于 self[key]             |
| \__setitem__(self, key, value)           | 定义设置容器中指定元素的行为，相当于 self[key] = value     |
| \__delitem__(self, key)                  | 定义删除容器中指定元素的行为，相当于 del self[key]         |
| \__iter__(self)                          | 定义当迭代容器中的元素的行为                           |
| \__reversed__(self)                      | 定义当被 reversed() 调用时的行为                   |
| \__contains__(self, item)                | 定义当使用成员测试运算符（in 或 not in）时的行为            |

## Python Mixin编程机制

**Mixin 简介**

Mixin 编程是一种开发模式，是一种将多个类中的功能单元的进行组合的利用的方式，这听起来就像是有类的继承机制就可以实现，然而这与传统的类继承有所不同。通常 Mixin 并不作为任何类的基类，也不关心与什么类一起使用，而是在运行时动态的同其他零散的类一起组合使用。

**特点**

使用 Mixin 机制有如下好处：

- 可以在不修改任何源代码的情况下，对已有类进行扩展；
- 可以保证组件的划分；
- 可以根据需要，使用已有的功能进行组合，来实现“新”类；
- 很好的避免了类继承的局限性，因为新的业务需要可能就需要创建新的子类。

**多继承**

Python支持多继承，即一个类可以继承多个子类。可以利用该特性，可以方便的实现mixin继承。如下代码，类A,B分别表示不同的功能单元，C为A,B功能的组合，这样类C就拥有了类A, B的功能。

```python
class A:
    def get_a(self):
    print 'a'

class B:
    def get_b(self):
    print 'b'

class C(A, B): 
    pass

c = C()
c.get_a()
c.get_b()
```

**\__bases\__**

多继承的实现就会创建新类，有时，我们在运行时，希望给类A添加类B的功能时，也可以利用python的元编程特性，\__bases__属性便在运行时轻松给类A添加类B的特性，如下代码：

```python
A.__bases__ += (B,)
a.get_b()
```

其实\__bases\__也是继承的机制，因为\__bases\__属性存储了类的基类。因此多继承的方法也可以这样实现：

```python
class C:
    pass

C.__bases__ += (A, B, )
```

**插件方式**

以上两种方式，都是基于多继承和python的元编程特性，然而在业务需求变化时，就需要新的功能组合，那么就需要重新修改A的基类，这回带来同步的问题，因为我们改的是类的特性，而不是对象的。因此以上修改会对所有引用该类的模块都收到影响，这是相当危险的。通常我们希望修改对象的行为，而不是修改类的。同样的我们可以利用__dict__来扩展对象的方法。

```
class PlugIn(object):
    def __init__(self):
        self._exported_methods = []
        
    def plugin(self, owner):
        for f in self._exported_methods:
            owner.__dict__[f.__name__] = f

    def plugout(self, owner):
        for f in self._exported_methods:
            del owner.__dict__[f.__name__]

class AFeature(PlugIn):
    def __init__(self):
        super(AFeature, self).__init__()
        self._exported_methods.append(self.get_a_value)

    def get_a_value(self):
        print 'a feature.'

class BFeature(PlugIn):
    def __init__(self):
        super(BFeature, self).__init__()
        self._exported_methods.append(self.get_b_value)

    def get_b_value(self):
        print 'b feature.'

class Combine:pass

c = Combine()
AFeature().plugin(c)
BFeature().plugin(c)

c.get_a_value()
c.get_b_value()
```

## property的详细使用方法

Property的原理（描述符）

描述符：将某种特殊类型的类的实例指派给另一个类的属性

特殊类型需实现下列方法：

\__get__(self, instance, owner)：用于访问属性，它返回属性的值
\__set__(self, instance, value)：将在属性分配操作中调用，不返回任何内容
\__delete__(self, instance)：控制删除操作，不返回任何内容

```python
class MyDecriptor:
	def __get__(self, instance, owner):
		print('getting...', self, instance, owner)
	def __set__(self, instance, value):
		print('setting...', self, instance, value)
	def __delete__(self, instance):
		print('deleting...', self, instance)
class Test:
	x = MyDecriptor()

>>>text = Test()
>>> text.x
getting... <__main__.MyDecriptor object at 0x00000000035FB438> <__main__.Test object at 0x000000000350B908> <class '__main__.Test'>
>>> text.x = 'X-man'
setting... <__main__.MyDecriptor object at 0x00000000035FB438> <__main__.Test object at 0x000000000350B908> X-man
>>> del text.x
deleting... <__main__.MyDecriptor object at 0x00000000035FB438> <__main__.Test object at 0x000000000350B908>
```

```python
class MyProperty:
	def __init__(self, fget=None, fset=None, fdel=None):
		self.fget = fget
		self.fset = fset
		self.fdel = fdel
	def __get__(self, instance, owner):
		return self.fget(instance)
	def __set__(self, instance, value):
		self.fset(instance, value)
	def __del__(self, instance):
		self.fdel(instance)
 class C:
	def __init__(self):
		self._x = None
	def getX(self):
		return self._x
	def setX(self, value):
		self._x = value
	def delX(self):
		del self._x
	x = MyProperty(getX, setX, delX)
>>> c = C()
>>> c.x = 'X-man'
>>> c.x
'X-man'
>>> c._x
'X-man'
```

```python
class Celsius:
    def __init__(self, value = 26):
        self.value = float(value)

    def __get__(self, instance, owner):
        return self.value

    def __set__(self, instance, value):
        self.value = float(value)

class Fahrenheit:
    def __get__(self, instance, owner):
        return instance.cel * 1.8 + 32
    def __set__(self, instance, value):
        instance.cel = (float(value) - 32) / 1.8

class Temperature:
    cel = Celsius()
    fah = Fahrenheit()

>>> temp = Temperature()
>>> temp.cel
26.0
>>> temp.cel = 30
>>> temp.fah
86.0
>>> temp.fah = 100
>>> temp.cel
37.77777777777778
```



**property(fget=None, fset=None, fdel=None, doc=None) **

俗话说条条大路通罗马，同样是完成一件事，Python 其实提供了好几个方式供你选择。

property() 是一个比较奇葩的BIF，它的作用把方法当作属性来访问，从而提供更加友好访问方式。

```python
class C:
    def __init__(self):
        self._x = None

    def getx(self):
        return self._x
    def setx(self, value):
        self._x = value
    def delx(self):
        del self._x
    x = property(getx, setx, delx, "I'm the 'x' property.")
```

property() 返回一个可以设置属性的属性，当然如何设置属性还是需要我们人为来写代码。第一个参数是获得属性的方法名（例子中是 getx），第二个参数是设置属性的方法名（例子中是 setx），第三个参数是删除属性的方法名（例子中是 delx）。

property() 有什么作用呢？举个例子，在上边的例题中，我们为用户提供 setx 方法名来设置 _x 属性，提供 getx 方法名来获取属性。但是有一天你心血来潮，突然想对程序进行大改，可能你需要把 setx 和 getx 修改为 set_x 和 get_x，那你不得不修改用户调用的接口，这样子的体验就非常不好。

有了 property() 所有问题就迎刃而解了，因为像上边一样，我们为用户访问 _x 属性只提供了 x 属性。无论我们内部怎么改动，只需要相应的修改 property() 的参数，用户仍然只需要去操作 x 属性即可，对他们来说没有任何影响。


**使用属性修饰符创建描述符**

使用属性修饰符创建描述符，也可以实现同样的功能（【扩展阅读】[Python 函数修饰符（装饰器）的使用](http://bbs.fishc.com/thread-51109-1-2.html)）：

```python
class C:
    def __init__(self):
        self._x = None

    @property
    def x(self):
        """I'm the 'x' property."""
        return self._x

    @x.setter
    def x(self, value):
        self._x = value

    @x.deleter
    def x(self):
        del self._x
```

注意：三个处理 _x 属性的方法名要相同（参数不同）。

**1.  修饰符的来源**

借用一个博客上的一段叙述：修饰符是一个很著名的设计模式，经常被用于有切面需求的场景，较为经典的有插入日志、性能测试、事务处理等。

修饰符是解决这类问题的绝佳设计，有了修饰符，我们就可以抽离出大量函数中与函数功能本身无关的雷同代码并继续重用。概括的讲，修饰符的作用就是为已经存在的对象添加额外的功能。

如下：

```python
import time

def timeslong(func):
    start = time.clock()
    print("It's time starting ! ")
    func()
    print("It's time ending ! ")
    end = time.clock()
    return "It's used : %s ." % (end - start)
```

上面的程序中，定义了一个函数，对另外一个对象的运行时间进行计算，如果采用通常的方式需要将 func() 重新在 timeslong 中重新写一遍。为了简化这种操作，便提出了修饰符的概念。

如下：

```python
import time

def timeslong(func):
    def call():
        start = time.clock()
        print("It's time starting ! ")
        func()
        print("It's time ending ! ")
        end = time.clock()
        return "It's used : %s ." % (end - start)
    return call

@timeslong
def f():
    y = 0
    for i in range(10):
        y = y + i + 1
        print(y)
    return y

print(f())
```

这样出现便不用再函数内部再进行嵌入函数，通过 @timeslong 对其进行修饰达到目的，是整个代码美观，而且少些部分代码。

修饰符也可以通过类来进行使用，共享该类，如下为一个实例：

```python
class timeslong(object):
def __init__(self,func):
    self.f = func
def __call__(self):
    start = time.clock()
    print("It's time starting ! ")
    self.f()
    print("It's time ending ! ")
    end = time.clock()
    return "It's used : %s ." % (end - start)

@timeslong
def f():
    y = 0
    for i in range(10):
        y = y + i + 1
        print(y)
    return y

print(f())
```

**2.  Python内置的修饰符**

内置的修饰符有三个，分别是 staticmethod、classmethod 和 property，作用分别是把类中定义的实例方法变成静态方法、类方法和类属性。由于模块里可以定义函数，所以静态方法和类方法的用处并不是太多。

```python
class Hello(object):
    def __init__:
        ...

@classmethod
def print_hello(cls):
    print("Hello")
```

classmethod 修饰过后，print_hello() 就变成了类方法，可以直接通过 Hello.print_hello() 调用，而无需绑定实例对象了。

## 如何使用静态方法、类方法或者抽象方法

**Python中方法的运作**

方法是作为类的属性（attribute）存储的函数，你可以以下面的方式声明和获取函数：

```python
>>> class Pizza(object):
...     def __init__(self, size):
...         self.size = size
...     def get_size(self):
...         return self.size
...
>>> Pizza.get_size
<unbound method Pizza.get_size>
```

Python告诉你的是，类Pizza的属性get_size是一个非绑定的方法。

这又指什么呢？很快我们就会知道，试着调用一下：

```python
>>> Pizza.get_size()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: unbound method get_size() must be called with Pizza instance as first argument (got nothing instead)
```

这里我们不能调用这个方法是因为它没有被绑定到任一 Pizza 的实例上。一个方法需要一个实例作为它第一个参数（在 Python2 中它必须是对应类的实例；在 Python3 中可以是任何东西）。

我们现在试试：

```python
>>> Pizza.get_size(Pizza(42))
42
```

现在可以了！我们试用一个实例作为 get_size 方法的第一个参数调用了它，所以一切变得很美好。

但是你很快会同意，这并不是一个很漂亮的调用方法的方式；因为每次我们想调用这个方法都必须使用到类。并且，如果我们不知道对象是哪个类的实例，这种方式就不方便了。所以，Python 为我们准备的是，它将类 Pizza 的所有的方法绑定到此类的任何实例上。

这意味着类 Pizza 的任意实例的属性 get_size 是一个已绑定的方法：第一个参数是实例本身的方法。

```python
>>> Pizza(42).get_size
<bound method Pizza.get_size of <__main__.Pizza object at 0x10314b310>>
>>> Pizza(42).get_size()
42
```

如我们预期，现在不需要提供任何参数给 get_size，因为它已经被绑定（bound），它的 self 参数是自动地设为 Pizza 类的实例。

下面是一个更好的证明：

```python
>>> m = Pizza(42).get_size
>>> m
<bound method Pizza.get_size of <__main__.Pizza object at 0x10314b350>>
>>> m()
42
```

因此，你甚至不要保存一个对 Pizza 对象的饮用。它的方法已经被绑定在对象上，所以这个方法已经足够。

但是如何知道已绑定的方法被绑定在哪个对象上？技巧如下：

```python
>>> m = Pizza(42).get_size
>>> m.__self__
<__main__.Pizza object at 0x10314b390>
>>> m == m.__self__.get_size
True
```

易见，我们仍然保存着一个对对象的引用，当需要知道时也可以找到。

在 Python3 中，归属于一个类的函数不再被看成未绑定方法（unbound method），但是作为一个简单的函数，如果要求可以绑定在对象上。所以，在 Python3 中原理是一样的，模型被简化了。

```python
>>> class Pizza(object):
...     def __init__(self, size):
...         self.size = size
...     def get_size(self):
...         return self.size
...
>>> Pizza.get_size
<function Pizza.get_size at 0x7f307f984dd0>
```

**静态方法**

静态方法是一类特殊的方法。有时，我们需要写属于一个类的方法，但是不需要用到对象本身。例如：

```python
class Pizza(object):
    @staticmethod
    def mix_ingredients(x, y):
        return x + y

    def cook(self):
        return self.mix_ingredients(self.cheese, self.vegetables)
```

这里，将方法 mix_ingredients 作为一个非静态的方法也可以 work，但是给它一个 self 的参数将没有任何作用。

这儿的 decorator@staticmethod 带来一些特别的东西：

```python
>>> Pizza().cook is Pizza().cook
False
>>> Pizza().mix_ingredients is Pizza().mix_ingredients
True
>>> Pizza().mix_ingredients is Pizza.mix_ingredients
True
>>> Pizza()
<__main__.Pizza object at 0x10314b410>
>>> Pizza()
<__main__.Pizza object at 0x10314b510>
```

Python 不需要对每个实例化的 Pizza 对象实例化一个绑定的方法。

绑定的方法同样是对象，创建它们需要付出代价。这里的静态方法避免了这样的情况：

- 降低了阅读代码的难度：看到 @staticmethod 便知道这个方法不依赖与对象本身的状态；
- 允许我们在子类中重载mix_ingredients方法。如果我们使用在模块最顶层定义的函数 mix_ingredients，一个继承自 Pizza 的类若不重载 cook，可能不可以改变混合成份（mix_ingredients）的方式。

**类方法**

什么是类方法？类方法是绑定在类而非对象上的方法！

```python
>>> class Pizza(object):
...     radius = 42
...     @classmethod
...     def get_radius(cls):
...         return cls.radius
... 
>>> Pizza.get_radius
<bound method type.get_radius of <class '__main__.Pizza'>>
>>> Pizza().get_radius
<bound method type.get_radius of <class '__main__.Pizza'>>
>>> Pizza.get_radius is Pizza().get_radius
False
>>> Pizza.get_radius()
42
```

不管你如何使用这个方法，它总会被绑定在其归属的类上，同时它第一个参数是类本身（记住：类同样是对象）

何时使用这种方法？类方法一般用于下面两种：

1. 工厂方法，被用来创建一个类的实例，完成一些预处理工作。如果我们使用一个 @staticmethod 静态方法，我们可能需要在函数中硬编码 Pizza 类的名称，使得任何继承自 Pizza 类的类不能使用我们的工厂用作自己的目的。

   ```python
   class Pizza(object):
       def __init__(self, ingredients):
           self.ingredients = ingredients

       @classmethod
       def from_fridge(cls, fridge):
           return cls(fridge.get_cheese() + fridge.get_vegetables())
   ```


2. 静态方法调静态方法：如果你将一个静态方法分解为几个静态方法，你不需要硬编码类名但可以使用类方法。使用这种方式来声明我们的方法，Pizza这个名字不需要直接被引用，并且继承和方法重载将会完美运作。

   ```python
   class Pizza(object):
   def __init__(self, radius, height):
        self.radius = radius
        self.height = height

   @staticmethod
   def compute_circumference(radius):
         return math.pi * (radius ** 2)

   @classmethod
   def compute_volume(cls, height, radius):
         return height * cls.compute_circumference(radius)

   def get_volume(self):
        return self.compute_volume(self.height, self.radius)
   ```

**抽象方法**

抽象方法在一个基类中定义，但是可能不会有任何的实现。在 Java 中，这被描述为一个接口的方法。

所以Python中最简单的抽象方法是：

```python
class Pizza(object):
def get_radius(self):
     raise NotImplementedError
```

任何继承自 Pizza 的类将实现和重载 get_radius 方法，否则会出现异常。这种独特的实现抽象方法的方式也有其缺点。如果你写一个继承自 Pizza 的类，忘记实现 get_radius，错误将会在你使用这个方法的时候才会出现。

```python
>>> Pizza()
<__main__.Pizza object at 0x106f381d0>
>>> Pizza().get_radius()
Traceback (most recent call last):
File "<stdin>", line 1, in <module>
File "<stdin>", line 3, in get_radius
NotImplementedError
```

有种提前引起错误发生的方法，那就是当对象被实例化时，使用 Python 提供的 abc 模块。

```python
import abc
class BasePizza(object):
__metaclass__ = abc.ABCMeta

@abc.abstractmethod
def get_radius(self):
     """Method that should do something."""
```

使用 abc 和它的特类，一旦你试着实例化 BasePizza 或者其他继承自它的类，就会得到 TypeError：

```python
>>> BasePizza()
Traceback (most recent call last):
File "<stdin>", line 1, in <module>
TypeError: Can't instantiate abstract class BasePizza with abstract methods get_radius
```

**混合静态方法、类方法和抽象方法**

当我们构建类和继承关系时，终将会碰到要混合这些方法 decorator 的情况。下面提几个 tip。

记住声明一个类为抽象类时，不要冷冻那个方法的 prototype。这是指这个方法必须被实现，不过是可以使用任何参数列表来实现。

```python
import abc
class BasePizza(object):
__metaclass__  = abc.ABCMeta
@abc.abstractmethod
def get_ingredients(self):
      """Returns the ingredient list."""
class Calzone(BasePizza):
def get_ingredients(self, with_egg=False):
     egg = Egg() if with_egg else None
     return self.ingredients + egg
```

这个是合法的，因为 Calzone 完成了为 BasePizza 类对象定义的接口需求。就是说，我们可以把它当作一个类方法或者静态方法来实现，例如：

```python
import abc
class BasePizza(object):
__metaclass__  = abc.ABCMeta
@abc.abstractmethod
def get_ingredients(self):
      """Returns the ingredient list."""

class DietPizza(BasePizza):
@staticmethod
def get_ingredients():
     return None
```

这样做同样正确，并且完成了与 BasePizza 抽象类达成一致的需求。get_ingredients 方法不需要知道对象，这是实现的细节，而非完成需求的评价指标。

因此，你不能强迫抽象方法的实现是正常的方法、类方法或者静态方法，并且可以这样说，你不能。从 Python3 开始（这就不会像在 Python2 中那样 work 了，见 [issue5867](http://bugs.python.org/issue5867)），现在可以在 @abstractmethod 之上使用 @staticmethod 和 @classmethod 了。

```python
import abc
class BasePizza(object):
__metaclass__  = abc.ABCMeta

ingredient = ['cheese']

@classmethod
@abc.abstractmethod
def get_ingredients(cls):
      """Returns the ingredient list."""
      return cls.ingredients
```

不要误解：如果你认为这是强迫你的子类将 get_ingredients 实现为一个类方法，那就错了。这个是表示你实现的 get_ingredients 在 BasePizza 类中是类方法而已。

在一个抽象方法的实现？是的！在 Python 中，对比与 Java 接口，你可以在抽象方法中写代码，并且使用 super() 调用：

```python
import abc
class BasePizza(object):
__metaclass__  = abc.ABCMeta
default_ingredients = ['cheese']
@classmethod
@abc.abstractmethod
def get_ingredients(cls):
      """Returns the ingredient list."""
      return cls.default_ingredients
class DietPizza(BasePizza):
def get_ingredients(self):
     return ['egg'] + super(DietPizza, self).get_ingredients()
```

现在，每个你从 BasePizza 类继承而来的 pizza 类将重载 get_ingredients 方法，但是可以使用默认机制来使用 super() 获得 ingredient 列表。



## 定制序列

如果希望定制的容器是不可变，需要定义\__len\__()和\__getitem\__()方法
如果希望定制的容器是可变的，除了需要定义\__len\__()和\__getitem\__()方法，还需要定义\__setitem\__()和\__delitem\__()方法。

\__len\__(self)：定义当被len()调用时的行为（返回容器中元素的个数）
\__getitem\__(self)：定义获取容器中指定元素的行为，相当于self[key]
\__setitem\__(self, key, value)：定义设置容器中指定元素的行为，相当于self[key] = value
\__delitem\__(self, key):定义删除容器中指定元素的行为，相当于del self[key]

```python
class CountList:
    def __init__(self, *args):
        self.values = [x for x in args]
        self.count = {}.fromkeys(range(len(self.values)), 0)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, key):
        self.count[key] += 1
        return self.values[key]
    
>>> c1 = CountList(1, 3, 5, 7, 9)
>>> c2 = CountList(2, 4, 6, 8, 10)
>>> c1[1]
3
>>> c2[1]
4
>>> c1[1] + c2[1]
7
>>> c1.count
{0: 0, 1: 2, 2: 0, 3: 0, 4: 0}
>>> c1[1]
3
>>> c1.count
{0: 0, 1: 3, 2: 0, 3: 0, 4: 0}
```

## 迭代器

iter()

next()

```python
class Fibs:
	def __init__(self):
		self.a = 0
		self.b = 1
	def __iter__(self):
		return self
	def __next__(self):
		self.a, self.b = self.b, self.a + self.b
		return self.a
>>> fibs = Fibs()
>>> for each in fibs:
	if each < 20:
		print(each)
	else:
		break
```

## 生成器

所谓的协同程序就是可以运行的独立函数调用，函数可以暂停或者挂起，并在需要的时候从程序离开的地方继续或者重新开始。

```python
def myGen():
	print('生成器被执行！')
	yield 1
	yield 2
	
>>> myG = myGen()
>>> next(myG)
生成器被执行！
1
>>> next(myG)
2
>>> next(myG)
Traceback (most recent call last):
  File "<pyshell#163>", line 1, in <module>
    next(myG)
StopIteration
>>> for i in myGen():
	print(i)

	
生成器被执行！
1
2
```

```python
def lib():
    a = 0
    b = 1
    while True:
        a, b = b, a+b
        yield a
       
for each in libs():
    if each > 100:
        break
    print(each, end=' ')
# 1 1 2 3 5 8 13 21 34 55 89 

>>> a = [i for i in range(100) if not (i % 2) and (i % 3)]
>>> a
[2, 4, 8, 10, 14, 16, 20, 22, 26, 28, 32, 34, 38, 40, 44, 46, 50, 52, 56, 58, 62, 64, 68, 70, 74, 76, 80, 82, 86, 88, 92, 94, 98]
>>> b = {i : i % 2 == 0 for i in range(10)}
>>> b
{0: True, 1: False, 2: True, 3: False, 4: True, 5: False, 6: True, 7: False, 8: True, 9: False}
>>> c = {i for i in [1, 1, 2, 3, 4, 5, 5, 6, 7, 8, 3, 2, 3]}
>>> c
{1, 2, 3, 4, 5, 6, 7, 8}
>>> e = (i for i in range(10))
>>> e
<generator object>
>>> next(e)
0
>>>sum(i for i in range(100) if i%2)
2500
```

## 解释yield和Generators

**协程（协同程序）与子例程**

我们调用一个普通的 Python 函数时，一般是从函数的第一行代码开始执行，结束于 return 语句、异常或者函数结束（可以看作隐式的返回 None）。一旦函数将控制权交还给调用者，就意味着全部结束。函数中做的所有工作以及保存在局部变量中的数据都将丢失。再次调用这个函数时，一切都将从头创建。 

对于在计算机编程中所讨论的函数，这是很标准的流程。这样的函数只能返回一个值，不过，有时可以创建能产生一个序列的函数还是有帮助的。要做到这一点，这种函数需要能够“保存自己的工作”。 

我说过，能够“产生一个序列”是因为我们的函数并没有像通常意义那样返回。return 隐含的意思是函数正将执行代码的控制权返回给函数被调用的地方。而 yield 的隐含意思是控制权的转移是临时和自愿的，我们的函数将来还会收回控制权。

在 Python 中，拥有这种能力的“函数”被称为生成器，它非常的有用。生成器（以及 yield 语句）最初的引入是为了让程序员可以更简单的编写用来产生值的序列的代码。 以前，要实现类似随机数生成器的东西，需要实现一个类或者一个模块，在生成数据的同时保持对每次调用之间状态的跟踪。引入生成器之后，这变得非常简单。

为了更好的理解生成器所解决的问题，让我们来看一个例子。在了解这个例子的过程中，请始终记住我们需要解决的问题：**生成值的序列**。

*注意：在 Python 之外，最简单的生成器应该是被称为协程（coroutines）的东西。在本文中，我将使用这个术语。请记住，在 Python 的概念中，这里提到的协程就是生成器。Python 正式的术语是生成器；协程只是便于讨论，在语言层面并没有正式定义。*


**例子：有趣的素数**

假设你的老板让你写一个函数，输入参数是一个 int 的 list，返回一个可以迭代的包含素数 1 的结果。

记住，迭代器（Iterable） 只是对象每次返回特定成员的一种能力。

你肯定认为"这很简单"，然后很快写出下面的代码：

```python
def get_primes(input_list):
    result_list = list()
    for element in input_list:
        if is_prime(element):
            result_list.append()

    return result_list

# 或者更好一些的...

def get_primes(input_list):
    return (element for element in input_list if is_prime(element))

# 下面是 is_prime 的一种实现...

def is_prime(number):
    if number > 1:
        if number == 2:
            return True
        if number % 2 == 0:
            return False
        for current in range(3, int(math.sqrt(number) + 1), 2):
            if number % current == 0:
                return False
        return True
    return False
```

上面 is_prime 的实现完全满足了需求，所以我们告诉老板已经搞定了。她反馈说我们的函数工作正常，正是她想要的。


**处理无限序列**

噢，真是如此吗？过了几天，老板过来告诉我们她遇到了一些小问题：她打算把我们的 get_primes 函数用于一个很大的包含数字的 list。实际上，这个 list 非常大，仅仅是创建这个 list 就会用完系统的所有内存。为此，她希望能够在调用 get_primes 函数时带上一个 start 参数，返回所有大于这个参数的素数（也许她要解决 [Project Euler problem 10](https://projecteuler.net/problem=10)）。

我们来看看这个新需求，很明显只是简单的修改 get_primes 是不可能的。 自然，我们不可能返回包含从 start 到无穷的所有的素数的列表（虽然有很多有用的应用程序可以用来操作无限序列）。看上去用普通函数处理这个问题的可能性比较渺茫。

在我们放弃之前，让我们确定一下最核心的障碍，是什么阻止我们编写满足老板新需求的函数。通过思考，我们得到这样的结论：函数只有一次返回结果的机会，因而必须一次返回所有的结果。得出这样的结论似乎毫无意义；“函数不就是这样工作的么”，通常我们都这么认为的。可是，不学不成，不问不知，“如果它们并非如此呢？”

想象一下，如果 get_primes 可以只是简单返回下一个值，而不是一次返回全部的值，我们能做什么？我们就不再需要创建列表。没有列表，就没有内存的问题。由于老板告诉我们的是，她只需要遍历结果，她不会知道我们实现上的区别。

不幸的是，这样做看上去似乎不太可能。即使是我们有神奇的函数，可以让我们从 n 遍历到无限大，我们也会在返回第一个值之后卡住：

```python
def get_primes(start):
    for element in magical_infinite_range(start):
        if is_prime(element):
            return element
```

假设这样去调用 get_primes：

```
def solve_number_10():
    # She *is* working on Project Euler #10, I knew it!
    total = 2
    for next_prime in get_primes(3):
        if next_prime < 2000000:
            total += next_prime
        else:
            print(total)
            return
```

显然，在 get_primes 中，一上来就会碰到输入等于 3 的，并且在函数的第 4 行返回。与直接返回不同，我们需要的是在退出时可以为下一次请求准备一个值。

不过函数做不到这一点。当函数返回时，意味着全部完成。我们保证函数可以再次被调用，但是我们没法保证说，“呃，这次从上次退出时的第 4 行开始执行，而不是常规的从第一行开始”。函数只有一个单一的入口：函数的第 1 行代码。


**走进生成器**

这类问题极其常见以至于 Python 专门加入了一个结构来解决它：生成器。一个生成器会“生成”值。创建一个生成器几乎和生成器函数的原理一样简单。

一个生成器函数的定义很像一个普通的函数，除了当它要生成一个值的时候，使用 yield 关键字而不是 return。如果一个 def 的主体包含 yield，这个函数会自动变成一个生成器（即使它包含一个 return）。除了以上内容，创建一个生成器没有什么多余步骤了。

生成器函数返回生成器的迭代器。这可能是你最后一次见到“生成器的迭代器”这个术语了， 因为它们通常就被称作“生成器”。要注意的是生成器就是一类特殊的迭代器。作为一个迭代器，生成器必须要定义一些方法（method），其中一个就是 \__next\__()。如同迭代器一样，我们可以使用 next() 函数来获取下一个值。

为了从生成器获取下一个值，我们使用 next() 函数，就像对付迭代器一样（next() 会操心如何调用生成器的 \__next\__() 方法，不用你操心）。既然生成器是一个迭代器，它可以被用在 for 循环中。

每当生成器被调用的时候，它会返回一个值给调用者。在生成器内部使用 yield 来完成这个动作（例如 yield 7）。为了记住 yield 到底干了什么，最简单的方法是把它当作专门给生成器函数用的特殊的 return（加上点小魔法）。

下面是一个简单的生成器函数：

```python
>>> def simple_generator_function():
>>>    yield 1
>>>    yield 2
>>>    yield 3
```

这里有两个简单的方法来使用它：

```python
>>> for value in simple_generator_function():
>>>     print(value)
1
2
3
>>> our_generator = simple_generator_function()
>>> next(our_generator)
1
>>> next(our_generator)
2
>>> next(our_generator)
3
```

**魔法？**

那么神奇的部分在哪里？

我很高兴你问了这个问题！当一个生成器函数调用 yield，生成器函数的“状态”会被冻结，所有的变量的值会被保留下来，下一行要执行的代码的位置也会被记录，直到再次调用 next()。一旦 next() 再次被调用，生成器函数会从它上次离开的地方开始。如果永远不调用 next()，yield 保存的状态就被无视了。

我们来重写 get_primes() 函数，这次我们把它写作一个生成器。注意我们不再需要 magical_infinite_range 函数了。使用一个简单的 while 循环，我们创造了自己的无穷串列。

```python
def get_primes(number):
    while True:
        if is_prime(number):
            yield number
        number += 1
```

如果生成器函数调用了 return，或者执行到函数的末尾，会出现一个 StopIteration 异常。 这会通知 next() 的调用者这个生成器没有下一个值了（这就是普通迭代器的行为）。这也是这个 while 循环在我们的 get_primes() 函数出现的原因。如果没有这个 while，当我们第二次调用 next() 的时候，生成器函数会执行到函数末尾，触发 StopIteration 异常。一旦生成器的值用完了，再调用 next() 就会出现错误，所以你只能将每个生成器的使用一次。下面的代码是错误的：

```python
>>> our_generator = simple_generator_function()
>>> for value in our_generator:
>>>     print(value)

>>> # 我们的生成器没有下一个值了...
>>> print(next(our_generator))
Traceback (most recent call last):
  File "<ipython-input-13-7e48a609051a>", line 1, in <module>
    next(our_generator)
StopIteration

>>> # 然而，我们总可以再创建一个生成器
>>> # 只需再次调用生成器函数即可

>>> new_generator = simple_generator_function()
>>> print(next(new_generator)) # 工作正常
1
```

因此，这个 while 循环是用来确保生成器函数永远也不会执行到函数末尾的。只要调用 next() 这个生成器就会生成一个值。这是一个处理无穷序列的常见方法（这类生成器也是很常见的）。


**执行流程**

让我们回到调用 get_primes 的地方：solve_number_10。

```python
def solve_number_10():
    # She *is* working on Project Euler #10, I knew it!
    total = 2
    for next_prime in get_primes(3):
        if next_prime < 2000000:
            total += next_prime
        else:
            print(total)
            return
```

我们来看一下 solve_number_10 的 for 循环中对 get_primes 的调用，观察一下前几个元素是如何创建的有助于我们的理解。当 for 循环从 get_primes 请求第一个值时，我们进入 get_primes，这时与进入普通函数没有区别。

- 进入第三行的 while 循环
- 停在 if 条件判断（3 是素数）
- 通过 yield 将 3 和执行控制权返回给 solve_number_10

接下来，回到 insolve_number_10：

- for 循环得到返回值 3
- for 循环将其赋给 next_prime
- total 加上 next_prime
- for 循环从 get_primes 请求下一个值

这次，进入 get_primes 时并没有从开头执行，我们从第 5 行继续执行，也就是上次离开的地方。

```python
def get_primes(number):
    while True:
        if is_prime(number):
            yield number
        number += 1 # <<<<<<<<<<
```

最关键的是，number 还保持我们上次调用 yield 时的值（例如 3）。记住，yield 会将值传给 next() 的调用方，同时还会保存生成器函数的“状态”。接下来，number 加到 4，回到 while 循环的开始处，然后继续增加直到得到下一个素数（5）。我们再一次把 number 的值通过 yield 返回给 solve_number_10 的 for 循环。这个周期会一直执行，直到 for 循环结束（得到的素数大于2,000,000）。


**更给力点**

在 PEP 342 中加入了将值传给生成器的支持。PEP 342 加入了新的特性，能让生成器在单一语句中实现，生成一个值（像从前一样），接受一个值，或同时生成一个值并接受一个值。

我们用前面那个关于素数的函数来展示如何将一个值传给生成器。这一次，我们不再简单地生成比某个数大的素数，而是找出比某个数的等比级数大的最小素数（例如 10， 我们要生成比 10，100，1000，10000 ... 大的最小素数）。我们从 get_primes 开始：

```python
def print_successive_primes(iterations, base=10):
    # 像普通函数一样，生成器函数可以接受一个参数
    
    prime_generator = get_primes(base)
    # 这里以后要加上点什么
    for power in range(iterations):
        # 这里以后要加上点什么

def get_primes(number):
    while True:
        if is_prime(number):
        # 这里怎么写?
```

get_primes 的后几行需要着重解释。yield 关键字返回 number 的值，而像 other = yield foo 这样的语句的意思是，“返回 foo 的值，这个值返回给调用者的同时，将 other 的值也设置为那个值”。你可以通过 send 方法来将一个值“发送”给生成器。

```python
def get_primes(number):
    while True:
        if is_prime(number):
            number = yield number
        number += 1
```

通过这种方式，我们可以在每次执行 yield 的时候为 number 设置不同的值。现在我们可以补齐 print_successive_primes 中缺少的那部分代码：

```python
def print_successive_primes(iterations, base=10):
    prime_generator = get_primes(base)
    prime_generator.send(None)
    for power in range(iterations):
        print(prime_generator.send(base ** power))
```

这里有两点需要注意：首先，我们打印的是 generator.send 的结果，这是没问题的，因为 send 在发送数据给生成器的同时还返回生成器通过 yield 生成的值（就如同生成器中 yield 语句做的那样）。

第二点，看一下 prime_generator.send(None) 这一行，当你用 send 来“启动”一个生成器时（就是从生成器函数的第一行代码执行到第一个 yield 语句的位置），你必须发送 None。这不难理解，根据刚才的描述，生成器还没有走到第一个 yield 语句，如果我们发生一个真实的值，这时是没有人去“接收”它的。一旦生成器启动了，我们就可以像上面那样发送数据了。


**综述**

在本系列文章的后半部分，我们将讨论一些 yield 的高级用法及其效果。yield 已经成为 Python 最强大的关键字之一。现在我们已经对 yield 是如何工作的有了充分的理解，我们已经有了必要的知识，可以去了解 yield 的一些更“费解”的应用场景。

不管你信不信，我们其实只是揭开了 yield 强大能力的一角。例如，send 确实如前面说的那样工作，但是在像我们的例子这样，只是生成简单的序列的场景下，send 几乎从来不会被用到。下面我贴一段代码，展示 send 通常的使用方式。对于这段代码如何工作以及为何可以这样工作，在此我并不打算多说，它将作为第二部分很不错的热身。

```python
import random

def get_data():
    """返回0到9之间的3个随机数"""
    return random.sample(range(10), 3)

def consume():
    """显示每次传入的整数列表的动态平均值"""
    running_sum = 0
    data_items_seen = 0

    while True:
        data = yield
        data_items_seen += len(data)
        running_sum += sum(data)
        print('The running average is {}'.format(running_sum / float(data_items_seen)))

def produce(consumer):
    """产生序列集合，传递给消费函数（consumer）"""
    while True:
        data = get_data()
        print('Produced {}'.format(data))
        consumer.send(data)
        yield

if __name__ == '__main__':
    consumer = consume()
    consumer.send(None)
    producer = produce(consumer)

    for _ in range(10):
        print('Producing...')
        next(producer)
```

总结

- generator 是用来产生一系列值的
- yield 则像是 generator 函数的返回结果
- yield 唯一所做的另一件事就是保存一个 generator 函数的状态
- generator 就是一个特殊类型的迭代器（iterator）
- 和迭代器相似，我们可以通过使用 next() 来从 generator 中获取下一个值
- 通过隐式地调用 next() 来忽略一些值

