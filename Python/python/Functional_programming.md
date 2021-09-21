## 函数式编程

来源：酷壳网-陈皓    链接：

https://coolshell.cn/articles/10822.html

当我们说起函数式编程来说，我们会看到如下函数式编程的长相：

- 函数式编程的三大特性：

- - **immutable data 不可变数据**：像Clojure一样，默认上变量是不可变的，如果你要改变变量，你需要把变量copy出去修改。这样一来，可以让你的程序少很多Bug。因为，程序中的状态不好维护，在并发的时候更不好维护。（你可以试想一下如果你的程序有个复杂的状态，当以后别人改你代码的时候，是很容易出bug的，在并行中这样的问题就更多了）
  - **first class functions**：这个技术可以让你的函数就像变量一样来使用。也就是说，你的函数可以像变量一样被创建，修改，并当成变量一样传递，返回或是在函数中嵌套函数。这个有点像Javascript的Prototype（参看Javascript的面向对象编程）
  - **尾递归优化**：我们知道递归的害处，那就是如果递归很深的话，stack受不了，并会导致性能大幅度下降。所以，我们使用尾递归优化技术——每次递归时都会重用stack，这样一来能够提升性能，当然，这需要语言或编译器的支持。Python就不支持。


- 函数式编程的几个技术

- - **map & reduce** ：这个技术不用多说了，函数式编程最常见的技术就是对一个集合做Map和Reduce操作。这比起过程式的语言来说，在代码上要更容易阅读。（传统过程式的语言需要使用for/while循环，然后在各种变量中把数据倒过来倒过去的）这个很像C++中的STL中的foreach，find_if，count_if之流的函数的玩法。
  - **pipeline**：这个技术的意思是，把函数实例成一个一个的action，然后，把一组action放到一个数组或是列表中，然后把数据传给这个action list，数据就像一个pipeline一样顺序地被各个函数所操作，最终得到我们想要的结果。
  - **recursing 递归** ：递归最大的好处就简化代码，他可以把一个复杂的问题用很简单的代码描述出来。注意：递归的精髓是描述问题，而这正是函数式编程的精髓。
  - **currying**：把一个函数的多个参数分解成多个函数， 然后把函数多层封装起来，每层函数都返回一个函数去接收下一个参数这样，可以简化函数的多个参数。在C++中，这个很像STL中的bind_1st或是bind2nd。
  - **higher order function 高阶函数**：所谓高阶函数就是函数当参数，把传入的函数做一个封装，然后返回这个封装函数。现象上就是函数传进传出，就像面向对象对象满天飞一样。


- 还有函数式的一些好处

- - **parallelization 并行：**所谓并行的意思就是在并行环境下，各个线程之间不需要同步或互斥。
  - **lazy evaluation 惰性求值**：这个需要编译器的支持。表达式不在它被绑定到变量之后就立即求值，而是在该值被取用的时候求值，也就是说，语句如*x:=expression;* (把一个表达式的结果赋值给一个变量)明显的调用这个表达式被计算并把结果放置到 *x* 中，但是先不管实际在 *x* 中的是什么，直到通过后面的表达式中到 *x* 的引用而有了对它的值的需求的时候，而后面表达式自身的求值也可以被延迟，最终为了生成让外界看到的某个符号而计算这个快速增长的依赖树。
  - **determinism 确定性**：所谓确定性的意思就是像数学那样 f(x) = y ，这个函数无论在什么场景下，都会得到同样的结果，这个我们称之为函数的确定性。而不是像程序中的很多函数那样，同一个参数，却会在不同的场景下计算出不同的结果。所谓不同的场景的意思就是我们的函数会根据一些运行中的状态信息的不同而发生变化。

上面的那些东西太抽象了，还是让我们来循序渐近地看一些例子吧。

我们先用一个最简单的例子来说明一下什么是函数式编程。

先看一个非函数式的例子：

```
int cnt;
void increment(){
    cnt++;
}
```

那么，函数式的应该怎么写呢？

```
int increment(int cnt){
    return cnt+1;
}
```

你可能会觉得这个例子太普通了。是的，这个例子就是函数式编程的准则：**不依赖于外部的数据，而且也不改变外部数据的值，而是返回一个新的值给你**。

我们再来看一个简单例子：

```
def inc(x):
    def incx(y):
        return x+y
    return incx

inc2 = inc(2)
inc5 = inc(5)

print inc2(5) # 输出 7
print inc5(5) # 输出 10
```

我们可以看到上面那个例子inc()函数返回了另一个函数incx()，于是我们可以用inc()函数来构造各种版本的inc函数，比如：inc2()和inc5()。这个技术其实就是上面所说的Currying技术。从这个技术上，你可能体会到函数式编程的理念：**把函数当成变量来用，关注于描述问题而不是怎么实现**，这样可以让代码更易读。

**Map & Reduce**

在函数式编程中，我们不应该用循环迭代的方式，我们应该用更为高级的方法，如下所示的Python代码

```
name_len = map(len, ["hao", "chen", "coolshell"])
print name_len
# 输出 [3, 4, 9]
```

你可以看到这样的代码很易读，因为，**这样的代码是在描述要干什么，而不是怎么干**。

我们再来看一个Python代码的例子：

```
def toUpper(item):
      return item.upper()

upper_name = map(toUpper, ["hao", "chen", "coolshell"])
print upper_name
# 输出 ['HAO', 'CHEN', 'COOLSHELL']
```

顺便说一下，上面的例子个是不是和我们的STL的transform有些像？

```
#include <iostream>
#include <algorithm>
#include <string>
using namespace std;

int main() {
  string s="hello";
  string out;
  transform(s.begin(), s.end(), back_inserter(out), ::toupper);
  cout << out << endl;
  // 输出：HELLO
}
```

在上面Python的那个例子中我们可以看到，我们写义了一个函数toUpper，这个函数没有改变传进来的值，只是把传进来的值做个简单的操作，然后返回。然后，我们把其用在map函数中，就可以很清楚地描述出我们想要干什么。而不会去理解一个在循环中的怎么实现的代码，最终在读了很多循环的逻辑后才发现原来是这个或那个意思。 下面，我们看看描述实现方法的过程式编程是怎么玩的（看上去是不是不如函数式的清晰？）：

```
upname =['HAO', 'CHEN', 'COOLSHELL']
lowname =[] 
for i in range(len(upname)):
    lowname.append( upname[i].lower() )
```

对于map我们别忘了lambda表达式：你可以简单地理解为这是一个inline的匿名函数。下面的lambda表达式相当于：def func(x): return x*x

```
squares = map(lambda x: x * x, range(9))
print squares
# 输出 [0, 1, 4, 9, 16, 25, 36, 49, 64]
```

我们再来看看reduce怎么玩？（下面的lambda表达式中有两个参数，也就是说每次从列表中取两个值，计算结果后把这个值再放回去，下面的表达式相当于：((((1+2)+3)+4)+5) ）

```
print reduce(lambda x, y: x+y, [1, 2, 3, 4, 5])
# 输出 15
```

Python中的除了map和reduce外，还有一些别的如filter, find, all, any的函数做辅助（其它函数式的语言也有），可以让你的代码更简洁，更易读。 我们再来看一个比较复杂的例子：

计算数组中正数的平均值

```
num =[2, -5, 9, 7, -2, 5, 3, 1, 0, -3, 8]
positive_num_cnt = 0
positive_num_sum = 0
for i in range(len(num)):
    if num[i] > 0:
        positive_num_cnt += 1
        positive_num_sum += num[i]

if positive_num_cnt > 0:
    average = positive_num_sum / positive_num_cnt

print average
# 输出 5
```

如果用函数式编程，这个例子可以写成这样：

```
positive_num = filter(lambda x: x>0, num)
average = reduce(lambda x,y: x+y, positive_num) / len( positive_num )
```

C++11玩的法：

```
#include <iostream>
#include <algorithm>
#include <string>
using namespace std;

int main() {
  string s="hello";
  string out;
  transform(s.begin(), s.end(), back_inserter(out), ::toupper);
  cout << out << endl;
  // 输出：HELLO
}
```

我们可以看到，函数式编程有如下好处：

1）代码更简单了。
2）数据集，操作，返回值都放到了一起。
3）你在读代码的时候，没有了循环体，于是就可以少了些临时变量，以及变量倒来倒去逻辑。
4）你的代码变成了在描述你要干什么，而不是怎么去干。

最后，我们来看一下Map/Reduce这样的函数是怎么来实现的（下面是Javascript代码）

MAP函数

```
var map = function (mappingFunction, list) {
  var result = [];
  forEach(list, function (item) {
    result.push(mappingFunction(item));
  });
  return result;
};
```

下面是reduce函数的javascript实现

REDUCE函数

```
function reduce(actionFunction, list, initial){
    var accumulate;
    var temp;
    if(initial){
        accumulate = initial;
    }
    else{
        accumulate = list.shfit();
    }
    temp = list.shift();
    while(temp){
        accumulate = actionFunction(accumulate,temp);
        temp = list.shift();
    }
    return accumulate;
};
```

**Declarative Programming vs Imperative Programming**

前面提到过多次的函数式编程关注的是：describe what to do, rather than how to do it. 于是，我们把以前的过程式的编程范式叫做 Imperative Programming – 指令式编程，而把函数式的这种范式叫做 Declarative Programming – 声明式编程。

下面我们看一下相关的示例（本示例来自这篇文章 ）。

比如，我们有3辆车比赛，简单起见，我们分别给这3辆车有70%的概率可以往前走一步，一共有5次机会，我们打出每一次这3辆车的前行状态。

对于Imperative Programming来说，代码如下（Python）：

```
from random import random

time = 5
car_positions = [1, 1, 1]

while time:
    # decrease time
    time -= 1

    print ''
    for i in range(len(car_positions)):
        # move car
        if random() > 0.3:
            car_positions[i] += 1

        # draw car
        print '-' * car_positions[i]
```

我们可以把这个两重循环变成一些函数模块，这样有利于我们更容易地阅读代码：

```
from random import random

def move_cars():
    for i, _ in enumerate(car_positions):
        if random() > 0.3:
            car_positions[i] += 1

def draw_car(car_position):
    print '-' * car_position

def run_step_of_race():
    global time
    time -= 1
    move_cars()

def draw():
    print ''
    for car_position in car_positions:
        draw_car(car_position)

time = 5
car_positions = [1, 1, 1]

while time:
    run_step_of_race()
    draw()
```

上面的代码，我们可以从主循环开始，我们可以很清楚地看到程序的主干，因为我们把程序的逻辑分成了几个函数，这样一来，我们的代码逻辑也会变得几个小碎片，于是我们读代码时要考虑的上下文就少了很多，阅读代码也会更容易。不像第一个示例，如果没有注释和说明，你还是需要花些时间理解一下。**而把代码逻辑封装成了函数后，我们就相当于给每个相对独立的程序逻辑取了个名字，于是代码成了自解释的**。

但是，你会发现，封装成函数后，这些函数都会依赖于共享的变量来同步其状态。于是，我们在读代码的过程时，每当我们进入到函数里，一量读到访问了一个外部的变量，我们马上要去查看这个变量的上下文，然后还要在大脑里推演这个变量的状态， 我们才知道程序的真正逻辑。也就是说，**这些函数间必需知道其它函数是怎么修改它们之间的共享变量的，所以，这些函数是有状态的**。

我们知道，有状态并不是一件很好的事情，无论是对代码重用，还是对代码的并行来说，都是有副作用的。因此，我们要想个方法把这些状态搞掉，于是出现了我们的 Functional Programming 的编程范式。下面，我们来看看函数式的方式应该怎么写？

```
from random import random

def move_cars(car_positions):
    return map(lambda x: x + 1 if random() > 0.3 else x,
               car_positions)

def output_car(car_position):
    return '-' * car_position

def run_step_of_race(state):
    return {'time': state['time'] - 1,
            'car_positions': move_cars(state['car_positions'])}

def draw(state):
    print ''
    print '\n'.join(map(output_car, state['car_positions']))

def race(state):
    draw(state)
    if state['time']:
        race(run_step_of_race(state))

race({'time': 5,
      'car_positions': [1, 1, 1]})
```

上面的代码依然把程序的逻辑分成了函数，不过这些函数都是functional的。因为它们有三个症状：

1）它们之间没有共享的变量。
2）函数间通过参数和返回值来传递数据。
3）在函数里没有临时变量。

我们还可以看到，for循环被递归取代了（见race函数）—— 递归是函数式编程中带用到的技术，正如前面所说的，递归的本质就是描述问题是什么。

State is like a box of chocolates, You never know what you are gonna get.

**Pipeline**

pipeline 管道借鉴于Unix Shell的管道操作——把若干个命令串起来，前面命令的输出成为后面命令的输入，如此完成一个流式计算。（注：管道绝对是一个伟大的发明，他的设哲学就是KISS – 让每个功能就做一件事，并把这件事做到极致，软件或程序的拼装会变得更为简单和直观。这个设计理念影响非常深远，包括今天的Web Service，云计算，以及大数据的流式计算等等）

比如，我们如下的shell命令：

```
ps auwwx | awk '{print $2}' | sort -n | xargs echo
```

如果我们抽象成函数式的语言，就像下面这样：

```
xargs(  echo, sort(n, awk('print $2', ps(auwwx)))  )
```

也可以类似下面这个样子：

```
pids = for_each(result, [ps_auwwx, awk_p2, sort_n, xargs_echo])
```

好了，让我们来看看函数式编程的Pipeline怎么玩？

我们先来看一个如下的程序，这个程序的process()有三个步骤：

1）找出偶数。
2）乘以3
3）转成字符串返回

```
def process(num):
    # filter out non-evens
    if num % 2 != 0:
        return
    num = num * 3
    num = 'The Number: %s' % num
    return num

nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

for num in nums:
    print process(num)

# 输出：
# None
# The Number: 6
# None
# The Number: 12
# None
# The Number: 18
# None
# The Number: 24
# None
# The Number: 30
```

我们可以看到，输出的并不够完美，另外，代码阅读上如果没有注释，你也会比较晕。下面，我们来看看函数式的pipeline（第一种方式）应该怎么写？

```
def even_filter(nums):
    for num in nums:
        if num % 2 == 0:
            yield num
def multiply_by_three(nums):
    for num in nums:
        yield num * 3
def convert_to_string(nums):
    for num in nums:
        yield 'The Number: %s' % num

nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
pipeline = convert_to_string(multiply_by_three(even_filter(nums)))
for num in pipeline:
    print num
# 输出：
# The Number: 6
# The Number: 12
# The Number: 18
# The Number: 24
# The Number: 30

```

我们动用了Python的关键字 yield，这个关键字主要是返回一个Generator，yield 是一个类似 return 的关键字，只是这个函数返回的是个Generator-生成器。所谓生成器的意思是，yield返回的是一个可迭代的对象，并没有真正的执行函数。也就是说，只有其返回的迭代对象被真正迭代时，yield函数才会正真的运行，运行到yield语句时就会停住，然后等下一次的迭代。（这个是个比较诡异的关键字）这就是lazy evluation。

好了，根据前面的原则——“**使用Map & Reduce，不要使用循环**”，那我们用比较纯朴的Map & Reduce吧。

```
def even_filter(nums):
    return filter(lambda x: x%2==0, nums)

def multiply_by_three(nums):
    return map(lambda x: x*3, nums)

def convert_to_string(nums):
    return map(lambda x: 'The Number: %s' % x,  nums)

nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
pipeline = convert_to_string(
               multiply_by_three(
                   even_filter(nums)
               )
            )
for num in pipeline:
    print num
```

但是他们的代码需要嵌套使用函数，这个有点不爽，如果我们能像下面这个样子就好了（第二种方式）。

```
pipeline_func(nums, [even_filter,
                     multiply_by_three,
                     convert_to_string])
```

那么，pipeline_func 实现如下：

```
def pipeline_func(data, fns):
    return reduce(lambda a, x: x(a),
                  fns,
                  data)
```

好了，在读过这么多的程序后，你可以回头看一下这篇文章的开头对函数式编程的描述，可能你就更有感觉了。

#### 参考

- Wikipedia: [Functional Programming](http://en.wikipedia.org/wiki/Functional_programming)

- [truly understanding the difference between procedural and functional](http://stackoverflow.com/questions/5226055/truly-understanding-the-difference-between-procedural-and-functional)

- [A practical introduction to functional programming](http://maryrosecook.com/post/a-practical-introduction-to-functional-programming)

- [What is the difference between procedural programming and functional programming?](http://stackoverflow.com/q/23277/211232)

- [Can someone give me examples of functional programming vs imperative/procedural programming?](http://stackoverflow.com/q/3249863/211232)

- [OOP vs Functional Programming vs Procedural](http://stackoverflow.com/q/552336/211232)

- Python – [Functional Programming HOWTO](http://docs.python.org/2/howto/functional.html#)

  ​

感谢谢网友提供的shell风格的python pipeline：

```
class Pipe(object):
    def __init__(self, func):
        self.func = func

    def __ror__(self, other):
        def generator():
            for obj in other:
                if obj is not None:
                    yield self.func(obj)
        return generator()

@Pipe
def even_filter(num):
    return num if num % 2 == 0 else None

@Pipe
def multiply_by_three(num):
    return num*3

@Pipe
def convert_to_string(num):
    return 'The Number: %s' % num

@Pipe
def echo(item):
    print item
    return item

def force(sqs):
    for item in sqs: pass

nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

force(nums | even_filter | multiply_by_three | convert_to_string | echo)
```









