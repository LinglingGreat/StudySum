problem-solving-with-algorithms-and-data-structure-using-python中文版笔记

github 地址: https://github.com/facert/python-data-structure-cn
gitbook 在线浏览: https://facert.gitbooks.io/python-data-structure-cn

python算法复杂度列表：

https://wiki.python.org/moin/TimeComplexity

### 乱序字符串检查

乱序字符串是指一个字符串只是另一个字符串的重新排列。例如，'heart' 和 'earth' 就是乱序字符串。'python' 和 'typhon' 也是。为了简单起见，我们假设所讨论的两个字符串具有相等的长度，并且他们由 26 个小写字母集合组成。我们的目标是写一个布尔函数，它将两个字符串做参数并返回它们是不是乱序。

解法1：检查

```python
def anagramSolution1(s1,s2):
	alist = list(s2)
    
	pos1 = 0
	stillOK = True
    
	while pos1 < len(s1) and stillOK:
		pos2 = 0
		found = False
		while pos2 < len(alist) and not found:
			if s1[pos1] == alist[pos2]:
				found = True
			else:
				pos2 = pos2 + 1
                
		if found:
			alist[pos2] = None
		else:
			stillOK = False
            
		pos1 = pos1 + 1
        
	return stillOK

print(anagramSolution1('abcd','dcba'))
```

算法复杂度：

$\sum_{i=1}^n i = n(n+1)/2=\frac12n^2+\frac12n$

$O(n^2)$

解法2：排序和比较

```python
def anagramSolution2(s1,s2):
	alist1 = list(s1)
	alist2 = list(s2)
	alist1.sort()
	alist2.sort()
	pos = 0
	matches = True
	while pos < len(s1) and matches:
		if alist1[pos]==alist2[pos]:
			pos = pos + 1
		else:
			matches = False
	return matches
print(anagramSolution2('abcde','edcba'))
```

算法复杂度O(n)+排序复杂度O(n^2)或O(nlogn)

解法3：穷举法

对于乱序检测，我们可以生成 s1 的所有乱序字符串列表，然后查看是不是有 s2。这种方法有一点困难。当 s1 生成所有可能的字符串时，第一个位置有 n 种可能，第二个位置有 n-1 种，第三个位置有 n-3 种，等等。总数为 n∗(n−1)∗(n−2)∗...∗3∗2∗1n∗(n−1)∗(n−2)∗...∗3∗2∗1， 即 n!。虽然一些字符串可能是重复的，程序也不可能提前知道这样，所以他仍然会生成 n! 个字符串。
事实证明，n! 比 n^2 增长还快，事实上，如果 s1 有 20个字符长，则将有 20! =2,432,902,008,176,640,000 个字符串产生。如果我们每秒处理一种可能字符串，那么需要77,146,816,596 年才能过完整个列表。所以这不是很好的解决方案。

解法4: 计数和比较

```python
def anagramSolution4(s1,s2):
    c1 = [0]*26
    c2 = [0]*26

    for i in range(len(s1)):
        pos = ord(s1[i])-ord('a')
        c1[pos] = c1[pos] + 1

    for i in range(len(s2)):
        pos = ord(s2[i])-ord('a')
        c2[pos] = c2[pos] + 1

    j = 0
    stillOK = True
    while j<26 and stillOK:
        if c1[j]==c2[j]:
            j = j + 1
        else:
            stillOK = False

    return stillOK

print(anagramSolution4('apple','pleap'))
```

T(n)=2n+26T(n)=2n+26，即O(n)，线性量级

虽然最后一个方案在线性时间执行，但它需要额外的存储来保存两个字符计数列表。换句话说，该算法牺牲了空间以获得时间。

### 列表

两个常见的操作是索引和分配到索引位置。无论列表有多大，这两个操作都需要相同的时间。当这样的操作和列表的大小无关时，它们是 O（1）。

另一个非常常见的编程任务是增加一个列表。有两种方法可以创建更长的列表，可以使用append 方法或拼接运算符。append 方法是 O（1)。 然而，拼接运算符是 O（k），其中 k是要拼接的列表的大小。

```python
def test1():
    l = []
    for i in range(1000):
        l = l + [i]

def test2():
    l = []
    for i in range(1000):
        l.append(i)

def test3():
    l = [i for i in range(1000)]

def test4():
    l = list(range(1000))
```

```python
t1 = Timer("test1()", "from __main__ import test1")
print("concat ",t1.timeit(number=1000), "milliseconds")
t2 = Timer("test2()", "from __main__ import test2")
print("append ",t2.timeit(number=1000), "milliseconds")
t3 = Timer("test3()", "from __main__ import test3")
print("comprehension ",t3.timeit(number=1000), "milliseconds")
t4 = Timer("test4()", "from __main__ import test4")
print("list range ",t4.timeit(number=1000), "milliseconds")

concat  6.54352807999 milliseconds
append  0.306292057037 milliseconds
comprehension  0.147661924362 milliseconds
list range  0.0655000209808 milliseconds
```

当列表末尾调用 pop 时，它需要 O(1), 但是当在列表中第一个元素或者中间任何地方调用 pop,它是 O(n)。原因在于 Python 实现列表的方式，当一个项从列表前面取出，列表中的其他元素靠近起始位置移动一个位置。你会看到索引操作为 O(1)。

| Operation        | Big-O Efficiency |
| ---------------- | ---------------- |
| index []         | O(1)             |
| index assignment | O(1)             |
| append           | O(1)             |
| pop()            | O(1)             |
| pop(i)           | O(n)             |
| insert(i,item)   | O(n)             |
| del operator     | O(n)             |
| iteration        | O(n)             |
| contains (in)    | O(n)             |
| get slice [x:y]  | O(k)             |
| del slice        | O(n)             |
| set slice        | O(n+k)           |
| reverse          | O(n)             |
| concatenate      | O(k)             |
| sort             | O(n log n)       |
| multiply         | O(nk)            |

### 字典

| operation     | Big-O Efficiency |
| ------------- | ---------------- |
| copy          | O(n)             |
| get item      | O(1)             |
| set item      | O(1)             |
| delete item   | O(1)             |
| contains (in) | O(1)             |
| iteration     | O(n)             |

列表的 contains 操作符是 O(n)，字典的 contains 操作符是 O(1)。

### 栈

Stack() 创建一个空的新栈。 它不需要参数，并返回一个空栈。
push(item)将一个新项添加到栈的顶部。它需要 item 做参数并不返回任何内容。
pop() 从栈中删除顶部项。它不需要参数并返回 item 。栈被修改。
peek() 从栈返回顶部项，但不会删除它。不需要参数。 不修改栈。
isEmpty() 测试栈是否为空。不需要参数，并返回布尔值。
size() 返回栈中的 item 数量。不需要参数，并返回一个整数。

| **Stack Operation** | **Stack Contents**   | **Return Value** |
| ------------------- | -------------------- | ---------------- |
| `s.isEmpty()`       | `[]`                 | `True`           |
| `s.push(4)`         | `[4]`                |                  |
| `s.push('dog')`     | `[4,'dog']`          |                  |
| `s.peek()`          | `[4,'dog']`          | `'dog'`          |
| `s.push(True)`      | `[4,'dog',True]`     |                  |
| `s.size()`          | `[4,'dog',True]`     | `3`              |
| `s.isEmpty()`       | `[4,'dog',True]`     | `False`          |
| `s.push(8.4)`       | `[4,'dog',True,8.4]` |                  |
| `s.pop()`           | `[4,'dog',True]`     | `8.4`            |
| `s.pop()`           | `[4,'dog']`          | `True`           |
| `s.size()`          | `[4,'dog']`          | `2`              |

**Python的栈实现**

以下栈实现（ActiveCode 1）假定列表的结尾将保存栈的顶部元素。随着栈增长（push 操作），新项将被添加到列表的末尾。 pop 也操作列表末尾的元素。

```python
class Stack:
     def __init__(self):
         self.items = []

     def isEmpty(self):
         return self.items == []

     def push(self, item):
         self.items.append(item)

     def pop(self):
         return self.items.pop()

     def peek(self):
         return self.items[len(self.items)-1]

     def size(self):
         return len(self.items)
```

Note pythonds 模块包含本书中讨论的所有数据结构的实现。它根据以下部分构造：基本数据类型，树和图。 该模块可以从 [pythonworks.org](http://www.pythonworks.org/pythonds). 下载。

```python
from pythonds.basic.stack import Stack

s=Stack()

print(s.isEmpty())
s.push(4)
s.push('dog')
print(s.peek())
s.push(True)
print(s.size())
print(s.isEmpty())
s.push(8.4)
print(s.pop())
print(s.pop())
print(s.size())

```

####简单括号匹配

区分括号是否匹配的能力是识别很多编程语言结构的重要部分。具有挑战的是如何编写一个算法，能够从左到右读取一串符号，并决定符号是否平衡。为了解决这个问题，我们需要做一个重要的观察。从左到右处理符号时，最近开始符号必须与下一个关闭符号相匹配。此外，处理的第一个开始符号必须等待直到其匹配最后一个符号。结束符号以相反的顺序匹配开始符号。他们从内到外匹配。这是一个可以用栈解决问题的线索。

```python
from pythonds.basic.stack import Stack

def parChecker(symbolString):
    s = Stack()
    balanced = True
    index = 0
    while index < len(symbolString) and balanced:
        symbol = symbolString[index]
        if symbol == "(":
            s.push(symbol)
        else:
            if s.isEmpty():
                balanced = False
            else:
                s.pop()

        index = index + 1

    if balanced and s.isEmpty():
        return True
    else:
        return False

print(parChecker('((()))'))
print(parChecker('(()'))
```

####符号匹配

匹配和嵌套不同种类的开始和结束符号的情况

```python
from pythonds.basic.stack import Stack

def parChecker(symbolString):
    s = Stack()
    balanced = True
    index = 0
    while index < len(symbolString) and balanced:
        symbol = symbolString[index]
        if symbol in "([{":
            s.push(symbol)
        else:
            if s.isEmpty():
                balanced = False
            else:
                top = s.pop()
                if not matches(top,symbol):
                       balanced = False
        index = index + 1
    if balanced and s.isEmpty():
        return True
    else:
        return False

def matches(open,close):
    opens = "([{"
    closers = ")]}"
    return opens.index(open) == closers.index(close)


print(parChecker('{{([][])}()}'))
print(parChecker('[{()]'))
```

####十进制转换二进制

```python
from pythonds.basic.stack import Stack

def divideBy2(decNumber):
    remstack = Stack()

    while decNumber > 0:
        rem = decNumber % 2
        remstack.push(rem)
        decNumber = decNumber // 2

    binString = ""
    while not remstack.isEmpty():
        binString = binString + str(remstack.pop())

    return binString

print(divideBy2(42))

```

指定基数

```python
from pythonds.basic.stack import Stack

def baseConverter(decNumber,base):
    digits = "0123456789ABCDEF"

    remstack = Stack()

    while decNumber > 0:
        rem = decNumber % base
        remstack.push(rem)
        decNumber = decNumber // base

    newString = ""
    while not remstack.isEmpty():
        newString = newString + digits[remstack.pop()]

    return newString

print(baseConverter(25,2))
print(baseConverter(25,16))
```

#### 中缀前缀和后缀表达式

| **Infix Expression** | **Prefix Expression** | **Postfix Expression** |
| -------------------- | --------------------- | ---------------------- |
| A + B                | + A B                 | A B +                  |
| A + B * C            | + A * B C             | A B C * +              |

| **Infix Expression** | **Prefix Expression** | **Postfix Expression** |
| -------------------- | --------------------- | ---------------------- |
| (A + B) * C          | * + A B C             | A B + C *              |

| **Infix Expression** | **Prefix Expression** | **Postfix Expression** |
| -------------------- | --------------------- | ---------------------- |
| A + B * C + D        | + + A * B C D         | A B C * + D +          |
| (A + B) * (C + D)    | * + A B + C D         | A B + C D + *          |
| A * B + C * D        | + * A B * C D         | A B * C D * +          |
| A + B + C + D        | + + + A B C D         | A B + C + D +          |

中缀表达式转换前缀和后缀

为了转换表达式，无论是对前缀还是后缀符号，先根据操作的顺序把表达式转换成完全括号表达式。然后将包含的运算符移动到左或右括号的位置，具体取决于需要前缀或后缀符号。

中缀转后缀通用法

1. 创建一个名为 opstack 的空栈以保存运算符。给输出创建一个空列表。
2. 通过使用字符串方法拆分将输入的中缀字符串转换为标记列表。
3. 从左到右扫描标记列表。
  如果标记是操作数，将其附加到输出列表的末尾。
  如果标记是左括号，将其压到 opstack 上。
  如果标记是右括号，则弹出 opstack，直到删除相应的左括号。将每个运算符附加到输出列表的末尾。
  如果标记是运算符， *，/，+ 或 - ，将其压入 opstack。但是，首先删除已经在opstack 中具有更高或相等优先级的任何运算符，并将它们加到输出列表中。
4. 当输入表达式被完全处理时，检查 opstack。仍然在栈上的任何运算符都可以删除并加到输出列表的末尾。

```python
from pythonds.basic.stack import Stack

def infixToPostfix(infixexpr):
    prec = {}
    prec["*"] = 3
    prec["/"] = 3
    prec["+"] = 2
    prec["-"] = 2
    prec["("] = 1
    opStack = Stack()
    postfixList = []
    tokenList = infixexpr.split()

    for token in tokenList:
        if token in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" or token in "0123456789":
            postfixList.append(token)
        elif token == '(':
            opStack.push(token)
        elif token == ')':
            topToken = opStack.pop()
            while topToken != '(':
                postfixList.append(topToken)
                topToken = opStack.pop()
        else:
            while (not opStack.isEmpty()) and \
               (prec[opStack.peek()] >= prec[token]):
                  postfixList.append(opStack.pop())
            opStack.push(token)

    while not opStack.isEmpty():
        postfixList.append(opStack.pop())
    return " ".join(postfixList)

print(infixToPostfix("A * B + C * D"))
print(infixToPostfix("( A + B ) * C - ( D - E ) * ( F + G )"))
```

后缀表达式求值

假设后缀表达式是一个由空格分隔的标记字符串。 运算符为 *，/，+ 和 - ，操作数假定为单个整数值。 输出将是一个整数结果。

1. 创建一个名为 operandStack 的空栈。
2. 拆分字符串转换为标记列表。
3. 从左到右扫描标记列表。
  如果标记是操作数，将其从字符串转换为整数，并将值压到operandStack。
  如果标记是运算符 *，/，+ 或 - ，它将需要两个操作数。弹出operandStack 两次。第一个弹出的是第二个操作数，第二个弹出的是第一个操作数。执行算术运算后，将结果压到操作数栈中。
4. 当输入的表达式被完全处理后，结果就在栈上，弹出 operandStack 并返回值。

```python
from pythonds.basic.stack import Stack

def postfixEval(postfixExpr):
    operandStack = Stack()
    tokenList = postfixExpr.split()

    for token in tokenList:
        if token in "0123456789":
            operandStack.push(int(token))
        else:
            operand2 = operandStack.pop()
            operand1 = operandStack.pop()
            result = doMath(token,operand1,operand2)
            operandStack.push(result)
    return operandStack.pop()

def doMath(op, op1, op2):
    if op == "*":
        return op1 * op2
    elif op == "/":
        return op1 / op2
    elif op == "+":
        return op1 + op2
    else:
        return op1 - op2

print(postfixEval('7 8 + 3 2 + /'))
```

### 队列

Queue() 创建一个空的新队列。 它不需要参数，并返回一个空队列。
enqueue(item) 将新项添加到队尾。 它需要 item 作为参数，并不返回任何内容。
dequeue() 从队首移除项。它不需要参数并返回 item。 队列被修改。
isEmpty() 查看队列是否为空。它不需要参数，并返回布尔值。
size() 返回队列中的项数。它不需要参数，并返回一个整数。

| **Queue Operation** | **Queue Contents**   | **Return Value** |
| ------------------- | -------------------- | ---------------- |
| `q.isEmpty()`       | `[]`                 | `True`           |
| `q.enqueue(4)`      | `[4]`                |                  |
| `q.enqueue('dog')`  | `['dog',4]`          |                  |
| `q.enqueue(True)`   | `[True,'dog',4]`     |                  |
| `q.size()`          | `[True,'dog',4]`     | `3`              |
| `q.isEmpty()`       | `[True,'dog',4]`     | `False`          |
| `q.enqueue(8.4)`    | `[8.4,True,'dog',4]` |                  |
| `q.dequeue()`       | `[8.4,True,'dog']`   | `4`              |
| `q.dequeue()`       | `[8.4,True]`         | `'dog'`          |
| `q.size()`          | `[8.4,True]`         | `2`              |

Python实现队列

```python
class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0,item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)
```

#### 模拟：烫手山芋

在这个游戏中，孩子们围成一个圈，并尽可能快的将一个山芋递给旁边的孩子。在某一个时间，动作结束，有山芋的孩子从圈中移除。游戏继续开始直到剩下最后一个孩子。

这个游戏相当于著名的约瑟夫问题。

```python
from pythonds.basic.queue import Queue

def hotPotato(namelist, num):
    simqueue = Queue()
    for name in namelist:
        simqueue.enqueue(name)

    while simqueue.size() > 1:
        for i in range(num):
            simqueue.enqueue(simqueue.dequeue())

        simqueue.dequeue()

    return simqueue.dequeue()

print(hotPotato(["Bill","David","Susan","Jane","Kent","Brad"],7))
```

#### 模拟：打印机

1. 创建打印任务的队列，每个任务都有个时间戳。队列启动的时候为空。
2. 每秒（currentSecond）：
  - 是否创建新的打印任务？如果是，将 currentSecond 作为时间戳添加到队列。
  - 如果打印机不忙并且有任务在等待
    - 从打印机队列中删除一个任务并将其分配给打印机
    - 从 currentSecond 中减去时间戳，以计算该任务的等待时间。
    - 将该任务的等待时间附件到列表中稍后处理。
    - 根据打印任务的页数，确定需要多少时间。
  - 打印机需要一秒打印，所以得从该任务的所需的等待时间减去一秒。
  - 如果任务已经完成，换句话说，所需的时间已经达到零，打印机空闲。
3. 模拟完成后，从生成的等待时间列表中计算平均等待时间。

为了设计此模拟，我们将为上述三个真实世界对象创建类： Printer , Task , PrintQueue
Printer 类需要跟踪它当前是否有任务。如果有，则它处于忙碌状态（13-17行），并且可以从任务的页数计算所需的时间。构造函数允许初始化每分钟页面的配置， tick 方法将内部定时器递减直到打印机设置为空闲(11 行)

```python
class Printer:
    def __init__(self, ppm):
        self.pagerate = ppm
        self.currentTask = None
        self.timeRemaining = 0

    def tick(self):
        if self.currentTask != None:
            self.timeRemaining = self.timeRemaining - 1
            if self.timeRemaining <= 0:
                self.currentTask = None

    def busy(self):
        if self.currentTask != None:
            return True
        else:
            return False

    def startNext(self,newtask):
        self.currentTask = newtask
        self.timeRemaining = newtask.getPages() * 60/self.pagerate
```

Task 类表示单个打印任务。创建任务时，随机数生成器将提供 1 到 20 页的长度。我们选择使用随机模块中的 randrange 函数。

每个任务还需要保存一个时间戳用于计算等待时间。此时间戳将表示任务被创建并放置到打印机队列中的时间。可以使用 waitTime 方法来检索在打印开始之前队列中花费的时间。

```python
import random

class Task:
    def __init__(self,time):
        self.timestamp = time
        self.pages = random.randrange(1,21)

    def getStamp(self):
        return self.timestamp

    def getPages(self):
        return self.pages

    def waitTime(self, currenttime):
        return currenttime - self.timestamp
```

PrintQueue 对象是我们现有队列 ADT 的一个实例。 newPrintTask 决定是否创建一个新的打印任务。我们再次选择使用随机模块的randrange 函数返回 1 到 180 之间的随机整数。打印任务每 180 秒到达一次。通过从随机整数（32 行）的范围中任意选择，我们可以模拟这个随机事件。模拟功能允许我们设置打印机的总时间和每分钟的页数。

```python
from pythonds.basic.queue import Queue

import random

def simulation(numSeconds, pagesPerMinute):

    labprinter = Printer(pagesPerMinute)
    printQueue = Queue()
    waitingtimes = []

    for currentSecond in range(numSeconds):

      if newPrintTask():
         task = Task(currentSecond)
         printQueue.enqueue(task)

      if (not labprinter.busy()) and (not printQueue.isEmpty()):
        nexttask = printQueue.dequeue()
        waitingtimes.append(nexttask.waitTime(currentSecond))
        labprinter.startNext(nexttask)

      labprinter.tick()

    averageWait=sum(waitingtimes)/len(waitingtimes)
    print("Average Wait %6.2f secs %3d tasks remaining."%(averageWait,printQueue.size()))

def newPrintTask():
    num = random.randrange(1,181)
    if num == 180:
        return True
    else:
        return False

for i in range(10):
    simulation(3600,5)
```

### Deque双端队列

Deque() 创建一个空的新 deque。它不需要参数，并返回空的 deque。
addFront(item) 将一个新项添加到 deque 的首部。它需要 item 参数 并不返回任何内容。
addRear(item) 将一个新项添加到 deque 的尾部。它需要 item 参数并不返回任何内容。
removeFront() 从 deque 中删除首项。它不需要参数并返回 item。deque 被修改。
removeRear() 从 deque 中删除尾项。它不需要参数并返回 item。deque 被修改。
isEmpty() 测试 deque 是否为空。它不需要参数，并返回布尔值。
size() 返回 deque 中的项数。它不需要参数，并返回一个整数。

| **Deque Operation** | **Deque Contents**         | **Return Value** |
| ------------------- | -------------------------- | ---------------- |
| `d.isEmpty()`       | `[]`                       | `True`           |
| `d.addRear(4)`      | `[4]`                      |                  |
| `d.addRear('dog')`  | `['dog',4,]`               |                  |
| `d.addFront('cat')` | `['dog',4,'cat']`          |                  |
| `d.addFront(True)`  | `['dog',4,'cat',True]`     |                  |
| `d.size()`          | `['dog',4,'cat',True]`     | `4`              |
| `d.isEmpty()`       | `['dog',4,'cat',True]`     | `False`          |
| `d.addRear(8.4)`    | `[8.4,'dog',4,'cat',True]` |                  |
| `d.removeRear()`    | `['dog',4,'cat',True]`     | `8.4`            |
| `d.removeFront()`   | `['dog',4,'cat']`          | `True`           |

Python实现Deque

```python
class Deque:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def addFront(self, item):
        self.items.append(item)

    def addRear(self, item):
        self.items.insert(0,item)

    def removeFront(self):
        return self.items.pop()

    def removeRear(self):
        return self.items.pop(0)

    def size(self):
        return len(self.items)
```

#### 回文检查

```python
from pythonds.basic.deque import Deque

def palchecker(aString):
    chardeque = Deque()

    for ch in aString:
        chardeque.addRear(ch)

    stillEqual = True

    while chardeque.size() > 1 and stillEqual:
        first = chardeque.removeFront()
        last = chardeque.removeRear()
        if first != last:
            stillEqual = False

    return stillEqual

print(palchecker("lsdkjfskf"))
print(palchecker("radar"))
```

### 无序列表抽象数据类型

List() 创建一个新的空列表。它不需要参数，并返回一个空列表。
add(item) 向列表中添加一个新项。它需要 item 作为参数，并不返回任何内容。假定该item 不在列表中。
remove(item) 从列表中删除该项。它需要 item 作为参数并修改列表。假设项存在于列表中。
search(item) 搜索列表中的项目。它需要 item 作为参数，并返回一个布尔值。
isEmpty() 检查列表是否为空。它不需要参数，并返回布尔值。
size（）返回列表中的项数。它不需要参数，并返回一个整数。
append(item) 将一个新项添加到列表的末尾，使其成为集合中的最后一项。它需要 item作为参数，并不返回任何内容。假定该项不在列表中。
index(item) 返回项在列表中的位置。它需要 item 作为参数并返回索引。假定该项在列表中。
insert(pos，item) 在位置 pos 处向列表中添加一个新项。它需要 item 作为参数并不返回任何内容。假设该项不在列表中，并且有足够的现有项使其有 pos 的位置。
pop() 删除并返回列表中的最后一个项。假设该列表至少有一个项。
pop(pos) 删除并返回位置 pos 处的项。它需要 pos 作为参数并返回项。假定该项在列表中。

#### 实现无序列表：链表

```python
class Node:
    def __init__(self,initdata):
        self.data = initdata
        self.next = None

    def getData(self):
        return self.data

    def getNext(self):
        return self.next

    def setData(self,newdata):
        self.data = newdata

    def setNext(self,newnext):
        self.next = newnext
```

Unordered List 类

```python
class UnorderedList:

    def __init__(self):
        self.head = None
        
    def isEmpty(self):
        return self.head == None
    
    def add(self,item):
        temp = Node(item)
        temp.setNext(self.head)
        self.head = temp
        
    def size(self):
        current = self.head
        count = 0
        while current != None:
            count = count + 1
            current = current.getNext()

        return count
    
    def search(self,item):
        current = self.head
        found = False
        while current != None and not found:
            if current.getData() == item:
                found = True
            else:
                current = current.getNext()

        return found
    
    def remove(self,item):
        current = self.head
        previous = None
        found = False
        while not found:
            if current.getData() == item:
                found = True
            else:
                previous = current
                current = current.getNext()

        if previous == None:
            self.head = current.getNext()
        else:
            previous.setNext(current.getNext())
```

### 有序列表抽象数据结构

OrderedList() 创建一个新的空列表。它不需要参数，并返回一个空列表。
add(item) 向列表中添加一个新项。它需要 item 作为参数，并不返回任何内容。假定该item 不在列表中。
remove(item) 从列表中删除该项。它需要 item 作为参数并修改列表。假设项存在于列表中。
search(item) 搜索列表中的项目。它需要 item 作为参数，并返回一个布尔值。
isEmpty() 检查列表是否为空。它不需要参数，并返回布尔值。
size（）返回列表中的项数。它不需要参数，并返回一个整数。
index(item) 返回项在列表中的位置。它需要 item 作为参数并返回索引。假定该项在列表中。
pop() 删除并返回列表中的最后一个项。假设该列表至少有一个项。
pop(pos) 删除并返回位置 pos 处的项。它需要 pos 作为参数并返回项。假定该项在列表中。

#### 实现有序列表

```python
class OrderedList:
    def __init__(self):
        self.head = None
        
    # 搜索
    def search(self,item):
        current = self.head
        found = False
        stop = False
        while current != None and not found and not stop:
            if current.getData() == item:
                found = True
            else:
                if current.getData() > item:
                    stop = True
                else:
                    current = current.getNext()

        return found
    
    # 添加元素
    def add(self,item):
        current = self.head
        previous = None
        stop = False
        while current != None and not stop:
            if current.getData() > item:
                stop = True
            else:
                previous = current
                current = current.getNext()

        temp = Node(item)
        if previous == None:
            temp.setNext(self.head)
            self.head = temp
        else:
            temp.setNext(current)
            previous.setNext(temp)
```

### 递归

#### 整数转换为任意进制字符串

假设你想将一个整数转换为一个二进制和十六进制字符串。例如，将整数 10 转换为十进制字符串表示为 10 ，或将其字符串表示为二进制 1010 。虽然有很多算法来解决这个问题，包括在栈部分讨论的算法，但递归的解决方法非常优雅。

```python
def toStr(n,base):
   convertString = "0123456789ABCDEF"
   if n < base:
      return convertString[n]
   else:
      return toStr(n//base,base) + convertString[n%base]

print(toStr(1453,16))
```

栈帧：实现递归
假设不是将递归调用的结果与来自 convertString 的字符串拼接到 toStr，我们修改了算法，以便在进行递归调用之前将字符串入栈

```python
from pythonds.basic.stack import Stack

rStack = Stack()

def toStr(n,base):
    convertString = "0123456789ABCDEF"
    while n > 0:
        if n < base:
            rStack.push(convertString[n])
        else:
            rStack.push(convertString[n % base])
        n = n // base
    res = ""
    while not rStack.isEmpty():
        res = res + str(rStack.pop())
    return res

print(toStr(1453,16))
```

#### 可视化递归

```python
import turtle

myTurtle = turtle.Turtle()
myWin = turtle.Screen()

def drawSpiral(myTurtle, lineLen):
    if lineLen > 0:
        myTurtle.forward(lineLen)
        myTurtle.right(90)  # 右转90度
        drawSpiral(myTurtle,lineLen-5)

drawSpiral(myTurtle,100)
myWin.exitonclick()  # 缩小窗口，使乌龟进入等待模式，直到用户单击窗口，程序清理并退出
```

分形树

分形的定义是，当你看着它时，无论你放大多少，分形有相同的基本形状。

```python
def tree(branchLen,t):
    if branchLen > 5:
        t.forward(branchLen)
        t.right(20)
        tree(branchLen-15,t) # 右树
        t.left(40)
        tree(branchLen-10,t)  # 左树
        t.right(20)
        t.backward(branchLen)
```

```python
import turtle

def tree(branchLen,t):
    if branchLen > 5:
        t.forward(branchLen)
        t.right(20)
        tree(branchLen-15,t)
        t.left(40)
        tree(branchLen-15,t)
        t.right(20)
        t.backward(branchLen)

def main():
    t = turtle.Turtle()
    myWin = turtle.Screen()
    t.left(90)
    t.up()
    t.backward(100)
    t.down()
    t.color("green")
    tree(75,t)
    myWin.exitonclick()

main()
```

#### 谢尔宾斯基三角形

谢尔宾斯基三角形阐明了三路递归算法。用手绘制谢尔宾斯基三角形的过程很简单。 从一个大三角形开始。通过连接每一边的中点，将这个大三角形分成四个新的三角形。忽略刚刚创建的中间三角形，对三个小三角形中的每一个应用相同的过程。 每次创建一组新的三角形时，都会将此过程递归应用于三个较小的角三角形。

基本情况被任意设置为我们想要将三角形划分成块的次数。有时我们把这个数字称为分形的“度”。 每次我们进行递归调用时，我们从度中减去 1，直到 0。当我们达到 0 度时，我们停止递归

```python
import turtle

def drawTriangle(points,color,myTurtle):
    myTurtle.fillcolor(color)
    myTurtle.up()
    myTurtle.goto(points[0][0],points[0][1])
    myTurtle.down()
    myTurtle.begin_fill()
    myTurtle.goto(points[1][0],points[1][1])
    myTurtle.goto(points[2][0],points[2][1])
    myTurtle.goto(points[0][0],points[0][1])
    myTurtle.end_fill()

def getMid(p1,p2):
    return ( (p1[0]+p2[0]) / 2, (p1[1] + p2[1]) / 2)

def sierpinski(points,degree,myTurtle):
    colormap = ['blue','red','green','white','yellow',
                'violet','orange']
    drawTriangle(points,colormap[degree],myTurtle)
    if degree > 0:
        sierpinski([points[0],
                        getMid(points[0], points[1]),
                        getMid(points[0], points[2])],
                   degree-1, myTurtle)
        sierpinski([points[1],
                        getMid(points[0], points[1]),
                        getMid(points[1], points[2])],
                   degree-1, myTurtle)
        sierpinski([points[2],
                        getMid(points[2], points[1]),
                        getMid(points[0], points[2])],
                   degree-1, myTurtle)

def main():
   myTurtle = turtle.Turtle()
   myWin = turtle.Screen()
   myPoints = [[-100,-50],[0,100],[100,-50]]
   sierpinski(myPoints,3,myTurtle)
   myWin.exitonclick()

main()
```

#### 汉诺塔游戏

这里是如何使用中间杆将塔从起始杆移动到目标杆的步骤：

1. 使用目标杆将 height-1 的塔移动到中间杆。
2. 将剩余的盘子移动到目标杆。
3. 使用起始杆将 height-1 的塔从中间杆移动到目标杆。

```python
def moveTower(height,fromPole, toPole, withPole):
    if height >= 1:
        moveTower(height-1,fromPole,withPole,toPole)
        moveDisk(fromPole,toPole)
        moveTower(height-1,withPole,toPole,fromPole)
```

```python
def moveDisk(fp,tp):
    print("moving disk from",fp,"to",tp)
```

#### 探索迷宫

为了使问题容易些，我们假设我们的迷宫被分成“正方形”。迷宫的每个正方形是开放的或被一段墙壁占据。乌龟只能通过迷宫的空心方块。 如果乌龟撞到墙上，它必须尝试不同的方向。
乌龟将需要一个程序，以找到迷宫的出路。这里是过程：

1. 从我们的起始位置，我们将首先尝试向北一格，然后从那里递归地尝试我们的程序。
2. 如果我们通过尝试向北作为第一步没有成功，我们将向南一格，并递归地重复我们的程序。
3. 如果向南也不行，那么我们将尝试向西一格，并递归地重复我们的程序。
4. 如果北，南和西都没有成功，则应用程序从当前位置递归向东。
5. 如果这些方向都没有成功，那么没有办法离开迷宫，我们失败。

假设我们第一步是向北走。按照我们的程序，我们的下一步也将是向北。但如果北面被一堵墙阻挡，我们必须看看程序的下一步，并试着向南。不幸的是，向南使我们回到原来的起点。如果我们从那里再次应用递归过程，我们将又回到向北一格，并陷入无限循环。所以，我们必须有一个策略来记住我们去过哪。在这种情况下，我们假设有一袋面包屑可以撒在我们走过的路上。如果我们沿某个方向迈出一步，发现那个位置上已经有面包屑，我们应该立即后退并尝试程序中的下一个方向。我们看看这个算法的代码，就像从递归函数调用返回一样简单。
正如我们对所有递归算法所做的一样，让我们回顾一下基本情况。其中一些你可能已经根据前一段的描述猜到了。在这种算法中，有四种基本情况要考虑：

1. 乌龟撞到了墙。由于这一格被墙壁占据，不能进行进一步的探索。
2. 乌龟找到一个已经探索过的格。我们不想继续从这个位置探索，否则会陷入循环。
3. 我们发现了一个外边缘，没有被墙壁占据。换句话说，我们发现了迷宫的一个出口。
4. 我们探索了一格在四个方向上都没有成功。

为了我们的程序工作，我们将需要有一种方式来表示迷宫。为了使这个更有趣，我们将使用turtle 模块来绘制和探索我们的迷宫，以使我们看到这个算法的功能。迷宫对象将提供以下方法让我们在编写搜索算法时使用：
__init__ 读取迷宫的数据文件，初始化迷宫的内部表示，并找到乌龟的起始位置。
drawMaze 在屏幕上的一个窗口中绘制迷宫。
updatePosition 更新迷宫的内部表示，并更改窗口中乌龟的位置。
isExit 检查当前位置是否是迷宫的退出位置。
Maze 类还重载索引运算符 [] ，以便我们的算法可以轻松访问任何特定格的状态。

```python
def searchFrom(maze, startRow, startColumn):
    maze.updatePosition(startRow, startColumn)
   #  Check for base cases:
   #  1. We have run into an obstacle, return false
   if maze[startRow][startColumn] == OBSTACLE :
        return False
    #  2. We have found a square that has already been explored
    if maze[startRow][startColumn] == TRIED:
        return False
    # 3. Success, an outside edge not occupied by an obstacle
    if maze.isExit(startRow,startColumn):
        maze.updatePosition(startRow, startColumn, PART_OF_PATH)
        return True
    maze.updatePosition(startRow, startColumn, TRIED)

    # Otherwise, use logical short circuiting to try each
    # direction in turn (if needed)
    found = searchFrom(maze, startRow-1, startColumn) or \
            searchFrom(maze, startRow+1, startColumn) or \
            searchFrom(maze, startRow, startColumn-1) or \
            searchFrom(maze, startRow, startColumn+1)
    if found:
        maze.updatePosition(startRow, startColumn, PART_OF_PATH)
    else:
        maze.updatePosition(startRow, startColumn, DEAD_END)
    return found
```

迷宫数据文件的内部表示

```
[ ['+','+','+','+',...,'+','+','+','+','+','+','+'],
  ['+',' ',' ',' ',...,' ',' ',' ','+',' ',' ',' '],
  ['+',' ','+',' ',...,'+','+',' ','+',' ','+','+'],
  ['+',' ','+',' ',...,' ',' ',' ','+',' ','+','+'],
  ['+','+','+',' ',...,'+','+',' ','+',' ',' ','+'],
  ['+',' ',' ',' ',...,'+','+',' ',' ',' ',' ','+'],
  ['+','+','+','+',...,'+','+','+','+','+',' ','+'],
  ['+',' ',' ',' ',...,'+','+',' ',' ','+',' ','+'],
  ['+',' ','+','+',...,' ',' ','+',' ',' ',' ','+'],
  ['+',' ',' ',' ',...,' ',' ','+',' ','+','+','+'],
  ['+','+','+','+',...,'+','+','+',' ','+','+','+']]
```

示例迷宫数据文件

```
++++++++++++++++++++++
+   +   ++ ++     +
+ +   +       +++ + ++
+ + +  ++  ++++   + ++
+++ ++++++    +++ +  +
+          ++  ++    +
+++++ ++++++   +++++ +
+     +   +++++++  + +
+ +++++++      S +   +
+                + +++
++++++++++++++++++ +++
```

```python
class Maze:
    def __init__(self,mazeFileName):
        rowsInMaze = 0
        columnsInMaze = 0
        self.mazelist = []
        mazeFile = open(mazeFileName,'r')
        rowsInMaze = 0
        for line in mazeFile:
            rowList = []
            col = 0
            for ch in line[:-1]:
                rowList.append(ch)
                if ch == 'S':
                    self.startRow = rowsInMaze
                    self.startCol = col
                col = col + 1
            rowsInMaze = rowsInMaze + 1
            self.mazelist.append(rowList)
            columnsInMaze = len(rowList)

        self.rowsInMaze = rowsInMaze
        self.columnsInMaze = columnsInMaze
        self.xTranslate = -columnsInMaze/2
        self.yTranslate = rowsInMaze/2
        self.t = Turtle(shape='turtle')
        setup(width=600,height=600)
        setworldcoordinates(-(columnsInMaze-1)/2-.5,
                            -(rowsInMaze-1)/2-.5,
                            (columnsInMaze-1)/2+.5,
                            (rowsInMaze-1)/2+.5)
```

```python
def drawMaze(self):
    for y in range(self.rowsInMaze):
        for x in range(self.columnsInMaze):
            if self.mazelist[y][x] == OBSTACLE:
                self.drawCenteredBox(x+self.xTranslate,
                                     -y+self.yTranslate,
                                     'tan')
    self.t.color('black','blue')

def drawCenteredBox(self,x,y,color):
    tracer(0)
    self.t.up()
    self.t.goto(x-.5,y-.5)
    self.t.color('black',color)
    self.t.setheading(90)
    self.t.down()
    self.t.begin_fill()
    for i in range(4):
        self.t.forward(1)
        self.t.right(90)
    self.t.end_fill()
    update()
    tracer(1)

def moveTurtle(self,x,y):
    self.t.up()
    self.t.setheading(self.t.towards(x+self.xTranslate,
                                     -y+self.yTranslate))
    self.t.goto(x+self.xTranslate,-y+self.yTranslate)

def dropBreadcrumb(self,color):
    self.t.dot(color)

def updatePosition(self,row,col,val=None):
    if val:
        self.mazelist[row][col] = val
    self.moveTurtle(col,row)

    if val == PART_OF_PATH:
        color = 'green'
    elif val == OBSTACLE:
        color = 'red'
    elif val == TRIED:
        color = 'black'
    elif val == DEAD_END:
        color = 'red'
    else:
        color = None

    if color:
        self.dropBreadcrumb(color)
```

```python
def isExit(self,row,col):
     return (row == 0 or
             row == self.rowsInMaze-1 or
             col == 0 or
             col == self.columnsInMaze-1 )

def __getitem__(self,idx):
     return self.mazelist[idx]
```

####动态规划

优化问题的典型例子包括使用最少的硬币找零。

```python
def recMC(coinValueList,change):
   minCoins = change
   if change in coinValueList:
     return 1
   else:
      for i in [c for c in coinValueList if c <= change]:
         numCoins = 1 + recMC(coinValueList,change-i)
         if numCoins < minCoins:
            minCoins = numCoins
   return minCoins

print(recMC([1,5,10,25],63))
```

上述方法低效

一个简单的解决方案是将最小数量的硬币的结果存储在表中

```python
def recDC(coinValueList,change,knownResults):
   minCoins = change
   if change in coinValueList:
      knownResults[change] = 1
      return 1
   elif knownResults[change] > 0:
      return knownResults[change]
   else:
       for i in [c for c in coinValueList if c <= change]:
         numCoins = 1 + recDC(coinValueList, change-i,
                              knownResults)
         if numCoins < minCoins:
            minCoins = numCoins
            knownResults[change] = minCoins
   return minCoins

print(recDC([1,5,10,25],63,[0]*64))
```

使用这个修改的算法减少了我们需要为四个硬币递归调用的数量

事实上，我们所做的不是动态规划，而是我们通过使用称为 记忆化 的技术来提高我们的程序的性能，或者更常见的叫做 缓存 。

```python
def dpMakeChange(coinValueList,change,minCoins):
   for cents in range(change+1):
      coinCount = cents
      for j in [c for c in coinValueList if c <= cents]:
            if minCoins[cents-j] + 1 < coinCount:
               coinCount = minCoins[cents-j]+1
      minCoins[cents] = coinCount
   return minCoins[change]
```

```python
def dpMakeChange(coinValueList,change,minCoins,coinsUsed):
   for cents in range(change+1):
      coinCount = cents
      newCoin = 1
      for j in [c for c in coinValueList if c <= cents]:
            if minCoins[cents-j] + 1 < coinCount:
               coinCount = minCoins[cents-j]+1
               newCoin = j
      minCoins[cents] = coinCount
      coinsUsed[cents] = newCoin
   return minCoins[change]

def printCoins(coinsUsed,change):
   coin = change
   while coin > 0:
      thisCoin = coinsUsed[coin]
      print(thisCoin)
      coin = coin - thisCoin

def main():
    amnt = 63
    clist = [1,5,10,21,25]
    coinsUsed = [0]*(amnt+1)
    coinCount = [0]*(amnt+1)

    print("Making change for",amnt,"requires")
    print(dpMakeChange(clist,amnt,coinCount,coinsUsed),"coins")
    print("They are:")
    printCoins(coinsUsed,amnt)
    print("The used list is as follows:")
    print(coinsUsed)

main()
```

### 搜索和排序

#### 顺序查找

```python
def sequentialSearch(alist, item):
	    pos = 0
	    found = False
	
	    while pos < len(alist) and not found:
	        if alist[pos] == item:
	            found = True
	        else:
	            pos = pos+1
	
	    return found
	
	testlist = [1, 2, 32, 8, 17, 19, 42, 13, 0]
	print(sequentialSearch(testlist, 3))
	print(sequentialSearch(testlist, 13))
```

| **Case**            | **Best Case** | **Worst Case** | **Average Case** |
| ------------------- | ------------- | -------------- | ---------------- |
| item is present     | 1             | n              | n/2              |
| item is not present | n             | n              | n                |

 ```python
def orderedSequentialSearch(alist, item):
	    pos = 0
	    found = False
	    stop = False
	    while pos < len(alist) and not found and not stop:
	        if alist[pos] == item:
	            found = True
	        else:
	            if alist[pos] > item:
	                stop = True
	            else:
	                pos = pos+1
	
	    return found
	
	testlist = [0, 1, 2, 8, 13, 17, 19, 32, 42,]
	print(orderedSequentialSearch(testlist, 3))
	print(orderedSequentialSearch(testlist, 13))
 ```

| item is present  | 1    | n    | n/2  |
| ---------------- | ---- | ---- | ---- |
| item not present | 1    | n    | n/2  |

 #### 二分查找

```python
def binarySearch(alist, item):
	    first = 0
	    last = len(alist)-1
	    found = False
	
	    while first<=last and not found:
	        midpoint = (first + last)//2
	        if alist[midpoint] == item:
	            found = True
	        else:
	            if item < alist[midpoint]:
	                last = midpoint-1
	            else:
	                first = midpoint+1
	
	    return found
	
	testlist = [0, 1, 2, 8, 13, 17, 19, 32, 42,]
	print(binarySearch(testlist, 3))
	print(binarySearch(testlist, 13))
```

```python
def binarySearch(alist, item):
	    if len(alist) == 0:
	        return False
	    else:
	        midpoint = len(alist)//2
	        if alist[midpoint]==item:
	          return True
	        else:
	          if item<alist[midpoint]:
	            return binarySearch(alist[:midpoint],item)
	          else:
	            return binarySearch(alist[midpoint+1:],item)
	
	testlist = [0, 1, 2, 8, 13, 17, 19, 32, 42,]
	print(binarySearch(testlist, 3))
	print(binarySearch(testlist, 13))
```

二分查找是 O( log^n )。

即使二分查找通常比顺序查找更好，但重要的是要注意，对于小的 n 值，排序的额外成本可能不值得。事实上，我们应该经常考虑采取额外的分类工作是否使搜索获得好处。如果我们可以排序一次，然后查找多次，排序的成本就不那么重要。然而，对于大型列表，一次排序可能是非常昂贵，从一开始就执行顺序查找可能是最好的选择。

#### Hash查找

给定项的集合，将每个项映射到唯一槽的散列函数被称为完美散列函数。如果我们知道项和集合将永远不会改变，那么可以构造一个完美的散列函数。不幸的是，给定任意的项集合，没有系统的方法来构建完美的散列函数。幸运的是，我们不需要散列函数是完美的，仍然可以提高性能。

我们的目标是创建一个散列函数，最大限度地减少冲突数，易于计算，并均匀分布在哈希表中的项。有很多常用的方法来扩展简单余数法。我们将在这里介绍其中几个。

分组求和法 将项划分为相等大小的块（最后一块可能不是相等大小）。然后将这些块加在一起以求出散列值。例如，如果我们的项是电话号码 436-555-4601 ，我们将取出数字，并将它们分成2位数 （43,65,55,46,01） 。 43 + 65 + 55 + 46 + 01 ，我们得到 210。我们假设哈希表有 11 个槽，那么我们需要除以 11 。在这种情况下， 210％11 为 1，因此电话号码 436-555-4601 散列到槽 1 。一些分组求和法会在求和之前每隔一个反转。对于上述示例，我们得到43 + 56 + 55 + 64 + 01 = 219 ，其给出 219％11 = 10 。

用于构造散列函数的另一数值技术被称为 平方取中法 。我们首先对该项平方，然后提取一部分数字结果。例如，如果项是 44，我们将首先计算 44^2 = 1,936 。通过提取中间两个数字93 ，我们得到 5（93％11） 。

我们还可以为基于字符的项（如字符串）创建哈希函数。 词 cat 可以被认为是 ascii 值的序列。

```python
>>> ord('c')
99
>>> ord('a')
97
>>> ord('t')
116
```

然后，我们可以获取这三个 ascii 值，将它们相加，并使用余数方法获取散列值.

展示了一个名为 hash 的函数，它接收字符串和表大小 作为参数，并返回从0 到 tablesize-1 的范围内的散列值。

```python
def hash(astring, tablesize):
    sum = 0
    for pos in range(len(astring)):
        sum = sum + ord(astring[pos])

    return sum%tablesize
```

有趣的是，当使用此散列函数时，字符串总是返回相同的散列值。 为了弥补这一点，我们可以使用字符的位置作为权重。

你可以思考一些其他方法来计算集合中项的散列值。重要的是要记住，哈希函数必须是高效的，以便它不会成为存储和搜索过程的主要部分。如果哈希函数太复杂，则计算槽名称的程序要比之前所述的简单地进行基本的顺序或二分搜索更耗时。 这将打破散列的目的。

**冲突解决**

解决冲突的一种方法是查找散列表，尝试查找到另一个空槽以保存导致冲突的项。一个简单的方法是从原始哈希值位置开始，然后以顺序方式移动槽，直到遇到第一个空槽。注意，我们可能需要回到第一个槽（循环）以查找整个散列表。这种冲突解决过程被称为开放寻址，因为它试图在散列表中找到下一个空槽或地址。通过系统地一次访问每个槽，我们执行称为线性探测的开放寻址技术。

一旦我们使用开放寻址和线性探测建立了哈希表，我们就必须使用相同的方法来搜索项。我们不能简单地返回 False，因为我们知道可能存在冲突。我们现在被迫做一个顺序搜索，从位置 10 开始寻找，直到我们找到项 20 或我们找到一个空槽。

线性探测的缺点是聚集的趋势;项在表中聚集。这意味着如果在相同的散列值处发生很多冲突，则将通过线性探测来填充多个周边槽。

处理聚集的一种方式是扩展线性探测技术，使得不是顺序地查找下一个开放槽，而是跳过槽，从而更均匀地分布已经引起冲突的项。这将潜在地减少发生的聚集。

在冲突后寻找另一个槽的过程叫 重新散列 。使用简单的线性探测，rehash 函数是newhashvalue = rehash(oldhashvalue) 其中 rehash(pos)=(pos + 1)％sizeoftable 。 加3 rehash 可以定义为 rehash(pos)=(pos + 3)％sizeoftable 。一般来说， rehash(pos)=(pos +skip)％sizeoftable 。重要的是要注意，“跳过”的大小必须使得表中的所有槽最终都被访问。否则，表的一部分将不被使用。为了确保这一点，通常建议表大小是素数。

线性探测思想的一个变种称为二次探测。代替使用常量 “跳过” 值，我们使用rehash 函数，将散列值递增 1，3，5，7，9， 依此类推。这意味着如果第一哈希值是 h ，则连续值是 h + 1，h + 4，h + 9，h + 16 ，等等。换句话说，二次探测使用由连续完全正方形组成的跳跃。



用于处理冲突问题的替代方法是允许每个槽保持对项的集合（或链）的引用。链接允许许多项存在于哈希表中的相同位置。当发生冲突时，项仍然放在散列表的正确槽中。随着越来越多的项哈希到相同的位置，搜索集合中的项的难度增加。

当我们要搜索一个项时，我们使用散列函数来生成它应该在的槽。由于每个槽都有一个集合，我们使用一种搜索技术来查找该项是否存在。优点是，平均来说，每个槽中可能有更少的项，因此搜索可能更有效.

#####实现 map 抽象数据类型

最有用的 Python 集合之一是字典。回想一下，字典是一种关联数据类型，你可以在其中存储键-值对。该键用于查找关联的值。我们经常将这个想法称为 map 。

Map() 创建一个新的 map 。它返回一个空的 map 集合。
put(key，val) 向 map 中添加一个新的键值对。如果键已经在 map 中，那么用新值替换旧值。
get(key) 给定一个键，返回存储在 map 中的值或 None。
del 使用 del map[key] 形式的语句从 map 中删除键值对。
len() 返回存储在 map 中的键值对的数量。
in 返回 True 对于 key in map 语句，如果给定的键在 map 中，否则为False。

字典一个很大的好处是，给定一个键，我们可以非常快速地查找相关的值。为了提供这种快速查找能力，我们需要一个支持高效搜索的实现。我们可以使用具有顺序或二分查找的列表，但是使用如上所述的哈希表将更好，因为查找哈希表中的项可以接近 O(1) 性能。

```python
class HashTable:
    def __init__(self):
        self.size = 11
        self.slots = [None] * self.size
        self.data = [None] * self.size
```

```python
def put(self,key,data):
  hashvalue = self.hashfunction(key,len(self.slots))

  if self.slots[hashvalue] == None:
    self.slots[hashvalue] = key
    self.data[hashvalue] = data
  else:
    if self.slots[hashvalue] == key:
      self.data[hashvalue] = data  #replace
    else:
      nextslot = self.rehash(hashvalue,len(self.slots))
      while self.slots[nextslot] != None and \
                      self.slots[nextslot] != key:
        nextslot = self.rehash(nextslot,len(self.slots))

      if self.slots[nextslot] == None:
        self.slots[nextslot]=key
        self.data[nextslot]=data
      else:
        self.data[nextslot] = data #replace

def hashfunction(self,key,size):
     return key%size

def rehash(self,oldhash,size):
    return (oldhash+1)%size
```

```python
def get(self,key):
  startslot = self.hashfunction(key,len(self.slots))

  data = None
  stop = False
  found = False
  position = startslot
  while self.slots[position] != None and  \
                       not found and not stop:
     if self.slots[position] == key:
       found = True
       data = self.data[position]
     else:
       position=self.rehash(position,len(self.slots))
       if position == startslot:
           stop = True
  return data

def __getitem__(self,key):
    return self.get(key)

def __setitem__(self,key,data):
    self.put(key,data)
```

```python
>>> H=HashTable()
>>> H[54]="cat"
>>> H[26]="dog"
>>> H[93]="lion"
>>> H[17]="tiger"
>>> H[77]="bird"
>>> H[31]="cow"
>>> H[44]="goat"
>>> H[55]="pig"
>>> H[20]="chicken"
>>> H.slots
[77, 44, 55, 20, 26, 93, 17, None, None, 31, 54]
>>> H.data
['bird', 'goat', 'pig', 'chicken', 'dog', 'lion',
       'tiger', None, None, 'cow', 'cat']

>>> H[20]
'chicken'
>>> H[17]
'tiger'
>>> H[20]='duck'
>>> H[20]
'duck'
>>> H.data
['bird', 'goat', 'pig', 'duck', 'dog', 'lion',
'tiger', None, None, 'cow', 'cat']
>> print(H[99])
None
```



#### 冒泡排序

冒泡排序需要多次遍历列表。它比较相邻的项并交换那些无序的项。每次遍历列表将下一个最大的值放在其正确的位置。实质上，每个项“冒泡”到它所属的位置。

```python
def bubbleSort(alist):
    for passnum in range(len(alist)-1,0,-1):
        for i in range(passnum):
            if alist[i]>alist[i+1]:
                temp = alist[i]
                alist[i] = alist[i+1]
                alist[i+1] = temp

alist = [54,26,93,17,77,31,44,55,20]
bubbleSort(alist)
print(alist)
```

在最好的情况下，如果列表已经排序，则不会进行交换。 但是，在最坏的情况下，每次比较都会导致交换元素。 平均来说，我们交换了一半时间。

冒泡排序通常被认为是最低效的排序方法，因为它必须在最终位置被知道之前交换项。 这些“浪费”的交换操作是非常昂贵的。 然而，因为冒泡排序遍历列表的整个未排序部分，它有能力做大多数排序算法不能做的事情。特别地，如果在遍历期间没有交换，则我们知道该列表已排序。 如果发现列表已排序，可以修改冒泡排序提前停止。这意味着对于只需要遍历几次列表，冒泡排序具有识别排序列表和停止的优点。

短冒泡排序

```python
def shortBubbleSort(alist):
    exchanges = True
    passnum = len(alist)-1
    while passnum > 0 and exchanges:
       exchanges = False
       for i in range(passnum):
           if alist[i]>alist[i+1]:
               exchanges = True
               temp = alist[i]
               alist[i] = alist[i+1]
               alist[i+1] = temp
       passnum = passnum-1

alist=[20,30,40,90,50,60,70,80,100,110]
shortBubbleSort(alist)
print(alist)
```

#### 选择排序

选择排序改进了冒泡排序，每次遍历列表只做一次交换。为了做到这一点，一个选择排序在他遍历时寻找最大的值，并在完成遍历后，将其放置在正确的位置。与冒泡排序一样，在第一次遍历后，最大的项在正确的地方。 第二遍后，下一个最大的就位。遍历 n-1 次排序 n 个项，因为最终项必须在第（n-1）次遍历之后。

```python
def selectionSort(alist):
   for fillslot in range(len(alist)-1,0,-1):
       positionOfMax=0
       for location in range(1,fillslot+1):
           if alist[location]>alist[positionOfMax]:
               positionOfMax = location

       temp = alist[fillslot]
       alist[fillslot] = alist[positionOfMax]
       alist[positionOfMax] = temp

alist = [54,26,93,17,77,31,44,55,20]
selectionSort(alist)
print(alist)
```

你可能会看到选择排序与冒泡排序有相同数量的比较，因此也是 O(n^2 )。 然而，由于交换数量的减少，选择排序通常在基准研究中执行得更快。 事实上，对于我们的列表，冒泡排序有20 次交换，而选择排序只有 8 次。

#### 插入排序

插入排序，尽管仍然是 O(n^2 )，工作方式略有不同。它始终在列表的较低位置维护一个排序的子列表。然后将每个新项 “插入” 回先前的子列表，使得排序的子列表称为较大的一个项。

插入排序的最大比较次数是 n-1 个整数的总和。同样，是 O(n^2 )。然而，在最好的情况下，每次通过只需要进行一次比较。这是已经排序的列表的情况。
关于移位和交换的一个注意事项也很重要。通常，移位操作只需要交换大约三分之一的处理工作，因为仅执行一次分配。在基准研究中，插入排序有非常好的性能。

```python
def insertionSort(alist):
   for index in range(1,len(alist)):

     currentvalue = alist[index]
     position = index

     while position>0 and alist[position-1]>currentvalue:
         alist[position]=alist[position-1]
         position = position-1

     alist[position]=currentvalue

alist = [54,26,93,17,77,31,44,55,20]
insertionSort(alist)
print(alist)
```

#### 希尔排序

希尔排序（有时称为“递减递增排序”）通过将原始列表分解为多个较小的子列表来改进插入排序，每个子列表使用插入排序进行排序。 选择这些子列表的方式是希尔排序的关键。不是将列表拆分为连续项的子列表，希尔排序使用增量i（有时称为 gap ），通过选择 i 个项的所有项来创建子列表。

```python
def shellSort(alist):
    sublistcount = len(alist)//2
    while sublistcount > 0:

      for startposition in range(sublistcount):
        gapInsertionSort(alist,startposition,sublistcount)

      print("After increments of size",sublistcount,
                                   "The list is",alist)

      sublistcount = sublistcount // 2

def gapInsertionSort(alist,start,gap):
    for i in range(start+gap,len(alist),gap):

        currentvalue = alist[i]
        position = i

        while position>=gap and alist[position-gap]>currentvalue:
            alist[position]=alist[position-gap]
            position = position-gap

        alist[position]=currentvalue

alist = [54,26,93,17,77,31,44,55,20]
shellSort(alist)
print(alist)
```

乍一看，你可能认为希尔排序不会比插入排序更好，因为它最后一步执行了完整的插入排序。 然而，结果是，该最终插入排序不需要进行非常多的比较（或移位），因为如上所述，该列表已经被较早的增量插入排序预排序。 换句话说，每个遍历产生比前一个“更有序”的列表。 这使得最终遍历非常有效。

虽然对希尔排序的一般分析远远超出了本文的范围，我们可以说，它倾向于落在 O(n) 和O(n^2 ) 之间的某处，基于以上所描述的行为。对于 上述中显示的增量，性能为 O(n^2 )。 通过改变增量，例如使用 2^k -1（1,3,7,15,31等等） ，希尔排序可以在 O（n^3/2 ）处执行。

#### 归并排序

我们现在将注意力转向使用分而治之策略作为提高排序算法性能的一种方法。 我们将研究的第一个算法是归并排序。归并排序是一种递归算法，不断将列表拆分为一半。 如果列表为空或有一个项，则按定义（基本情况）进行排序。如果列表有多个项，我们分割列表，并递归调用两个半部分的合并排序。 一旦对这两半排序完成，就执行称为合并的基本操作。合并是获取两个较小的排序列表并将它们组合成单个排序的新列表的过程。

```python
def mergeSort(alist):
    print("Splitting ",alist)
    if len(alist)>1:
        mid = len(alist)//2
        lefthalf = alist[:mid]
        righthalf = alist[mid:]

        mergeSort(lefthalf)
        mergeSort(righthalf)

        i=0
        j=0
        k=0
        while i < len(lefthalf) and j < len(righthalf):
            if lefthalf[i] < righthalf[j]:
                alist[k]=lefthalf[i]
                i=i+1
            else:
                alist[k]=righthalf[j]
                j=j+1
            k=k+1

        while i < len(lefthalf):
            alist[k]=lefthalf[i]
            i=i+1
            k=k+1

        while j < len(righthalf):
            alist[k]=righthalf[j]
            j=j+1
            k=k+1
    print("Merging ",alist)

alist = [54,26,93,17,77,31,44,55,20]
mergeSort(alist)
print(alist)
```

为了分析 mergeSort 函数，我们需要考虑组成其实现的两个不同的过程。首先，列表被分成两半。我们已经计算过（在二分查找中）将列表划分为一半需要 log^n 次，其中 n 是列表的长度。第二个过程是合并。列表中的每个项将最终被处理并放置在排序的列表上。因此，大小为 n 的列表的合并操作需要 n 个操作。此分析的结果是 log^n 的拆分，其中每个操作花费n，总共 nlog^n 。归并排序是一种 O(nlogn) 算法。
回想切片 是 O(k)，其中 k 是切片的大小。为了保证 mergeSort 是 O(nlog^n )，我们将需要删除 slice 运算符。这是可能的，如果当我们进行递归调用，我们简单地传递开始和结束索引与列表。
重要的是注意，mergeSort 函数需要额外的空间来保存两个半部分，因为它们是使用切片操作提取的。如果列表很大，这个额外的空间可能是一个关键因素，并且在处理大型数据集时可能会导致此类问题。

#### 快速排序

快速排序使用分而治之来获得与归并排序相同的优点，而不使用额外的存储。然而，作为权衡，有可能列表不能被分成两半。当这种情况发生时，我们将看到性能降低。

快速排序首先选择一个值，该值称为 枢轴值 。虽然有很多不同的方法来选择枢轴值，我们将使用列表中的第一项。枢轴值的作用是帮助拆分列表。枢轴值属于最终排序列表（通常称为拆分点）的实际位置，将用于将列表划分为快速排序的后续调用。

分区从通过在列表中剩余项目的开始和结束处定位两个位置标记（我们称为左标记和右标记）开始。分区的目标是移动相对于枢轴值位于错误侧的项，同时也收敛于分裂点。

我们首先增加左标记，直到我们找到一个大于枢轴值的值。 然后我们递减右标，直到我们找到小于枢轴值的值。我们发现了两个相对于最终分裂点位置不适当的项。现在我们可以交换这两个项目，然后重复该过程。

在右标变得小于左标记的点，我们停止。右标记的位置现在是分割点。枢轴值可以与拆分点的内容交换，枢轴值现在就位。 此外，分割点左侧的所有项都小于枢轴值，分割点右侧的所有项都大于枢轴值。现在可以在分割点处划分列表，并且可以在两半上递归调用快速排序。

```python
def quickSort(alist):
   quickSortHelper(alist,0,len(alist)-1)

def quickSortHelper(alist,first,last):
   if first<last:

       splitpoint = partition(alist,first,last)

       quickSortHelper(alist,first,splitpoint-1)
       quickSortHelper(alist,splitpoint+1,last)


def partition(alist,first,last):
   pivotvalue = alist[first]

   leftmark = first+1
   rightmark = last

   done = False
   while not done:

       while leftmark <= rightmark and alist[leftmark] <= pivotvalue:
           leftmark = leftmark + 1

       while alist[rightmark] >= pivotvalue and rightmark >= leftmark:
           rightmark = rightmark -1

       if rightmark < leftmark:
           done = True
       else:
           temp = alist[leftmark]
           alist[leftmark] = alist[rightmark]
           alist[rightmark] = temp

   temp = alist[first]
   alist[first] = alist[rightmark]
   alist[rightmark] = temp


   return rightmark

alist = [54,26,93,17,77,31,44,55,20]
quickSort(alist)
print(alist)
```



对于长度为 n 的列表，如果分区总是出现在列表中间，则
会再次出现 log⁡n 分区。为了找到分割点，需要针对枢轴值检查 n 个项中的每一个。结果是nlog⁡n。此外，在归并排序过程中不需要额外的存储器。

不幸的是，在最坏的情况下，分裂点可能不在中间，并且可能非常偏向左边或右边，留下非常不均匀的分割。在这种情况下，对 n 个项的列表进行排序划分为对0 个项的列表和 n-1 个项目的列表进行排序。然后将 n-1 个划分的列表排序为大小为0的列表和大小为 n-2 的列表，以此类推。结果是具有递归所需的所有开销的 O(n) 排序。

我们之前提到过，有不同的方法来选择枢纽值。特别地，我们可以尝试通过使用称为中值三的技术来减轻一些不均匀分割的可能性。要选择枢轴值，我们将考虑列表中的第一个，中间和最后一个元素。

想法是，在列表中的第一项不属于列表的中间的情况下，中值三将选择更好的“中间”值。当原始列表部分有序时，这将特别有用。

### 树

节点
节点是树的基本部分。它可以有一个名称，我们称之为“键”。节点也可以有附加信息。我们将这个附加信息称为“有效载荷”。虽然有效载荷信息不是许多树算法的核心，但在利用树的应用中通常是关键的。

边
边是树的另一个基本部分。边连接两个节点以显示它们之间存在关系。每个节点（除根之外）都恰好从另一个节点的传入连接。每个节点可以具有多个输出边。

根
树的根是树中唯一没有传入边的节点。

路径
路径是由边连接节点的有序列表。

子节点
具有来自相同传入边的节点 c 的集合称为该节点的子节点

父节点
具有和它相同传入边的所连接的节点称为父节点

兄弟
树中作为同一父节点的子节点的节点被称为兄弟节点

子树
子树是由父节点和该父节点的所有后代组成的一组节点和边

叶节点
叶节点是没有子节点的节点。

层数
节点 n 的层数为从根结点到该结点所经过的分支数目。

高度
树的高度等于树中任何节点的最大层数。

定义一：树由一组节点和一组连接节点的边组成。树具有以下属性：

- 树的一个节点被指定为根节点。
- 除了根节点之外，每个节点 n 通过一个其他节点 p 的边连接，其中 p 是 n 的父节点。
- 从根路径遍历到每个节点路径唯一。
- 如果树中的每个节点最多有两个子节点，我们说该树是一个二叉树。

定义二：树是空的，或者由一个根节点和零个或多个子树组成，每个子树也是一棵树。每个子树的根节点通过边连接到父树的根节点。

#### 列表表示

在列表树的列表中，我们将根节点的值存储为列表的第一个元素。列表的第二个元素本身将是一个表示左子树的列表。列表的第三个元素将是表示右子树的另一个列表。

该列表方法的一个非常好的属性是表示子树的列表的结构遵守树定义的结构; 结构本身是递归的！具有根值和两个空列表的子树是叶节点。列表方法的另一个很好的特性是它可以推广到一个有许多子树的树。在树超过二叉树的情况下，另一个子树只是另一个列表。

```python
def BinaryTree(r):
    return [r, [], []]

def insertLeft(root,newBranch):
    t = root.pop(1)
    if len(t) > 1:
        root.insert(1,[newBranch,t,[]])
    else:
        root.insert(1,[newBranch, [], []])
    return root

def insertRight(root,newBranch):
    t = root.pop(2)
    if len(t) > 1:
        root.insert(2,[newBranch,[],t])
    else:
        root.insert(2,[newBranch,[],[]])
    return root

def getRootVal(root):
    return root[0]

def setRootVal(root,newVal):
    root[0] = newVal

def getLeftChild(root):
    return root[1]

def getRightChild(root):
    return root[2]
```

注意，要插入一个左子节点，我们首先获得与当前左子节点对应的（可能为空的）列表。然后我们添加新的左子树，添加旧的左子数作为新子节点的左子节点。这允许我们在任何位置将新节点拼接到树中。

#### 节点表示

第二种表示树的方法使用节点和引用。在这种情况下，我们将定义一个具有根值属性的类，以及左和右子树。

```python
class BinaryTree:
    def __init__(self,rootObj):
        self.key = rootObj
        self.leftChild = None
        self.rightChild = None

    def insertLeft(self,newNode):
        if self.leftChild == None:
            self.leftChild = BinaryTree(newNode)
        else:
            t = BinaryTree(newNode)
            t.leftChild = self.leftChild
            self.leftChild = t

    def insertRight(self,newNode):
        if self.rightChild == None:
            self.rightChild = BinaryTree(newNode)
        else:
            t = BinaryTree(newNode)
            t.rightChild = self.rightChild
            self.rightChild = t


    def getRightChild(self):
        return self.rightChild

    def getLeftChild(self):
        return self.leftChild

    def setRootVal(self,obj):
        self.key = obj

    def getRootVal(self):
        return self.key


r = BinaryTree('a')
print(r.getRootVal())
print(r.getLeftChild())
r.insertLeft('b')
print(r.getLeftChild())
print(r.getLeftChild().getRootVal())
r.insertRight('c')
print(r.getRightChild())
print(r.getRightChild().getRootVal())
r.getRightChild().setRootVal('hello')
print(r.getRightChild().getRootVal())
```

#### 分析树

分析树可以用于表示诸如句子或数学表达式的真实世界构造。

构建分析树的第一步是将表达式字符串拆分成符号列表。 有四种不同的符号要考虑：左括号，右括号，运算符和操作数。 我们知道，每当我们读一个左括号，我们开始一个新的表达式，因此我们应该创建一个新的树来对应于该表达式。 相反，每当我们读一个右括号，我们就完成了一个表达式。 我们还知道操作数将是叶节点和它们的操作符的子节点。 最后，我们知道每个操作符都将有一个左和右孩子。

使用上面的信息，我们可以定义四个规则如下：
如果当前符号是 '(' ，添加一个新节点作为当前节点的左子节点，并下降到左子节点。
如果当前符号在列表 ['+'，' - '，'/'，'*'] 中，请将当前节点的根值设置为由当前符号表示的运算符。 添加一个新节点作为当前节点的右子节点，并下降到右子节点。
如果当前符号是数字，请将当前节点的根值设置为该数字并返回到父节点。
如果当前令牌是 ')' ，则转到当前节点的父节点。

很明显，我们需要跟踪当前节点以及当前节点的父节点。树接口为我们提供了一种通过 getLeftChild 和 getRightChild 方法获取节点的子节点的方法，但是我们如何跟踪父节点呢？当我们遍历树时，保持跟踪父对象的简单解决方案是使用栈。每当我们想下降到当前节点的子节点时，我们首先将当前节点入到栈上。当我们想要返回到当前节点的父节点时，我们将父节点从栈中弹出。

```python
from pythonds.basic.stack import Stack
from pythonds.trees.binaryTree import BinaryTree

def buildParseTree(fpexp):
    fplist = fpexp.split()
    pStack = Stack()
    eTree = BinaryTree('')
    pStack.push(eTree)
    currentTree = eTree
    for i in fplist:
        if i == '(':
            currentTree.insertLeft('')
            pStack.push(currentTree)
            currentTree = currentTree.getLeftChild()
        elif i not in ['+', '-', '*', '/', ')']:
            currentTree.setRootVal(int(i))
            parent = pStack.pop()
            currentTree = parent
        elif i in ['+', '-', '*', '/']:
            currentTree.setRootVal(i)
            currentTree.insertRight('')
            pStack.push(currentTree)
            currentTree = currentTree.getRightChild()
        elif i == ')':
            currentTree = pStack.pop()
        else:
            raise ValueError
    return eTree

pt = buildParseTree("( ( 10 + 5 ) * 3 )")
pt.postorder()  #defined and explained in the next section
```



```python
def evaluate(parseTree):
    opers = {'+':operator.add, '-':operator.sub, '*':operator.mul, '/':operator.truediv}

    leftC = parseTree.getLeftChild()
    rightC = parseTree.getRightChild()

    if leftC and rightC:
        fn = opers[parseTree.getRootVal()]
        return fn(evaluate(leftC),evaluate(rightC))
    else:
        return parseTree.getRootVal()
```

#### 树的遍历

前序 在前序遍历中，我们首先访问根节点，然后递归地做左侧子树的前序遍历，随后是右侧子树的递归前序遍历。 中序 在一个中序遍历中，我们递归地对左子树进行一次遍历，访问根节点，最后递归遍历右子树。 后序 在后序遍历中，我们递归地对左子树和右子树进行后序遍历，然后访问根节点。

```python
# 作为外部函数
def preorder(tree):
    if tree:
        print(tree.getRootVal())
        preorder(tree.getLeftChild())
        preorder(tree.getRightChild())
# 作为 BinaryTree 类        
def preorder(self):
    print(self.key)
    if self.leftChild:
        self.leftChild.preorder()
    if self.rightChild:
        self.rightChild.preorder()      
```

以上哪种方式实现前序最好？ 答案是在这种情况下，实现前序作为外部函数可能更好。原因是你很少只是想遍历树。在大多数情况下，将要使用其中一个基本的遍历模式来完成其他任务。 事实上，我们将在下面的例子中看到后序遍历模式与我们前面编写的用于计算分析树的代码非常接近。 因此，我们用外部函数实现其余的遍历。

```python
def postorder(tree):
    if tree != None:
        postorder(tree.getLeftChild())
        postorder(tree.getRightChild())
        print(tree.getRootVal())
        
def postordereval(tree):
    opers = {'+':operator.add, '-':operator.sub, '*':operator.mul, '/':operator.truediv}
    res1 = None
    res2 = None
    if tree:
        res1 = postordereval(tree.getLeftChild())
        res2 = postordereval(tree.getRightChild())
        if res1 and res2:
            return opers[tree.getRootVal()](res1,res2)
        else:
            return tree.getRootVal()
```

```python
def inorder(tree):
  if tree != None:
      inorder(tree.getLeftChild())
      print(tree.getRootVal())
      inorder(tree.getRightChild())
      
def printexp(tree):
  sVal = ""
  if tree:
      sVal = '(' + printexp(tree.getLeftChild())
      sVal = sVal + str(tree.getRootVal())
      sVal = sVal + printexp(tree.getRightChild())+')'
  return sVal
```

#### 基于二叉堆的优先队列

队列的一个重要变种称为优先级队列。优先级队列的作用就像一个队列，你可以通过从前面删除一个项目来出队。然而，在优先级队列中，队列中的项的逻辑顺序由它们的优先级确定。最高优先级项在队列的前面，最低优先级的项在后面。因此，当你将项排入优先级队列时，新项可能会一直移动到前面。

你可能想到了几种简单的方法使用排序函数和列表实现优先级队列。然而，插入列表是 O(n)并且排序列表是 O(nlogn)。我们可以做得更好。实现优先级队列的经典方法是使用称为二叉堆的数据结构。二叉堆将允许我们在 O(logn) 中排队和取出队列。

我们的二叉堆实现的基本操作如下：

- BinaryHeap() 创建一个新的，空的二叉堆。
- insert(k) 向堆添加一个新项。
- findMin() 返回具有最小键值的项，并将项留在堆中。
- delMin() 返回具有最小键值的项，从堆中删除该项。
- 如果堆是空的，isEmpty() 返回 true，否则返回 false。
- size() 返回堆中的项数。
- buildHeap(list) 从键列表构建一个新的堆。

```python
from pythonds.trees.binheap import BinHeap

bh = BinHeap()
bh.insert(5)
bh.insert(7)
bh.insert(3)
bh.insert(11)

print(bh.delMin())

print(bh.delMin())

print(bh.delMin())

print(bh.delMin())
```

为了使我们的堆有效地工作，我们将利用二叉树的对数性质来表示我们的堆。 为了保证对数性能，我们必须保持树平衡。平衡二叉树在根的左和右子树中具有大致相同数量的节点。 在我们的堆实现中，我们通过创建一个 完整二叉树 来保持树平衡。 一个完整的二叉树是一个树，其中每个层都有其所有的节点，除了树的最底层，从左到右填充。

完整二叉树的另一个有趣的属性是，我们可以使用单个列表来表示它。 我们不需要使用节点和引用，甚至列表的列表。因为树是完整的，父节点的左子节点（在位置 p 处）是在列表中位置 2p 中找到的节点。 类似地，父节点的右子节点在列表中的位置 2p + 1。为了找到树中任意节点的父节点，我们可以简单地使用Python 的整数除法。 假定节点在列表中的位置 n，则父节点在位置 n/2。树的列表表示以及完整的结构属性允许我们仅使用几个简单的数学运算来高效地遍历一个完整的二叉树。 我们将看到，这也是我们的二叉堆的有效实现。

我们用于堆中存储项的方法依赖于维护堆的排序属性。 堆的排序属性如下：在堆中，对于具有父 p 的每个节点 x，p 中的键小于或等于 x 中的键。



```python
class BinHeap:
    def __init__(self):
        # 这个零只是放那里，用于以后简单的整数除法
        self.heapList = [0]
        self.currentSize = 0
        
    def percUp(self,i):
        # 获取父节点
        while i // 2 > 0:
          if self.heapList[i] < self.heapList[i // 2]:
             tmp = self.heapList[i // 2]
             self.heapList[i // 2] = self.heapList[i]
             self.heapList[i] = tmp
          i = i // 2
        
    def insert(self,k):
        self.heapList.append(k)
        self.currentSize = self.currentSize + 1
        self.percUp(self.currentSize)
        
        
    def percDown(self,i):
    while (i * 2) <= self.currentSize:
        mc = self.minChild(i)
        if self.heapList[i] > self.heapList[mc]:
            tmp = self.heapList[i]
            self.heapList[i] = self.heapList[mc]
            self.heapList[mc] = tmp
        i = mc

    def minChild(self,i):
        if i * 2 + 1 > self.currentSize:
            return i * 2
        else:
            if self.heapList[i*2] < self.heapList[i*2+1]:
                return i * 2
            else:
                return i * 2 + 1
            
    def delMin(self):
        retval = self.heapList[1]
        self.heapList[1] = self.heapList[self.currentSize]
        self.currentSize = self.currentSize - 1
        self.heapList.pop()
        self.percDown(1)
        return retval
```

delMin 的难点在根被删除后恢复堆结构和堆顺序属性。 我们可以分两步恢复我们的堆。首先，我们将通过获取列表中的最后一个项并将其移动到根位置来恢复根项，保持我们的堆结构属性。 但是，我们可能已经破坏了我们的二叉堆的堆顺序属性。 第二，我们通过将新的根节点沿着树向下推到其正确位置来恢复堆顺序属性。

为了维护堆顺序属性，我们所需要做的是将根节点和最小的子节点交换。在初始交换之后，我们可以将节点和其子节点重复交换，直到节点被交换到正确的位置，使它小于两个子节点。树交换节点的代码可以在代码中的 percDown 和 minChild 方法中找到。

为了完成我们对二叉堆的讨论，我们将看从一个列表构建整个堆的方法。你可能想到的第一种方法如下所示。给定一个列表，通过一次插入一个键轻松地构建一个堆。由于你从一个项的列表开始，该列表是有序的，可以使用二分查找找到正确的位置，以大约 O(log^n ) 操作的成本插入下一个键。 但是，请记住，在列表中间插入项可能需要 O(n) 操作来移动列表的其余部分，为新项腾出空间。 因此，要在堆中插入 n 个键，将需要总共 O(nlogn) 操作。 然而，如果我们从整个列表开始，那么我们可以在 O(n) 操作中构建整个堆。

```python
def buildHeap(self,alist):
    i = len(alist) // 2
    self.currentSize = len(alist)
    self.heapList = [0] + alist[:]
    while (i > 0):
        self.percDown(i)
        i = i - 1
```

完整代码：

```python
class BinHeap:
    def __init__(self):
        self.heapList = [0]
        self.currentSize = 0


    def percUp(self,i):
        while i // 2 > 0:
          if self.heapList[i] < self.heapList[i // 2]:
             tmp = self.heapList[i // 2]
             self.heapList[i // 2] = self.heapList[i]
             self.heapList[i] = tmp
          i = i // 2

    def insert(self,k):
      self.heapList.append(k)
      self.currentSize = self.currentSize + 1
      self.percUp(self.currentSize)

    def percDown(self,i):
      while (i * 2) <= self.currentSize:
          mc = self.minChild(i)
          if self.heapList[i] > self.heapList[mc]:
              tmp = self.heapList[i]
              self.heapList[i] = self.heapList[mc]
              self.heapList[mc] = tmp
          i = mc

    def minChild(self,i):
      if i * 2 + 1 > self.currentSize:
          return i * 2
      else:
          if self.heapList[i*2] < self.heapList[i*2+1]:
              return i * 2
          else:
              return i * 2 + 1

    def delMin(self):
      retval = self.heapList[1]
      self.heapList[1] = self.heapList[self.currentSize]
      self.currentSize = self.currentSize - 1
      self.heapList.pop()
      self.percDown(1)
      return retval

    def buildHeap(self,alist):
      i = len(alist) // 2
      self.currentSize = len(alist)
      self.heapList = [0] + alist[:]
      while (i > 0):
          self.percDown(i)
          i = i - 1

bh = BinHeap()
bh.buildHeap([9,5,6,2,3])

print(bh.delMin())
print(bh.delMin())
print(bh.delMin())
print(bh.delMin())
print(bh.delMin())
```

我们可以在 O(n) 中构建堆的断言可能看起来有点神秘，证明超出了本书的范围。 然而，理解的关键是记住 log^n 因子是从树的高度派生的。 对于 buildHeap 中的大部分工作，树比log^n 短。
基于可以从 O(n) 时间构建堆的事实，你可以使用堆对列表在 O(nlog⁡n) 时间内排序

#### 二叉查找树

在我们看实现之前，先来看看 map ADT 提供的接口。你会注意到，这个接口与Python 字典非常相似。

- Map() 创建一个新的空 map 。
- put(key，val) 向 map 中添加一个新的键值对。如果键已经在 map 中，那么用新值替换旧值。
- get(key) 给定一个键，返回存储在 map 中的值，否则为 None。
- del 使用 del map[key] 形式的语句从 map 中删除键值对。
- len() 返回存储在映射中的键值对的数量。
- in 返回 True 如果给定的键在 map 中。

二叉搜索树依赖于在左子树中找到的键小于父节点的属性，并且在右子树中找到的键大于父代。 我们将这个称为 bst属性。 当我们如上所述实现 Map 接口时，bst 属性将指导我们的实现。

为了实现二叉搜索树，我们将使用类似于我们用于实现链表的节点和引用方法，以及表达式树。但是，因为我们必须能够创建和使用一个空的二叉搜索树，我们的实现将使用两个类。
第一个类称为 BinarySearchTree ，第二个类称为 TreeNode 。 BinarySearchTree 类具有对作为二叉搜索树的根的 TreeNode 的引用。在大多数情况下，外部类中定义的外部方法只是检查树是否为空。如果树中有节点，请求只是传递到 BinarySearchTree 类中定义的私有方法，该方法以 root 作为参数。在树是空的或者我们想要删除树根的键的情况下，我们必须采取特殊的行动。

```python
class BinarySearchTree:

    def __init__(self):
        self.root = None
        self.size = 0

    def length(self):
        return self.size

    def __len__(self):
        return self.size

    def __iter__(self):
        return self.root.__iter__()
```

```python
class TreeNode:
   def __init__(self,key,val,left=None,right=None,
                                       parent=None):
        self.key = key
        self.payload = val
        self.leftChild = left
        self.rightChild = right
        self.parent = parent

    def hasLeftChild(self):
        return self.leftChild

    def hasRightChild(self):
        return self.rightChild

    def isLeftChild(self):
        return self.parent and self.parent.leftChild == self

    def isRightChild(self):
        return self.parent and self.parent.rightChild == self

    def isRoot(self):
        return not self.parent

    def isLeaf(self):
        return not (self.rightChild or self.leftChild)

    def hasAnyChildren(self):
        return self.rightChild or self.leftChild

    def hasBothChildren(self):
        return self.rightChild and self.leftChild

    def replaceNodeData(self,key,value,lc,rc):
        self.key = key
        self.payload = value
        self.leftChild = lc
        self.rightChild = rc
        if self.hasLeftChild():
            self.leftChild.parent = self
        if self.hasRightChild():
            self.rightChild.parent = self
```

```python
def put(self,key,val):
    if self.root:
        self._put(key,val,self.root)
    else:
        self.root = TreeNode(key,val)
    self.size = self.size + 1

def _put(self,key,val,currentNode):
    if key < currentNode.key:
        if currentNode.hasLeftChild():
               self._put(key,val,currentNode.leftChild)
        else:
               currentNode.leftChild = TreeNode(key,val,parent=currentNode)
    else:
        if currentNode.hasRightChild():
               self._put(key,val,currentNode.rightChild)
        else:
               currentNode.rightChild = TreeNode(key,val,parent=currentNode)
```

我们实现插入的一个重要问题是重复的键不能正确处理。当我们的树被实现时，重复键将在具有原始键的节点的右子树中创建具有相同键值的新节点。这样做的结果是，具有新键的节点将永远不会在搜索期间被找到。处理插入重复键的更好方法是将新键相关联的值替换旧值。

```python
def __setitem__(self,k,v):
    self.put(k,v)
```

```python
def get(self,key):
    if self.root:
        res = self._get(key,self.root)
        if res:
               return res.payload
        else:
               return None
    else:
        return None

def _get(self,key,currentNode):
    if not currentNode:
        return None
    elif currentNode.key == key:
        return currentNode
    elif key < currentNode.key:
        return self._get(key,currentNode.leftChild)
    else:
        return self._get(key,currentNode.rightChild)

def __getitem__(self,key):
    return self.get(key)
    
def __contains__(self,key):
    if self._get(key,self.root):
        return True
    else:
        return False
```

```python
if 'Northfield' in myZipTree:
    print("oom ya ya")
```

```python
def delete(self,key):
   if self.size > 1:
      nodeToRemove = self._get(key,self.root)
      if nodeToRemove:
          self.remove(nodeToRemove)
          self.size = self.size-1
      else:
          raise KeyError('Error, key not in tree')
   elif self.size == 1 and self.root.key == key:
      self.root = None
      self.size = self.size - 1
   else:
      raise KeyError('Error, key not in tree')

def __delitem__(self,key):
    self.delete(key)
```

一旦我们找到了我们要删除的键的节点，我们必须考虑三种情况：

1. 要删除的节点没有子节点。
2. 要删除的节点只有一个子节点。
3. 要删除的节点有两个子节点。

```python
if currentNode.isLeaf():
    if currentNode == currentNode.parent.leftChild:
        currentNode.parent.leftChild = None
    else:
        currentNode.parent.rightChild = None
```



对于第二种情况, 仅讨论当前节点具有左孩子的情况

1. 如果当前节点是左子节点，则我们只需要更新左子节点的父引用以指向当前节点的父节点，然后更新父节点的左子节点引用以指向当前节点的左子节点。

2. 如果当前节点是右子节点，则我们只需要更新左子节点的父引用以指向当前节点的父节点，然后更新父节点的右子节点引用以指向当前节点的左子节点。
3. 如果当前节点没有父级，则它是根。在这种情况下，我们将通过在根上调用 replaceNodeData 方法来替换 key ， payload ， leftChild 和 rightChild 数据。

```python
else: # this node has one child
   if currentNode.hasLeftChild():
      if currentNode.isLeftChild():
          currentNode.leftChild.parent = currentNode.parent
          currentNode.parent.leftChild = currentNode.leftChild
      elif currentNode.isRightChild():
          currentNode.leftChild.parent = currentNode.parent
          currentNode.parent.rightChild = currentNode.leftChild
      else:
          currentNode.replaceNodeData(currentNode.leftChild.key,
                             currentNode.leftChild.payload,
                             currentNode.leftChild.leftChild,
                             currentNode.leftChild.rightChild)
   else:
      if currentNode.isLeftChild():
          currentNode.rightChild.parent = currentNode.parent
          currentNode.parent.leftChild = currentNode.rightChild
      elif currentNode.isRightChild():
          currentNode.rightChild.parent = currentNode.parent
          currentNode.parent.rightChild = currentNode.rightChild
      else:
          currentNode.replaceNodeData(currentNode.rightChild.key,
                             currentNode.rightChild.payload,
                             currentNode.rightChild.leftChild,
                             currentNode.rightChild.rightChild)
```



第三种情况是最难处理的情况。 如果一个节点有两个孩子，那么我们不太可能简单地提升其中一个节点来占据节点的位置。 然而，我们可以在树中搜索可用于替换被调度删除的节点的节点。 我们需要的是一个节点，它将保留现有的左和右子树的二叉搜索树关系。 执行此操作的节点是树中具有次最大键的节点。 我们将这个节点称为后继节点，我们将看一种方法来很快找到后继节点。 继承节点保证没有多于一个孩子，所以我们知道使用已经实现的两种情况删除它。 一旦删除了后继，我们只需将它放在树中，代替要删除的节点。

```python
elif currentNode.hasBothChildren(): #interior
        succ = currentNode.findSuccessor()
        succ.spliceOut()
        currentNode.key = succ.key
        currentNode.payload = succ.payload
```



找到后继的代码，是 TreeNode 类的一个方法。此代码利用二叉搜索树的相同属性，采用中序遍历从最小到最大打印树中的节点。在寻找接班人时，有三种情况需要考虑：

1. 如果节点有右子节点，则后继节点是右子树中的最小的键。
2. 如果节点没有右子节点并且是父节点的左子节点，则父节点是后继节点。
3. 如果节点是其父节点的右子节点，并且它本身没有右子节点，则此节点的后继节点是其父节点的后继节点，不包括此节点。

```python
def findSuccessor(self):
    succ = None
    if self.hasRightChild():
        succ = self.rightChild.findMin()
    else:
        if self.parent:
               if self.isLeftChild():
                   succ = self.parent
               else:
                   self.parent.rightChild = None
                   succ = self.parent.findSuccessor()
                   self.parent.rightChild = self
    return succ

def findMin(self):
    current = self
    while current.hasLeftChild():
        current = current.leftChild
    return current

def spliceOut(self):
    if self.isLeaf():
        if self.isLeftChild():
               self.parent.leftChild = None
        else:
               self.parent.rightChild = None
    elif self.hasAnyChildren():
        if self.hasLeftChild():
               if self.isLeftChild():
                  self.parent.leftChild = self.leftChild
               else:
                  self.parent.rightChild = self.leftChild
               self.leftChild.parent = self.parent
        else:
               if self.isLeftChild():
                  self.parent.leftChild = self.rightChild
               else:
                  self.parent.rightChild = self.rightChild
               self.rightChild.parent = self.parent
```

按中序遍历树中的所有键

```python
def __iter__(self):
   if self:
      if self.hasLeftChild():
             for elem in self.leftChiLd:
                yield elem
      yield self.key
      if self.hasRightChild():
             for elem in self.rightChild:
                yield elem
```

完整代码

```python
class TreeNode:
    def __init__(self,key,val,left=None,right=None,parent=None):
        self.key = key
        self.payload = val
        self.leftChild = left
        self.rightChild = right
        self.parent = parent

    def hasLeftChild(self):
        return self.leftChild

    def hasRightChild(self):
        return self.rightChild

    def isLeftChild(self):
        return self.parent and self.parent.leftChild == self

    def isRightChild(self):
        return self.parent and self.parent.rightChild == self

    def isRoot(self):
        return not self.parent

    def isLeaf(self):
        return not (self.rightChild or self.leftChild)

    def hasAnyChildren(self):
        return self.rightChild or self.leftChild

    def hasBothChildren(self):
        return self.rightChild and self.leftChild

    def replaceNodeData(self,key,value,lc,rc):
        self.key = key
        self.payload = value
        self.leftChild = lc
        self.rightChild = rc
        if self.hasLeftChild():
            self.leftChild.parent = self
        if self.hasRightChild():
            self.rightChild.parent = self


class BinarySearchTree:

    def __init__(self):
        self.root = None
        self.size = 0

    def length(self):
        return self.size

    def __len__(self):
        return self.size

    def put(self,key,val):
        if self.root:
            self._put(key,val,self.root)
        else:
            self.root = TreeNode(key,val)
        self.size = self.size + 1

    def _put(self,key,val,currentNode):
        if key < currentNode.key:
            if currentNode.hasLeftChild():
                   self._put(key,val,currentNode.leftChild)
            else:
                   currentNode.leftChild = TreeNode(key,val,parent=currentNode)
        else:
            if currentNode.hasRightChild():
                   self._put(key,val,currentNode.rightChild)
            else:
                   currentNode.rightChild = TreeNode(key,val,parent=currentNode)

    def __setitem__(self,k,v):
       self.put(k,v)

    def get(self,key):
       if self.root:
           res = self._get(key,self.root)
           if res:
                  return res.payload
           else:
                  return None
       else:
           return None

    def _get(self,key,currentNode):
       if not currentNode:
           return None
       elif currentNode.key == key:
           return currentNode
       elif key < currentNode.key:
           return self._get(key,currentNode.leftChild)
       else:
           return self._get(key,currentNode.rightChild)

    def __getitem__(self,key):
       return self.get(key)

    def __contains__(self,key):
       if self._get(key,self.root):
           return True
       else:
           return False

    def delete(self,key):
      if self.size > 1:
         nodeToRemove = self._get(key,self.root)
         if nodeToRemove:
             self.remove(nodeToRemove)
             self.size = self.size-1
         else:
             raise KeyError('Error, key not in tree')
      elif self.size == 1 and self.root.key == key:
         self.root = None
         self.size = self.size - 1
      else:
         raise KeyError('Error, key not in tree')

    def __delitem__(self,key):
       self.delete(key)

    def spliceOut(self):
       if self.isLeaf():
           if self.isLeftChild():
                  self.parent.leftChild = None
           else:
                  self.parent.rightChild = None
       elif self.hasAnyChildren():
           if self.hasLeftChild():
                  if self.isLeftChild():
                     self.parent.leftChild = self.leftChild
                  else:
                     self.parent.rightChild = self.leftChild
                  self.leftChild.parent = self.parent
           else:
                  if self.isLeftChild():
                     self.parent.leftChild = self.rightChild
                  else:
                     self.parent.rightChild = self.rightChild
                  self.rightChild.parent = self.parent

    def findSuccessor(self):
      succ = None
      if self.hasRightChild():
          succ = self.rightChild.findMin()
      else:
          if self.parent:
                 if self.isLeftChild():
                     succ = self.parent
                 else:
                     self.parent.rightChild = None
                     succ = self.parent.findSuccessor()
                     self.parent.rightChild = self
      return succ

    def findMin(self):
      current = self
      while current.hasLeftChild():
          current = current.leftChild
      return current

    def remove(self,currentNode):
         if currentNode.isLeaf(): #leaf
           if currentNode == currentNode.parent.leftChild:
               currentNode.parent.leftChild = None
           else:
               currentNode.parent.rightChild = None
         elif currentNode.hasBothChildren(): #interior
           succ = currentNode.findSuccessor()
           succ.spliceOut()
           currentNode.key = succ.key
           currentNode.payload = succ.payload

         else: # this node has one child
           if currentNode.hasLeftChild():
             if currentNode.isLeftChild():
                 currentNode.leftChild.parent = currentNode.parent
                 currentNode.parent.leftChild = currentNode.leftChild
             elif currentNode.isRightChild():
                 currentNode.leftChild.parent = currentNode.parent
                 currentNode.parent.rightChild = currentNode.leftChild
             else:
                 currentNode.replaceNodeData(currentNode.leftChild.key,
                                    currentNode.leftChild.payload,
                                    currentNode.leftChild.leftChild,
                                    currentNode.leftChild.rightChild)
           else:
             if currentNode.isLeftChild():
                 currentNode.rightChild.parent = currentNode.parent
                 currentNode.parent.leftChild = currentNode.rightChild
             elif currentNode.isRightChild():
                 currentNode.rightChild.parent = currentNode.parent
                 currentNode.parent.rightChild = currentNode.rightChild
             else:
                 currentNode.replaceNodeData(currentNode.rightChild.key,
                                    currentNode.rightChild.payload,
                                    currentNode.rightChild.leftChild,
                                    currentNode.rightChild.rightChild)




mytree = BinarySearchTree()
mytree[3]="red"
mytree[4]="blue"
mytree[6]="yellow"
mytree[2]="at"

print(mytree[6])
print(mytree[2])
```

查找树分析

让我们先来看看 put方法。其性能的限制因素是二叉树的高度。从词汇部分回忆一下树的高度是根和最深叶节点之间的边的数量。高度是限制因素，因为当我们寻找合适的位置将一个节点插入到树中时，我们需要在树的每个级别最多进行一次比较。

二叉树的高度可能是多少？这个问题的答案取决于如何将键添加到树。如果按照随机顺序添加键，树的高度将在 log2^⁡n 附近，其中 n 是树中的节点数。这是因为如果键是随机分布的，其中大约一半将小于根，一半大于根。请记住，在二叉树中，根节点有一个节点，下一级节点有两个节点，下一个节点有四个节点。任何特定级别的节点数为 2^d ，其中 d 是级别的深度。完全平衡的二叉树中的节点总数为 2^h+1 - 1，其中 h 表示树的高度。

完全平衡的树在左子树中具有与右子树相同数量的节点。在平衡二叉树中， put 的最坏情况性能是 O(log2^⁡n ），其中 n 是树中的节点数。注意，这是与前一段中的计算的反比关系。所以 log2^⁡n 给出了树的高度，并且表示了在适当的位置插入新节点时，需要做的最大比较次数。
不幸的是，可以通过以排序顺序插入键来构造具有高度 n 的搜索树！在这种情况下，put方法的性能是 O(n)。

现在你明白了 put 方法的性能受到树的高度的限制，你可能猜测其他方法 get ， in 和del 也是有限制的。 由于 get 搜索树以找到键，在最坏的情况下，树被一直搜索到底部，并且没有找到键。 乍一看， del 似乎更复杂，因为它可能需要在删除操作完成之前搜索后继。 但请记住，找到后继者的最坏情况也只是树的高度，这意味着你只需要加倍工作。 因为加倍是一个常数因子，它不会改变最坏的情况

#### AVL平衡二叉搜索树

在上一节中，我们考虑构建一个二叉搜索树。正如我们所学到的，二叉搜索树的性能可以降级到 O(n) 的操作，如 get 和 put ，如果树变得不平衡。在本节中，我们将讨论一种特殊
类型的二叉搜索树，它自动确保树始终保持平衡。这棵树被称为 AVL树，以其发明人命名：G.M. Adelson-Velskii 和E.M.Landis。

AVL树实现 Map 抽象数据类型就像一个常规的二叉搜索树，唯一的区别是树的执行方式。为了实现我们的 AVL树，我们需要跟踪树中每个节点的平衡因子。我们通过查看每个节点的左右子树的高度来做到这一点。更正式地，我们将节点的平衡因子定义为左子树的高度和右子树的高度之间的差。

balanceFactor = height(leftSubTree) - height(rightSubTree)

使用上面给出的平衡因子的定义，我们说如果平衡因子大于零，则子树是左重的。如果平衡因子小于零，则子树是右重的。如果平衡因子是零，那么树是完美的平衡。为了实现AVL树，并且获得具有平衡树的好处，如果平衡因子是 -1,0 或 1，我们将定义树平衡。一旦树中的节点的平衡因子是在这个范围之外，我们将需要一个程序来使树恢复平衡

我们看到的高度h(Nh) 的树中的节点数量的模式是

$N_h=1+N_{h−1}+N_{h−2}$  

经过一系列推导(借助斐波那契数列的结论)，在任何时候，我们的AVL树的高度等于树中节点数目的对数的常数（1.44）倍。 这是搜索我们的AVL树的好消息，因为它将搜索限制为O(logN）。

现在我们已经证明保持 AVL树的平衡将是一个很大的性能改进，让我们看看如何增加过程来插入一个新的键到树。由于所有新的键作为叶节点插入到树中，并且我们知道新叶的平衡因子为零，所以刚刚插入的节点没有新的要求。但一旦添加新叶，我们必须更新其父的平衡因子。这个新叶如何影响父的平衡因子取决于叶节点是左孩子还是右孩子。如果新节点是右子节点，则父节点的平衡因子将减少1。如果新节点是左子节点，则父节点的平衡因子将增加1。这个关系可以递归地应用到新节点的祖父节点，并且应用到每个祖先一直到树的根。由于这是一个递归过程，我们来看一下用于更新平衡因子的两种基本情况：

- 递归调用已到达树的根。
- 父节点的平衡因子已调整为零。你应该说服自己，一旦一个子树的平衡因子为零，那么它的祖先节点的平衡不会改变。

我们将实现 AVL 树作为 BinarySearchTree 的子类。首先，我们将覆盖 _put 方法并编写一个新的 updateBalance 辅助方法。你将注意到， _put 的定义与简单二叉搜索树中的完全相同，除了第 7 行和第 13 行上对 updateBalance 的调用的添加。

```python
def _put(self,key,val,currentNode):
    if key < currentNode.key:
        if currentNode.hasLeftChild():
                self._put(key,val,currentNode.leftChild)
        else:
                currentNode.leftChild = TreeNode(key,val,parent=currentNode)
                self.updateBalance(currentNode.leftChild)
    else:
        if currentNode.hasRightChild():
                self._put(key,val,currentNode.rightChild)
        else:
                currentNode.rightChild = TreeNode(key,val,parent=currentNode)
                self.updateBalance(currentNode.rightChild)

def updateBalance(self,node):
    if node.balanceFactor > 1 or node.balanceFactor < -1:
        self.rebalance(node)
        return
    if node.parent != None:
        if node.isLeftChild():
                node.parent.balanceFactor += 1
        elif node.isRightChild():
                node.parent.balanceFactor -= 1

        if node.parent.balanceFactor != 0:
                self.updateBalance(node.parent)
```

新的 updateBalance 方法完成了大多数工作。这实现了我们刚才描述的递归过程。
updateBalance 方法首先检查当前节点是否不够平衡，需要重新平衡（第16行）。如果平衡，则重新平衡完成，并且不需要对父节点进行进一步更新。如果当前节点不需要重新平
衡，则调整父节点的平衡因子。如果父的平衡因子不为零，那么算法通过递归调用父对象上的 updateBalance ，继续沿树向根向上运行。
当需要树重新平衡时，我们如何做呢？有效的重新平衡是使AVL树在不牺牲性能的情况下正常工作的关键。为了使AVL树恢复平衡，我们将在树上执行一个或多个旋转。

要执行左旋转，我们基本上执行以下操作：

- 提升右孩子（B）成为子树的根。
- 将旧根（A）移动为新根的左子节点。
- 如果新根（B）已经有一个左孩子，那么使它成为新左孩子（A）的右孩子。注意：由于新根（B）是A的右孩子，A 的右孩子在这一点上保证为空。这允许我们添加一个新的节点作为右孩子，不需进一步的考虑。

虽然这个过程在概念上相当容易，但是代码的细节有点棘手，因为我们需要按照正确的顺序移动事物，以便保留二叉搜索树的所有属性。此外，我们需要确保适当地更新所有的父指针。

要执行右旋转，我们基本上执行以下操作：

- 提升左子节点（C）为子树的根。
- 将旧根（E）移动为新根的右子树。
- 如果新根（C）已经有一个正确的孩子（D），那么使它成为新的右孩子（E）的左孩子。注意：由于新根（C）是 E 的左子节点，因此 E 的左子节点在此时保证为空。这允许我们添加一个新节点作为左孩子，不需进一步的考虑。

在第2行中，我们创建一个临时变量来跟踪子树的新根。正如我们之前所说的，新的根是上一个根的右孩子。现在对这个临时变量存储了一个对右孩子的引用，我们用新的左孩子替换旧根的右孩子。下一步是调整两个节点的父指针。如果 newRoot 有一个左子节点，那么左子节点的新父节点变成旧的根节点。新根的父节点设置为旧根的父节点。如果旧根是整个树的根，那么我们必须设置树的根以指向这个新根。否则，如果旧根是左孩子，则我们将左孩子的父节点更改为
指向新根;否则我们改变右孩子的父亲指向新的根。（行10-13）。最后，我们将旧根的父节点设置为新根

```python
def rotateLeft(self,rotRoot):
    newRoot = rotRoot.rightChild
    rotRoot.rightChild = newRoot.leftChild
    if newRoot.leftChild != None:
        newRoot.leftChild.parent = rotRoot
    newRoot.parent = rotRoot.parent
    if rotRoot.isRoot():
        self.root = newRoot
    else:
        if rotRoot.isLeftChild():
                rotRoot.parent.leftChild = newRoot
        else:
            rotRoot.parent.rightChild = newRoot
    newRoot.leftChild = rotRoot
    rotRoot.parent = newRoot
    rotRoot.balanceFactor = rotRoot.balanceFactor + 1 - min(newRoot.balanceFactor, 0)
    newRoot.balanceFactor = newRoot.balanceFactor + 1 + max(rotRoot.balanceFactor, 0)
```

最后，第16-17行需要一些解释。 在这两行中，我们更新旧根和新根的平衡因子。 由于所有其他移动都是移动整个子树，所以所有其他节点的平衡因子不受旋转的影响。

```python
def rebalance(self,node):
  if node.balanceFactor < 0:
         if node.rightChild.balanceFactor > 0:
            self.rotateRight(node.rightChild)
            self.rotateLeft(node)
         else:
            self.rotateLeft(node)
  elif node.balanceFactor > 0:
         if node.leftChild.balanceFactor < 0:
            self.rotateLeft(node.leftChild)
            self.rotateRight(node)
         else:
            self.rotateRight(node)
```

通过保持树在所有时间的平衡，我们可以确保 get 方法将按 O(log2(n)) 时间运行。但问题是我们的 put 方法有什么成本？让我们将它分解为 put 执行的操作。由于将新节点作为叶子插入，更新所有父节点的平衡因子将需要最多log2^n 运算，树的每层一个运算。如果发现子树不平衡，则需要最多两次旋转才能使树重新平衡。但是，每个旋转在 O(1)时间中工作，因此我们的put操作仍然是O（log2^n )。
在这一点上，我们已经实现了一个功能AVL树，除非你需要删除一个节点的能力。

#### Map抽象数据结构总结

| operation | Sorted List | Hash Table | Binary Search Tree | AVL Tree    |
| --------- | ----------- | ---------- | ------------------ | ----------- |
| put       | O(n)        | O(1)       | O(n)               | O($log_2n$) |
| get       | O($log_2n$) | O(1)       | O(n)               | O($log_2n$) |
| in        | O($log_2n$) | O(1)       | O(n)               | O($log_2n$) |
| del       | O(n)        | O(1)       | O(n)               | O($log_2n$) |

 ### 图

顶点 顶点（也称为“节点”）是图的基本部分。它可以有一个名称，我们将称为“键”。一个顶点也可能有额外的信息。我们将这个附加信息称为“有效载荷”。

边 边（也称为“弧”）是图的另一个基本部分。边连接两个顶点，以表明它们之间存在关系。边可以是单向的或双向的。如果图中的边都是单向的，我们称该图是 有向图 。

权重 边可以被加权以示出从一个顶点到另一个顶点的成本。例如，在将一个城市连接到另一个城市的道路的图表中，边上的权重可以表示两个城市之间的距离。

利用这些定义，我们可以正式定义图。图可以由 G 表示，其中 G =(V，E) 。对于图 G，V 是一组顶点，E 是一组边。每个边是一个元组 (v，w) ，其中 w,v ∈ V 。我们可以添加第三个组件到边元组来表示权重。子图 s 是边 e 和顶点 v 的集合，使得 e⊂E 和 v⊂V 。

路径 图中的路径是由边连接的顶点序列。形式上，我们将定义一个路径为 w1，w2，...，wn ，使得 (wi，wi + 1) ∈ E , 当 1≤i≤ n-1 。未加权路径长度是路径中的边的数目，具体是 n-1。加权路径长度是路径中所有边的权重的总和。

循环 有向图中的循环是在同一顶点开始和结束的路径。没有循环的图形称为非循环图形。没有循环的有向图称为有向无环图或DAG 。我们将看到，如果问题可以表示为 DAG ，我们可以解决几个重要的问题。

图抽象数据类型（ADT）定义如下：
Graph() 创建一个新的空图。
addVertex(vert) 向图中添加一个顶点实例。
addEdge(fromVert, toVert) 向连接两个顶点的图添加一个新的有向边。
addEdge(fromVert, toVert, weight) 向连接两个顶点的图添加一个新的加权的有向边。
getVertex(vertKey) 在图中找到名为 vertKey 的顶点。
getVertices() 返回图中所有顶点的列表。
in 返回 True 如果 vertex in graph 里给定的顶点在图中，否则返回False。

实现图的最简单的方法之一是使用二维矩阵。在该矩阵实现中，每个行和列表示图中的顶点。存储在行 v 和列 w 的交叉点处的单元中的值表示是否存在从顶点 v 到顶点 w 的边。 当两个顶点通过边连接时，我们说它们是相邻的。

邻接矩阵的优点是简单，对于小图，很容易看到哪些节点连接到其他节点。 然而，注意矩阵中的大多数单元格是空的。 因为大多数单元格是空的，我们说这个矩阵是“稀疏的”。矩阵不是一种非常有效的方式来存储稀疏数据。

当边的数量大时，邻接矩阵是图的良好实现

实现稀疏连接图的更空间高效的方法是使用邻接表。在邻接表实现中，我们保存Graph 对象中的所有顶点的主列表，然后图中的每个顶点对象维护连接到的其他顶点的列表。 在我们的顶点类的实现中，我们将使用字典而不是列表，其中字典键是顶点，值是权重。

邻接表实现的优点是它允许我们紧凑地表示稀疏图。 邻接表还允许我们容易找到直接连接到特定顶点的所有链接。

使用字典，很容易在 Python 中实现邻接表

```python
class Vertex:
    def __init__(self,key):
        self.id = key
        self.connectedTo = {}

    def addNeighbor(self,nbr,weight=0):
        self.connectedTo[nbr] = weight

    def __str__(self):
        return str(self.id) + ' connectedTo: ' + str([x.id for x in self.connectedTo])

    def getConnections(self):
        return self.connectedTo.keys()

    def getId(self):
        return self.id

    def getWeight(self,nbr):
        return self.connectedTo[nbr]
```

```python
class Graph:
    def __init__(self):
        self.vertList = {}
        self.numVertices = 0

    def addVertex(self,key):
        self.numVertices = self.numVertices + 1
        newVertex = Vertex(key)
        self.vertList[key] = newVertex
        return newVertex

    def getVertex(self,n):
        if n in self.vertList:
            return self.vertList[n]
        else:
            return None

    def __contains__(self,n):
        return n in self.vertList

    def addEdge(self,f,t,cost=0):
        if f not in self.vertList:
            nv = self.addVertex(f)
        if t not in self.vertList:
            nv = self.addVertex(t)
        self.vertList[f].addNeighbor(self.vertList[t], cost)

    def getVertices(self):
        return self.vertList.keys()

    def __iter__(self):
        return iter(self.vertList.values())
```

```python
>>> g = Graph()
>>> for i in range(6):
...    g.addVertex(i)
>>> g.vertList
{0: <adjGraph.Vertex instance at 0x41e18>,
 1: <adjGraph.Vertex instance at 0x7f2b0>,
 2: <adjGraph.Vertex instance at 0x7f288>,
 3: <adjGraph.Vertex instance at 0x7f350>,
 4: <adjGraph.Vertex instance at 0x7f328>,
 5: <adjGraph.Vertex instance at 0x7f300>}
>>> g.addEdge(0,1,5)
>>> g.addEdge(0,5,2)
>>> g.addEdge(1,2,4)
>>> g.addEdge(2,3,9)
>>> g.addEdge(3,4,7)
>>> g.addEdge(3,5,3)
>>> g.addEdge(4,0,1)
>>> g.addEdge(5,4,8)
>>> g.addEdge(5,2,1)
>>> for v in g:
...    for w in v.getConnections():
...        print("( %s , %s )" % (v.getId(), w.getId()))
...
( 0 , 5 )
( 0 , 1 )
( 1 , 2 )
( 2 , 3 )
( 3 , 4 )
( 3 , 5 )
( 4 , 0 )
( 5 , 4 )
( 5 , 2 )
```

#### 字梯的问题

将单词 “FOOL” 转换为单词 “SAGE”。 在字梯中你通过改变一个字母逐渐发生变化。 在每一步，你必须将一个字变换成另一个字。

在本节中，我们将计算起始字转换为结束字所需的最小转换次数。
毫不奇怪，因为这一章是图，我们可以使用图算法解决这个问题。 这里是我们需要的步骤：

- 将字之间的关系表示为图。
- 使用称为广度优先搜索的图算法来找到从起始字到结束字的有效路径。

我们的第一个问题是弄清楚如何将大量的单词集合转换为图。 如果两个词只有一个字母不同，我们就创建从一个词到另一个词的边。如果我们可以创建这样的图，则从一个词到另一个词的任意路径就是词梯子拼图的解决方案。

假设我们有大量的桶，每个桶在外面有一个四个字母的单
词，除了标签中的一个字母已经被下划线替代。例如，我们可能有一个标记为“pop” 的桶。当我们处理列表中的每个单词时，我们使用 “” 作为通配符比较每个桶的单词，所以 “pope” 和 “pops “ 将匹配 ”pop_“。每次我们找到一个匹配的桶，我们就把单词放在那个桶。一旦我们把所有单词放到适当的桶中，就知道桶中的所有单词必须连接。

在 Python 中，我们使用字典来实现我们刚才描述的方案。我们刚才描述的桶上的标签是我们字典中的键。该键存储的值是单词列表。 一旦我们建立了字典，我们可以创建图。 我们通过为图中的每个单词创建一个顶点来开始图。 然后，我们在字典中的相同键下找到的所有顶点创建边。

```python
from pythonds.graphs import Graph

def buildGraph(wordFile):
    d = {}
    g = Graph()
    wfile = open(wordFile,'r')
    # create buckets of words that differ by one letter
    for line in wfile:
        word = line[:-1]
        for i in range(len(word)):
            bucket = word[:i] + '_' + word[i+1:]
            if bucket in d:
                d[bucket].append(word)
            else:
                d[bucket] = [word]
    # add vertices and edges for words in the same bucket
    for bucket in d.keys():
        for word1 in d[bucket]:
            for word2 in d[bucket]:
                if word1 != word2:
                    g.addEdge(word1,word2)
    return g
```

广度优先搜索算法

BFS 从起始顶点开始，颜色从灰色开始，表明它正在被探索。另外两个值，即距离和前导，对于起始顶点分别初始化为 0 和 None 。最后，放到一个队列中。下一步是开始系统地检查队列前面的顶点。我们通过迭代它的邻接表来探索队列前面的每个新节点。当检查邻接表上的每个节点时，检查其颜色。如果它是白色的，顶点是未开发的，有四件事情发生：

1. 新的，未开发的顶点 nbr，被着色为灰色。
2. nbr 的前导被设置为当前节点 currentVert
3. 到 nbr 的距离设置为到 currentVert + 1 的距离
4. nbr 被添加到队列的末尾。 将 nbr 添加到队列的末尾有效地调度此节点以进行进一步探
  索，但不是直到 currentVert 的邻接表上的所有其他顶点都被探索。

```python
from pythonds.graphs import Graph, Vertex
from pythonds.basic import Queue

def bfs(g,start):
  start.setDistance(0)
  start.setPred(None)
  vertQueue = Queue()
  vertQueue.enqueue(start)
  while (vertQueue.size() > 0):
    currentVert = vertQueue.dequeue()
    for nbr in currentVert.getConnections():
      if (nbr.getColor() == 'white'):
        nbr.setColor('gray')
        nbr.setDistance(currentVert.getDistance() + 1)
        nbr.setPred(currentVert)
        vertQueue.enqueue(nbr)
    currentVert.setColor('black')
```

如何按前导链接打印出字梯

```python
def traverse(y):
    x = y
    while (x.getPred()):
        print(x.getId())
        x = x.getPred()
    print(x.getId())

traverse(g.getVertex('sage'))
```

在继续使用其他图算法之前，让我们分析广度优先搜索算法的运行时性能。首先要观察的是，对于图中的每个顶点 |V| 最多执行一次 while 循环。因为一个顶点必须是白色，才能被检查和添加到队列。这给出了用于 while 循环的 O(v)。嵌套在 while 内部的 for 循环对于图中的每个边执行最多一次， |E| 。原因是每个顶点最多被出列一次，并且仅当节点 u 出队时，我们才检查从节点 u 到节点 v 的边。这给出了用于 for 循环的 O(E) 。组合这两个环路给出了O(V+E)。
当然做广度优先搜索只是任务的一部分。从起始节点到目标节点的链接之后是任务的另一部分。最糟糕的情况是，如果图是单个长链。在这种情况下，遍历所有顶点将是 O(V)。正常情况将是 |V| 的一小部分但我们仍然写 O(V)。
最后，至少对于这个问题，存在构建初始图形所需的时间。

#### 骑士之旅

骑士之旅图是在一个棋盘上用一个棋子当骑士玩。图的目的是找到一系列的动作，让骑士访问板上的每格一次。一个这样的序列被称为“旅游”

我们将使用两个主要步骤解决问题：

- 表示骑士在棋盘上作为图的动作。
- 使用图算法来查找长度为 rows×columns-1 的路径，其中图上的每个顶点都被访问一次。

为了将骑士的旅游问题表示为图，我们将使用以下两个点：棋盘上的每个正方形可以表示为图形中的一个节点。 骑士的每个合法移动可以表示为图形中的边。

knightGraph 函数在整个板上进行一次遍历。 在板上的每个方块上， knightGraph 函数调用 genLegalMoves，为板上的位置创建一个移动列表。 所有移动在图形中转换为边。 另一个帮助函数posToNodeId 按照行和列将板上的位置转换为顶点数的线性顶点数。

```python
from pythonds.graphs import Graph

def knightGraph(bdSize):
    ktGraph = Graph()
    for row in range(bdSize):
       for col in range(bdSize):
           nodeId = posToNodeId(row,col,bdSize)
           newPositions = genLegalMoves(row,col,bdSize)
           for e in newPositions:
               nid = posToNodeId(e[0],e[1],bdSize)
               ktGraph.addEdge(nodeId,nid)
    return ktGraph

def posToNodeId(row, column, board_size):
    return (row * board_size) + column
```

genLegalMoves 函数使用板上骑士的位置，并生成八个可能移动中的一个。
legalCoord 辅助函数确保生成的特定移动仍在板上。

```python
def genLegalMoves(x,y,bdSize):
    newMoves = []
    moveOffsets = [(-1,-2),(-1,2),(-2,-1),(-2,1),
                   ( 1,-2),( 1,2),( 2,-1),( 2,1)]
    for i in moveOffsets:
        newX = x + i[0]
        newY = y + i[1]
        if legalCoord(newX,bdSize) and \
                        legalCoord(newY,bdSize):
            newMoves.append((newX,newY))
    return newMoves

def legalCoord(x,bdSize):
    if x >= 0 and x < bdSize:
        return True
    else:
        return False
```

深度优先算法

knightTour 函数有四个参数： n ，搜索树中的当前深度; path ，到此为止访问的顶点的列表; u ，图中我们希望探索的顶点; limit 路径中的节点数。 knightTour 函数是递归的。当调用 knightTour 函数时，它首先检查基本情况。如果我们有一个包含 64 个顶点的路径，我们状态为 True 的 knightTour 返回，表示我们找到了一个成功的线路。如果路径不够长，我们继续通过选择一个新的顶点来探索一层，并对这个顶点递归调用knightTour。

DFS 还使用颜色来跟踪图中的哪些顶点已被访问。未访问的顶点是白色的，访问的顶点是灰色的。如果已经探索了特定顶点的所有邻居，并且我们尚未达到64个顶点的目标长度，我们已经到达死胡同。当我们到达死胡同时，我们必须回溯。当我们从状态为 False 的 knightTour返回时，发生回溯。在广度优先搜索中，我们使用一个队列来跟踪下一个要访问的顶点。由于深度优先搜索是递归的，我们隐式使用一个栈来帮助我们回溯。当我们从第 11 行的状态为False 的 knightTour 调用返回时，我们保持在 while 循环中，并查看 nbrList 中的下一个顶点。

```python
from pythonds.graphs import Graph, Vertex
def knightTour(n,path,u,limit):
        u.setColor('gray')
        path.append(u)
        if n < limit:
            nbrList = list(u.getConnections())
            i = 0
            done = False
            while i < len(nbrList) and not done:
                if nbrList[i].getColor() == 'white':
                    done = knightTour(n+1, path, nbrList[i], limit)
                i = i + 1
            if not done:  # prepare to backtrack
                path.pop()
                u.setColor('white')
        else:
            done = True
        return done
```

knightTour 对于你选择下一个要访问的顶点的方法非常敏感。我们到目前为止所实现的骑士之旅问题是大小为 O(k^N ) 的指数算法，其中 N 是棋盘上的方格数，k 是小常数。

使用具有最多可用移动的顶点作为路径上的下一个顶点的问题是，它倾向于让骑士在游览中早访问中间的方格。当这种情况发生时，骑士很容易陷入板的一侧，在那里它不能到达在板的另一侧的未访问的方格。另一方面，访问具有最少可用移动的方块首先推动骑士访问围绕板的边缘的方块。这确保了骑士能够尽早地访问难以到达的角落，并且只有在必要时才使用中间的方块跳过棋盘。利用这种知识加速算法被称为启发式。人类每天都使用启发式来帮助做出决策，启发式搜索通常用于人工智能领域。这个特定的启发式称为 Warnsdorff 算法，由 H. C. Warnsdorff 命名，他在 1823 年发表了他的算法。

```python
def orderByAvail(n):
    resList = []
    for v in n.getConnections():
        if v.getColor() == 'white':
            c = 0
            for w in v.getConnections():
                if w.getColor() == 'white':
                    c = c + 1
            resList.append((c,v))
    resList.sort(key=lambda x: x[0])
    return [y[1] for y in resList]
```

#### 通用深度优先搜索

```python
from pythonds.graphs import Graph
class DFSGraph(Graph):
    def __init__(self):
        super().__init__()
        self.time = 0

    def dfs(self):
        for aVertex in self:
            aVertex.setColor('white')
            aVertex.setPred(-1)
        for aVertex in self:
            if aVertex.getColor() == 'white':
                self.dfsvisit(aVertex)

    def dfsvisit(self,startVertex):
        startVertex.setColor('gray')
        self.time += 1
        startVertex.setDiscovery(self.time)
        for nextVertex in startVertex.getConnections():
            if nextVertex.getColor() == 'white':
                nextVertex.setPred(startVertex)
                self.dfsvisit(nextVertex)
        startVertex.setColor('black')
        self.time += 1
        startVertex.setFinish(self.time)
```

bfs 使用队列， dfsvisit 使用栈

每个节点的开始和结束时间展示一个称为 括号属性 的属性。 该属性意味着深度优先树中的特定节点的所有子节点具有比它们的父节点更晚的发现时间和更早的完成时间。

深度优先搜索的一般运行时间如下。 dfs 中的循环都在 O(V) 中运行，不计入 dfsvisit 中发生的情况，因为它们对图中的每个顶点执行一次。 在 dfsvisit 中，对当前顶点的邻接表中
的每个边执行一次循环。 由于只有当顶点为白色时， dfsvisit 才被递归调用，所以循环对图中的每个边或 O(E) 执行最多一次。 因此，深度优先搜索的总时间是 O(V + E)。

#### 拓扑排序

为了帮助我们决定应该做的每一个步骤的精确顺序，我们转向一个图算法称为 拓扑排序 。
拓扑排序采用有向无环图，并且产生所有其顶点的线性排序，使得如果图 G 包含边（v，w），则顶点 v 在排序中位于顶点 w 之前。定向非循环图在许多应用中使用以指示事件的优先级。制作煎饼只是一个例子;其他示例包括软件项目计划，用于数据库查询的优先图以及乘法矩阵。
拓扑排序是深度优先搜索的简单但有用的改造。拓扑排序的算法如下：

1. 对于某些图 g 调用 dfs(g)。我们想要调用深度优先搜索的主要原因是计算每个顶点的完
  成时间。
2. 以完成时间的递减顺序将顶点存储在列表中。
3. 返回有序列表作为拓扑排序的结果。

#### 强连通分量

我们将用来研究一些附加算法的图，由互联网上的主机之间的连接和网页之间的链接产生的图。 我们将从网页开始。
像 Google 和 Bing 这样的搜索引擎利用了网页上的页面形成非常大的有向图。 为了将万维网变换为图，我们将把一个页面视为一个顶点，并将页面上的超链接作为将一个顶点连接到另一个顶点的边缘。

可以帮助找到图中高度互连的顶点的集群的一种图算法被称为强连通分量算法（SCC）。我们正式定义图 G 的强连通分量 C 作为顶点 C⊂V 的最大子集，使得对于每对顶点 v,w∈C，我们具有从 v 到 w 的路径和从 w 到 v 的路径。

一旦确定了强连通分量，我们就可以通过将一个强连通分量中的所有顶点组合成一个较大的顶点来显示该图的简化视图。

我们再次看到，我们可以通过使用深度优先搜索来创建一个非常强大和高效的算法。 在我们处理主 SCC 算法之前，我们必须考虑另一个定义。 图 G 的转置被定义为图 G^T ，其中图中的所有边已经反转。 也就是说，如果在原始图中存在从节点 A 到节点 B 的有向边，则 G^T将包含从节点 B 到节点 A 的边。

我们现在可以描述用于计算图的强连通分量的算法。

1. 调用 dfs 为图 G 计算每个顶点的完成时间。
2. 计算 G^T 。
3. 为图 G^T 调用 dfs，但在 DFS 的主循环中，以完成时间的递减顺序探查每个顶点。
4. 在步骤 3 中计算的森林中的每个树是强连通分量。输出森林中每个树中每个顶点的顶点标识组件。

#### 最短路径问题

Internet 上的通信如何工作的高层概述。当使用浏览器从服务器请求网页时，请求必须通过局域网传输，并通过路由器传输到 Internet上。 该请求通过因特网传播，并最终到达服务器所在的局域网路由器。 请求的网页然后通过相同的路由器回到您的浏览器。 在
所有这些路由器一起工作，让信息从一个地方到另一个地方。 可以看到有许多路由器，如果你的计算机支持 traceroute 命令。下面的文本显示 traceroute 命令的输出，说明在 Luther College 的Web服务器和明尼苏达大学的邮件服务器之间有13个路由器

互联网上的每个路由器都连接到一个或多个路由器。因此，如果在一天的不同时间运行traceroute 命令，你很可能会看到你的信息在不同的时间流经不同的路由器。这是因为存在与一对路由器之间的每个连接相关联的成本，这取决于业务量，一天中的时间以及许多其他因素。到这个时候，你不会惊讶，我们可以将路由器的网络表示为带有加权边的图形。

我们要解决的问题是找到具有最小总权重的路径，沿着该路径路由传送任何给定的消息。这个问题听起来很熟悉，因为它类似于我们使用广度优先搜索解决的问题，我们这里关心的是路径的总权重，而不是路径中的跳数。应当注意，如果所有权重相等，则问题是相同的。

#### Dijkstra算法

我们将用于确定最短路径的算法称为“Dijkstra算法”。Dijkstra算法是一种迭代算法，它为我们提供从一个特定起始节点到图中所有其他节点的最短路径。这也类似于广度优先搜索的结果。
为了跟踪从开始节点到每个目的地的总成本，我们将使用顶点类中的 dist 实例变量。 dist实例变量将包含从开始到所讨论的顶点的最小权重路径的当前总权重。该算法对图中的每个顶点重复一次;然而，我们在顶点上迭代的顺序由优先级队列控制。用于确定优先级队列中对象顺序的值为dist。当首次创建顶点时，dist被设置为非常大的数。理论上，你将 dist 设置为无穷大，但在实践中，我们只是将它设置为一个数字，大于任何真正的距离，我们将在问题中试图解决。

```python
from pythonds.graphs import PriorityQueue, Graph, Vertex
def dijkstra(aGraph,start):
    pq = PriorityQueue()
    start.setDistance(0)
    pq.buildHeap([(v.getDistance(),v) for v in aGraph])
    while not pq.isEmpty():
        currentVert = pq.delMin()
        for nextVert in currentVert.getConnections():
            newDist = currentVert.getDistance() \
                    + currentVert.getWeight(nextVert)
            if newDist < nextVert.getDistance():
                nextVert.setDistance( newDist )
                nextVert.setPred(currentVert)
                pq.decreaseKey(nextVert,newDist)
```

Dijkstra的算法使用优先级队列。你可能还记得，优先级队列是基于我们在树章节中实现的堆。这个简单的实现和我们用于Dijkstra算法的实现之间有几个区别。首先，PriorityQueue 类存储键值对的元组。这对于Dijkstra的算法很重要，因为优先级队列中的键必须匹配图中顶点的键。其次，值用于确定优先级，并且用于确定键在优先级队列中的位置。在这个实现中，我们使用到顶点的距离作为优先级，因为我们看到当探索下一个顶点时，我们总是要探索具有最小距离的顶点。第二个区别是增加 decreaseKey 方法。正如你看到的，当一个已经在队列中的顶点的距离减小时，使用这个方法，将该顶点移动到队列的前面。

重要的是要注意，Dijkstra的算法只有当权重都是正数时才起作用。 如果你在图的边引入一个负权重，算法永远不会退出。
我们将注意到，通过因特网路由发送消息，可以使用其他算法来找到最短路径。 在互联网上使用 Dijkstra 算法的一个问题是，为了使算法运行，你必须有一个完整的图表示。 这意味着每个路由器都有一个完整的互联网中所有路由器地图。 实际上不是这种情况，算法的其他变种允许每个路由器在它们发送时发现图。 你可能想要了解的一种这样的算法称为 “距离矢量”路由算法。

我们首先注意到，构建优先级队列需要 O(v) 时间，因为我们最初将图中的每个顶点添加到优先级队列。 一旦构造了队列，则对于每个顶点执行一次 while 循环，因为顶点都在开始处添加，并且在那之后才被移除。 在该循环中每次调用 delMin，需要 O(log^V )时间。 将该部分循环和对 delMin 的调用取为 O(Vlog^V )。 for循环对于图中的每个边执行一次，并且在 for 循环中，对 decreaseKey 的调用需要时间
O(Elog^V) 。 因此，组合运行时间为 O((V + E)log^V )。

#### Prim生成树算法

对于我们最后的图算法，让我们考虑一个在线游戏设计师和网络收音机提供商面临的问题。问题是他们想有效地将一条信息传递给任何人和每个可能在听的人。 这在游戏中是重要的，使得所有玩家知道每个其他玩家的最新位置。 对于网络收音机是重要的，以便所有该调频的收听者获得他们需要的所有数据来刷新他们正在收听的歌曲。

这个问题有一些强力的解决方案，所以先看看他们如何更好地理解广播问题。这也将帮助你理解我们最后提出的解决方案。首先，广播主机有一些收听者都需要接收的信息。最简单的解决方案是广播主机保存所有收听者的列表并向每个收听者发送单独的消息。

暴力解决方案是广播主机发送广播消息的单个副本，并让路由器整理出来。在这种情况下，最简单的解决方案是称为 不受控泛洪 的策略。洪水策略工作如下。每个消息开始于将存活时间（ttl）值设置为大于或等于广播主机与其最远听者之间的边数量的某个数。每个路由器获得消息的副本，并将消息传递到其所有相邻路由器。当消息传递到 ttl 减少。每个路由器继续向其所有邻居发送消息的副本，直到 ttl 值达到 0。不受控制的洪泛比我们的第一个策略产生更多的不必要的消息。

这个问题的解决方案在于建立最小权重 生成树 。正式地，我们为图 G =（V，E）定义最小生成树 T 如下。 T 是连接 V 中所有顶点的 E 的非循环子集。 T中的边的权重的和被最小化。

我们将用来解决这个问题的算法称为 Prim 算法。 Prim 算法属于称为 “贪婪算法” 一系列算法，，因为在每个步骤，我们将选择最小权重的下一步。 在这种情况下，最小权重的下一步是以最小的权重跟随边。 我们的最后一步是开发 Prim 算法。
构建生成树的基本思想如下：

While T is not yet a spanning tree
	Find an edge that is safe to add to the tree
	Add the new edge to T

诀窍是指导我们 “找到一个安全的边”。我们定义一个安全边作为将生成树中的顶点连接到不在生成树中的顶点的任何边。这确保树将始终保持为树并且没有循环。

```python
from pythonds.graphs import PriorityQueue, Graph, Vertex

def prim(G,start):
    pq = PriorityQueue()
    for v in G:
        v.setDistance(sys.maxsize)
        v.setPred(None)
    start.setDistance(0)
    pq.buildHeap([(v.getDistance(),v) for v in G])
    while not pq.isEmpty():
        currentVert = pq.delMin()
        for nextVert in currentVert.getConnections():
          newCost = currentVert.getWeight(nextVert)
          if nextVert in pq and newCost<nextVert.getDistance():
              nextVert.setPred(currentVert)
              nextVert.setDistance(newCost)
              pq.decreaseKey(nextVert,newCost)
```

