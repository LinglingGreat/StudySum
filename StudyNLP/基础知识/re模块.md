re模块

1. search/match/fullmatch/split/findall
2. re模块调用方法和Pattern对象区别
3. match对象（子组匹配、命名子组）
4. 例子（最好是每个函数都有例子，然后要有若干大段文本的实际案例，可以是html解析或者其他相关内容）

re模块是Python的内置模块，无需安装。

re模块的调用方式：import re。

re库默认采用贪婪匹配，即输出匹配最长的子串。

模式和被搜索的字符串既可以是Unicode字符串(str)，也可以是8位字节串(bytes)。但是两者不能混用，即不能用一个字节串模式去匹配Unicode字符串，反之亦然。替换操作也是一样的，替换字符串的类型、模式、搜索字符串的类型必须一致。

正则表达式使用反斜杠字符`\`来进行转义，使得某些特殊字符能够表示其本身的含义。比如`.`表示匹配除换行符以外的任意字符，`\.`则表示`.`本身。

而Python中也会使用反斜杠字符来进行转义，这就可能造成困扰。比如要匹配`\`，正则表达式可以表示成`\\`，这里使用反斜杠转移。而在Python里每个反斜杠又必须表示成`\\`，因此最终的模式字符串就是`\\\\`。这未免过于复杂。

解决办法是使用Python的原生字符串表示法，在字符串前加'r'。原生字符串中的每个字符都表示它本身的含义。因此`r"\n"`表示`\`和`n`两个字符的字符串，而`"\n"`则表示一个换行符的字符串。上述反斜杠的模式就可以表示成`r"\\"`。



## 一、常用函数说明

### (1)re.search(pattern, string, flags=0)

`re.search()` 在一个字符串中**搜索匹配正则表达式的第一个位置**。

如果函数匹配成功则会返回一个匹配对象(Match对象，后面会详细说明)，否则返回None。

**参数说明**：pattern是匹配的正则表达式。string是要匹配的字符串。flags是正则表达式使用时的控制标记，后面会详细说明。

函数的用途：可以用来进行数据验证，在正则表达式两端加上`\A`和`\Z`，并判断返回值是否为None

例如：验证字符串是否为6位数字

```python
# 数据验证
# ^和$，\A和\Z一般成对出现
re.search(r"\A\d{6}\Z", "123456\n") != None   # False
re.search(r"\A\d{6}\Z", "123456") != None   # True
```

作业：验证输入的文本是否为合法的日期格式。规定合法的日期格式为"yyyy-mm-dd"。月份和日期可以不规范写，例如2020-12-2和2020-2-12也是合法的。不考虑日期是否真实存在。要求表达式如果合法返回True，否则返回False。

答案：

```python
re.search(r"^\d{4}-((1[0-2])|(0?[1-9]))-(([1-2][0-9])|(0?[1-9])|(3[0-1]))$", "2020-12-12") != None
```

### (2)re.match(pattern, string, flags=0)

`re.match()` 从一个字符串的**起始位置**匹配正则表达式，如果不是起始位置匹配成功的话就返回None

参数和`re.search`的参数一样。

注意即便是 `MULTILINE`多行模式（多行模式下`^`会将给定字符串的每行当作匹配开始）， `re.match()` 也只匹配字符串的开始位置，而不匹配每行开始。



### (3)re.fullmatch(pattern, string, flags=0)

`re.fullmatch()`如果整个字符串匹配到正则表达式，就返回一个相应的匹配对象，否则返回None。

### (4)re.split(pattern, string, maxsplit=0, flags=0)

`re.split()` 将一个字符串按照正则表达式匹配结果进行**分割**，返回列表类型。

maxsplit是分隔次数，例如maxsplit=1表示分隔一次，默认为 0，表示不限制次数。 如果是负数，表示不做任何切分。

例如：

```python
re.split(r'\W+', 'Words, words, words.')
# ['Words', 'words', 'words', '']
re.split(r'(\W+)', 'Words, words, words.')
# ['Words', ', ', 'words', ', ', 'words', '.', '']
re.split(r'\W+', 'Words, words, words.', 1)
# ['Words', 'words, words.']
re.split('[a-f]+', '0a3B9', flags=re.IGNORECASE)  # re.IGNORECASE表示忽略大小写
# ['0', '3', '9']
```

加上()会使得分割后的列表保留分隔符，不加的话就不会保留。

如果分隔符里有捕获组合，并且匹配到字符串的开始，那么结果将会以一个空字符串开始。对于结尾也是一样

```python
re.split(r'(\W+)', '...words, words...')
# ['', '...', 'words', ', ', 'words', '...', '']
```

样式的空匹配将分开字符串，但只在不相临的状况生效。

```python
re.split(r'\b', 'Words, words, words.')
# ['', 'Words', ', ', 'words', ', ', 'words', '.']
re.split(r'\W*', '...words...')
# ['', '', 'w', 'o', 'r', 'd', 's', '', '']
re.split(r'(\W*)', '...words...')
# ['', '...', '', '', 'w', '', 'o', '', 'r', '', 'd', '', 's', '...', '', '', '']
```

作业：以数字和空格（即空格和数字都必须有）作为分隔符分割这句话"我有1只松鼠 2条狗  66只猫 还有鱼7 信不信"，使得分割后结果为['我有1只松鼠', '2条狗', '66只猫 还有鱼7', '信不信']

答案：

```python
name = "我有1只松鼠 2条狗  66只猫 还有鱼7 信不信"
re.split('(?<=\d)\s+|\s+(?=\d)', name)
# ['我有1只松鼠', '2条狗', '66只猫 还有鱼7', '信不信']
```

看到结果是：有空格+数字，或者数字+空格的地方都被分隔开了。且分隔后将空格去掉，数字保留。

`(?<=\d)\s+`是查找数字后面的空格（可以是多个），`\s+(?=\d)`是查找数字前面的空格（可以是多个）。

### (5)re.findall(pattern, string, flags=0)

`re.findall()` **搜索**字符串，以列表类型返回全部不重复的能匹配的子串。如果样式里存在一到多个组，就返回一个组合列表；就是一个元组的列表（如果样式里有超过一个组合的话）。空匹配也会包含在结果里。

例如寻找文本中出现的所有英文字母：

```python
name = "abc 123 def 456"
re.findall(r"[a-zA-Z]", name)
# ['a', 'b', 'c', 'd', 'e', 'f']
```



###(6)re.finditer(pattern, string, flags=0)

`re.finditer()` **搜索**字符串，返回一个非重复匹配结果的迭代类型，每个迭代元素是match对象。

例如寻找关键词在文本中出现的所有起始位置：

```python
name = "123123"
keyword = "123"
start_list = [m.start() for m in re.finditer(keyword, name)]
start_list
# [0, 3]
```

如果只想要第一个起始位置，可以用：

```python
name.find(keyword)
# 0
```

### (7)re.sub(pattern, repl, string, count=0, flags=0)

`re.sub()` 在一个字符串中**替换**所有匹配正则表达式的子串，返回替换后的字符串

参数说明

- pattern是匹配的正则表达式
- repl : 替换的字符串，也可为一个函数。 例如`def capitalize(match): return match.group(1).upper+match.group(2).lower()`
- string是要被查找替换的字符串
- count : 模式匹配后替换的最大次数，默认 0 表示替换所有的匹配。count必须是非负整数。 

例如

```python
re.sub(r'def\s+([a-zA-Z_][a-zA-Z_0-9]*)\s*\(\s*\):',
        r'static PyObject*\npy_\1(void)\n{',
        'def myfunc():')
# 'static PyObject*\npy_myfunc(void)\n{'
```

这里的pattern是匹配一个def函数，将其替换成`static PyObject*\npy_\1(void)\n{`，这里面的`\1`是pattern匹配字符串的第一个子表达式，即`([a-zA-Z_][a-zA-Z_0-9]*)`匹配到的内容，在这里就是`myfunc`。

再来看一个替换字符串repl为函数的例子：

```python
def dashrepl(matchobj):
    if matchobj.group(0) == '-': return ' '
    else: return '-'
re.sub('-{1,2}', dashrepl, 'pro----gram-files')
# 'pro--gram files'

re.sub(r'\sAND\s', ' & ', 'Baked Beans And Spam', flags=re.IGNORECASE)
# 'Baked Beans & Spam'
```

`-{1,2}`匹配到两个`--`和一个`-`，通过函数dashrepl将连续的`--`替换成`-`，将`-`替换成空格，因此得到`pro--gram files`。

第二个例子中，`\sAND\s`匹配到And及其前后的2个空格，将其替换成' & '。

另外，**空匹配只在不相临连续的情况被更替**，所以 `re.sub('x*', '-', 'abxd')` 返回 `'-a-b--d-'` 。



作业1：在中英文字符之间加上空格

例如文本"今天真开心happy快乐"，加上空格后变成"今天真开心 happy 快乐"

答案：

```python
name = "今天真开心happy快乐"
name = re.sub('([A-Za-z]+)([\u4e00-\u9fa5]+)', r'\1 \2', name)
name = re.sub('([\u4e00-\u9fa5]+)([A-Za-z]+)', r'\1 \2', name)
name
# '今天真开心 happy 快乐'
```

两个表达式是类似地，只不过第一个表达式是在英文和中文字符之间加上空格，第二个表达式是在中文和英文字符之间加上空格。我们来看第一个表达式。第一个()是匹配1个或多个英文，第二个()是匹配1个或多个中文字符，r'\1 \2'是匹配第1个分组和第2个分组，且中间加了空格。

作业2：将格式如"年-月-日"的日期表示法替换成"日/月/年"的方法表示。

答案：

```python
re.sub("(\d{4})-(\d{2})-(\d{2})", r"\2/\3/\1", "2010-12-22")
# 12/22/2010
re.sub("(\d{4})-(\d{2})-(\d{2})", r"\1年\2年\3日", "2010-12-22")
# 2010年12月22日
```

因为`\1`,`\2`不是字符串中的合法转义序列，所以必须指定为原生字符串，在字符串前面加一个"r"。

如果想引用整个表达式匹配的文本，不能使用`\0`，因为`\0`开头的转义序列通常表示用八进制形式表示的字符，`\0`本身表示ASCII字符编码为0的字符。如果一定要引用整个表达式匹配的文本，则可以稍加变通，给整个表达式加上一对括号，之后用`\1`来引用。

```python
re.sub("((\d{4})-(\d{2})-(\d{2}))", "[\\1]", "2010-12-22")
# [2010-12-22]
re.sub("((\d{4})-(\d{2})-(\d{2}))", r"[\1]", "2010-12-22")
# [2010-12-22]
```

`\num`的用法可能产生二义性，例如`\10`是表示第10个捕获分组还是第1个捕获分组之后跟着一个字符0呢？

在Python中，`\10`会被解释成第10个捕获分组匹配的文本，下面的程序会报错：

```python
print(re.sub(r"(\d)", r"\10", "123"))
```

如果希望效果是第1个捕获分组之后跟着一个字符0，那就要写成`\g<1>0`

```python
re.sub(r"(\d)", r"\g<1>0", "123")
# 102030
```

### (8)re.subn(pattern, repl, string, count=0, flags=0)

行为与 `sub()`相同，但是返回一个元组 `(字符串, 替换次数)`.

```python
re.subn(r'\sAND\s', ' & ', 'Baked Beans And Spam', flags=re.IGNORECASE)
# ('Baked Beans & Spam', 1)
```

### (9)re.escape(pattern)

转义 pattern 中的特殊字符。

```python
print(re.escape('http://www.python.org'))
# http://www\.python\.org

legal_chars = string.ascii_lowercase + string.digits + "!#$%&'*+-.^_`|~:"
print('[%s]+' % re.escape(legal_chars))
# [abcdefghijklmnopqrstuvwxyz0123456789!\#\$%\&'\*\+\-\.\^_`\|\~:]+

operators = ['+', '-', '*', '/', '**']
print('|'.join(map(re.escape, sorted(operators, reverse=True))))
# /|\-|\+|\*\*|\*
```



### (10)re.purge()

清除正则表达式的缓存。

## 二、Match对象

匹配对象支持以下方法和属性

### (1)Match.expand(template)

对template进行反斜杠转义替换并且返回。转义如同 `\n` 被转换成合适的字符，数字引用(`\1`, `\2`)和命名组合(`\g<1>`, `\g<name>`) 替换为相应组合的内容。

```python
m = re.match(r"(\w+) (\w+)", "Isaac Newton, physicist")
m.expand(r"\1\n\2")
# 'Isaac\nNewton'
```



###(2)Match.group([group1, ...])

返回一个或者多个匹配的子组。如果只有一个参数，结果就是一个字符串；如果有多个参数，结果就是一个元组；如果没有参数，整个匹配都被返回。如果一个组N 参数值为 0，相应的返回值就是整个匹配字符串；如果它是一个范围 [1..99]，结果就是相应的括号组字符串。如果一个组号是负数，或者大于样式中定义的组数，一个 [`IndexError`](https://docs.python.org/zh-cn/3.9/library/exceptions.html#IndexError) 索引错误就 `raise`。如果一个组包含在样式的一部分，并被匹配多次，就返回最后一个匹配。

```python
m = re.match(r"(\w+) (\w+)", "Isaac Newton, physicist")
m.group(0)
# 'Isaac Newton'
m.group(1)
# 'Isaac'
m.group(2)
# 'Newton'
m.group(1, 2)
# ('Isaac', 'Newton')
```

另一个例子：

```python
re.search("(\d{4})-(\d{2})-(\d{2})", "2010-12-22").group(1)
# 2010
re.search("(\d{4})-(\d{2})-(\d{2})", "2010-12-22").group(2)
# 12
re.search("(\d{4})-(\d{2})-(\d{2})", "2010-12-22").group(3)
# 22
re.search("(\d{4})-(\d{2})-(\d{2})", "2010-12-22").group(0)
# 2010-12-22
re.search("(\d{4})-(\d{2})-(\d{2})", "2010-12-22").group()
# 2010-12-22
```

注意，如果被匹配多次，就返回最后一个匹配：

```python
re.search("(\d{4})-(\d{2})-(\d{2})", "2010-12-22").group(1)
# 2010
re.search("(\d){4}-(\d{2})-(\d{2})", "2010-12-22").group(1)
# 0
# 编号为1的分组匹配的文本的值依次是2、0、1、0，最后的结果是0
```

上面的分组编号是1、2、3等数字编号，当分组多了之后难以记忆，而且很难对应到希望捕获的相应分组上。这时可以用命名分组。

命名分组也就是标识分组为容易记忆和辨别的名字，而不是数字编号。

Python中用`(?P<name>regex)`来分组。例如：

```python
namedRegex = r"(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})"
result = re.search(namedRegex, "2010-12-22")
print(result.group("year"))  # 2010，等价于result.group(1)
print(result.group("month"))  # 12，等价于result.group(2)
print(result.group("day"))  # 22，等价于result.group(3)
```

即便使用了命名分组，每个命名分组同时也具有数字编号。

Python中，如果使用了命名分组，在表达式反向引用时，必须使用`(?P=name)`的记法，而要进行正则表达式替换，则要写作`\g<name>`

```python
re.search(r"^(?P<char>[a-z])(?P=char)$", "aa") != None  #True
re.sub("(?P<digit>\d)", r"\g<digit>0", "123")  # 102030
```



###(3)Match.\__getitem__(g)

等价于m.group(g)

```python
m = re.match(r"(\w+) (\w+)", "Isaac Newton, physicist")
m[0]      
#'Isaac Newton'
m[1]      
#'Isaac'
m[2]       
#'Newton'
```

###(4)Match.groups(default=None)

返回一个元组，包含所有匹配的子组，在样式中出现的从1到任意多的组合。 *default* 参数用于不参与匹配的情况，默认为 `None`。

```python
m = re.match(r"(\d+)\.(\d+)", "24.1632")
m.groups()
# ('24', '1632')
```

如果我们使点号可选，那么不是所有的组都会有匹配结果。这些组合默认会返回一个 `None` ，除非指定了 *default* 参数。

```python
m = re.match(r"(\d+)\.?(\d+)?", "24")
m.groups()
# ('24', None)
m.groups('0')
# ('24', '0')
```

###(5)Match.groupdict(default=None)

返回一个字典，包含了所有的 *命名* 子组。key就是组名。 *default* 参数用于不参与匹配的组合；默认为 `None`。 例如

```python
m = re.match(r"(?P<first_name>\w+) (?P<last_name>\w+)", "Malcolm Reynolds")
m.groupdict()
# {'first_name': 'Malcolm', 'last_name': 'Reynolds'}
```

###(6)Match.start(group=0)和Match.end(group=0)

返回 *group* 匹配到的字串的开始和结束位置。*group* 默认为0（意思是整个匹配的子串）。如果 *group* 存在，但未产生匹配，就返回 `-1` 。

```python
email = "tony@tiremove_thisger.net"
m = re.search("remove_this", email)
email[:m.start()] + email[m.end():]
# 'tony@tiger.net'
```

###(7)Match.span(group=0)

对于一个匹配 *m* ， 返回一个二元组 `(m.start(group), m.end(group))` 。 注意如果 *group* 没有在这个匹配中，就返回 `(-1, -1)` 。*group* 默认为0，就是整个匹配。

```python
email = "tony@tiremove_thisger.net"
m = re.search("remove_this", email)
m.span()
# (7, 18)
email[m.span()[0]:m.span()[1]]
# 'remove_this'
```



###(8)Match.pos和Match.endpos

pos和endpos的值，会传递给 `search()` 或 `match()`的方法一个正则对象。这个是正则引擎开始（对应pos）或停止（对应endpos）在字符串搜索一个匹配的索引位置。

###(9)Match.lastindex

捕获组的最后一个匹配的整数索引值，如果没有匹配产生的话就是 `None` 。比如，对于字符串 `'ab'`，表达式 `(a)b`, `((a)(b))`, 和 `((ab))` 将得到 `lastindex == 1` ， 而 `(a)(b)` 会得到 `lastindex == 2` 。

###(10)Match.lastgroup

最后一个匹配的命名组名字，如果没有产生匹配的话就是 `None` 。

###(11)Match.re

返回产生这个实例的正则对象， 这个实例是由 正则对象的 `match()`或 `search()`方法产生的。

```python
email = "tony@tiremove_thisger.net"
m = re.search("remove_this", email)
m.re
# re.compile(r'remove_this', re.UNICODE)
```



###(12)Match.string

传递到match()或search()的字符串。

```python
email = "tony@tiremove_thisger.net"
m = re.search("remove_this", email)
m.string
# "tony@tiremove_thisger.net"
```



## 三、flags参数

flags参数是正则表达式使用时的控制标记。控制标记用来改变表达式的行为模式，可以用模式修饰符来指定，也可以用预定义的常量（也就是flags的取值）作为特殊参数传入来指定。

模式修饰符即模式名称对应的单个字符，使用时将其填入特定结构`(?modifier)`中(modifier为模式修饰符)，嵌在正则表达式的开头。

模式修饰符写在最开头表示整个正则表达式都指定此模式，如果出现在中间，则表示此模式从这里开始生效；如果出现在某个括号内，那么它的作用范围只限于括号内部。Python的情况不同，不管出现在哪个位置，都对整个正则表达式生效。

flags参数的可取值及其对应的模式修饰符有以下几种：

1.re.A或re.ASCII: 对应模式修饰符`a`。让 `\w`, `\W`, `\b`, `\B`, `\d`, `\D`, `\s` 和 `\S` 只匹配ASCII，而不是Unicode。Python3以上的版本中，正则表达式默认采用Unicode匹配规则，如果希望让`\d`, `\w`等字符组简记法恢复到ASCII匹配规则，可以使用此模式。

2.re.DEBUG: 显示编译时的debug信息。

3.re.I或re.IGNORECASE: 对应模式修饰符`i`。忽略正则表达式的大小写，使得正则表达式如`[A‐Z]`能够匹配小写字符。Unicode匹配（比如 `Ü` 匹配 `ü`）同样有用，除非设置了`re.ASCII`标记来禁用非ASCII匹配。

4.re.L或re.LOCALE: 对应模式修饰符 `L` 。由当前语言区域决定 `\w`, `\W`, `\b`, `\B` 和大小写敏感匹配。这个标记只能对byte样式有效。这个标记不推荐使用，因为语言区域机制很不可靠，它一次只能处理一个"习惯"，而且只对8位字节有效。Unicode匹配在Python 3 里默认启用，并可以处理不同语言。

5.re.M或re.MULTILINE: 对应模式修饰符`m`。多行匹配，影响`^`和`$`。这种模式下正则表达式中的`^`操作符能够将给定字符串的每行当作匹配开始。默认情况下`^`是匹配整个字符串的开始。

一个多行模式的例子，这里是匹配每一行以数字开头的内容。

```python
multilineString = "1 line\nNot digit\n2 line"
lineBeginWithDigitRegex = r"(?m)^\d.*"
for line in re.findall(lineBeginWithDigitRegex, multilineString):
    print(line)
# 1 line
# 2 line
```

等价于

```python
multilineString = "1 line\nNot digit\n2 line"
lineBeginWithDigitRegex = r"^\d.*"
for line in re.findall(lineBeginWithDigitRegex, multilineString, re.M):
    print(line)
# 1 line
# 2 line
```

如果不使用多行模式的话，只能匹配到`1 line`即第一行的内容。

6.re.S或re.DOTALL: 对应模式修饰符 `s` 。使得正则表达式中的`.`操作符能够匹配所有字符，注意`.`默认匹配除换行外的所有字符。这也称为单行模式。单行模式下所有文本似乎只在一行里，换行符变成了普通字符，因此点号可以匹配。

7.re.U或re.UNICODE：对应模式修饰符 `u` 。此模式下，`\w`, `\d`, `\s`等字符组简记法的匹配规则会发生改变，比如`\w`能匹配Unicode中的“单词字符”，包括中文字符，`\d`也能匹配１、２之类的全角数字字符。在Python3中默认字符串已经是Unicode了，所以这个模式是冗余的。

8.re.X或re.VERBOSE: 对应模式修饰符 `x` 。这个标记允许你编写更具可读性更友好的正则表达式。通过分段和添加注释。空白符号会被忽略，除非在一个字符集合当中或者由反斜杠转义，或者在 `*?`, `(?:` or `(?P<…>` 分组之内。当一个行内有 `#` 不在字符集和转义序列，那么它之后的所有字符都是注释。

意思就是下面两个正则表达式等价地匹配一个十进制数字：

```python
a = re.compile(r"""\d +  # the integral part
                   \.    # the decimal point
                   \d *  # some fractional digits""", re.X)
b = re.compile(r"\d+\.\d*")
```

如果记不住这么多取值也没关系，只需要知道常用的几个就可以了，即不区分大小写模式re.I、单行模式re.S、多行模式re.M、注释模式re.X。

如果需要同时使用多种模式，只要将模式修饰符排列起来就可以了。比如`(?mx)`表示同时使用多行模式和注释模式。或者将预定义常量用`|`组合起来，比如`re.M|re.X`。

## 四、re.compile函数

compile函数将正则表达式编译成一个**正则表达式对象**，可以用于匹配。

函数语法：`re.compile(pattern, flags=0)`

pattern是正则表达式，flags的含义和re.search中的一样。

```python
prog = re.compile(pattern)
result = prog.match(string)
```

等价于

```python
result = re.match(pattern, string)
```

如果正则表达式需要多次使用的话，用`re.compile()`保存正则表达式对象更加高效。

编译后的正则表达式对象支持以下方法和属性：

(1) Pattern.search(string, pos=0, endpos=len(string))

作用和re.search()类似。参数pos给出了字符串中开始搜索的位置索引，参数endpos限定了字符串搜索的结束。即只有从pos到endpos-1的字符会被匹配。如果endpos小于pos，就不会有匹配产生。

```python
pattern = re.compile("d")
pattern.search("dog")  # 匹配，"d"出现在"dog"的第0个位置
# 输出：<re.Match object; span=(0, 1), match='d'>
pattern.search("dog", 1)  # 不匹配，"d"没有出现在字符串"og"中
```



(2)Pattern.match(string, pos=0, endpos=len(string))

作用和re.match()类似。参数含义和Pattern.search()含义相同。

```python
pattern = re.compile("o")
pattern.match("dog")   #不匹配，因为"o"不是"dog"的起始位置
pattern.match("dog", 1)  # 匹配
# 输出：<re.Match object; span=(1, 2), match='o'>
```

(3)Pattern.fullmatch(string, pos=0, endpos=len(string))

作用和re.fullmatch()类似。参数含义和Pattern.search()含义相同。

```python
pattern = re.compile("o[gh]")
pattern.fullmatch("dog")  # 整个字符串与正则表达式不匹配
pattern.fullmatch("ogre")  # 整个字符串与正则表达式不匹配
pattern.fullmatch("doggie", 1, 3) # "og"和正则表达式匹配
# 输出：<re.Match object; span=(1, 3), match='og'>
```

下面的几个函数都和对应的re函数类似，不再赘述。

(4)Pattern.split(string, maxsplit=0)

(5)Pattern.findall(string, pos=0, endpos=len(string))

(6)Pattern.finditer(string, pos=0, endpos=len(string))

(7)Pattern.sub(repl, string, count=0)

(8)Pattern.subn(repl, string, count=0)

(9)Pattern.flags

正则匹配标记，这是可以传递给compile()的参数。

(10)Pattern.groups

捕获到的模式串中组的数量

(11)Pattern.groupindex

映射由 `(?P<id>)` 定义的命名符号组合和数字组合的字典。如果没有符号组，那字典就是空的。

(12)Pattern.pattern

编译对象的原始样式字符串。



re.compile()还可以用来观察某个正则表达式的详细信息，方法是指定第二个参数为re.DEBUG。如果遇到复杂的表达式，或者不确定某个表达式的意义，可以通过它来观察。

```python
re.compile("ab|cd", re.DEBUG)
# 会输出下列信息
BRANCH
  LITERAL 97
  LITERAL 98
OR
  LITERAL 99
  LITERAL 100
re.compile(r'ab|cd', re.UNICODE|re.DEBUG)
```

缩进表示了表达式各结构的层级关系，而字符本身则显示为其码值的十进制表示(比如字符a的码值是十进制的97)。

## 五、其它

### 条件匹配

Python的正则表达式支持条件匹配。语法是`(?(id/name)yes-pattern|no-pattern)`。其中id/name是对应捕获分组的名称或者编号，如果该捕获分组成功匹配文本，则后续匹配交由yes-pattern来完成，否则交由no-pattern来完成。no-pattern可以省略。

假设我们需要验证价格：如果前面没有美元符号$，则价格只能包含整数部分，否则还应当包含小数点和两位小数。

```python
regex = r"^(\$)?[0-9]+(?(1)\.[0-9]{2}|)$"
re.search(regex, "34") != None   # True
re.search(regex, "12.00") != None   # False
re.search(regex, "$34") != None   # False
re.search(regex, "$12.00") != None   # True
```

其中`(?(1)\.[0-9]{2}|)`就是一个条件匹配表达式，1对应的是`(\$)`捕获到的内容。如果起始是`$`符号，则后面的价格部分除了整数部分外，还必须包含小数点和2位小数，即`\.[0-9]{2}`；如果起始不是`$`符号，价格只能包含整数部分，不能有其它的。这里的no-pattern是省略的。



## 参考资料

https://www.runoob.com/python/python-reg-expressions.html

https://docs.python.org/zh-cn/3.9/library/re.html