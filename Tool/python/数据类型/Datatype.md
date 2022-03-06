## 列表相关方法
< list > . append ( x ) 将元素x增加到列表的最后  
< list > . sort ( ) 将列表元素排序  
< list > . reverse ( ) 将序列元素反转  
< list > . index ( ) 返回第一次出现元素x的索引值  
< list > . insert ( i, x ) 在位置i处插入新元素x  
< list > . count ( x ) 返回元素x在列表中的数量  
< list > . remove ( x ) 删除列表中第一次出现的元素x  
< list > . pop ( i ) 取出列表中位置i的元素，并删除它   

两个列表的并集：a or b

交集：a and b

列表推导-嵌套

```
##不推荐
for sub_list in nested_list:
    if list_condition(sub_list):
        for item in sub_list:
            if item_condition(item):
                # do something...  
##推荐
gen = (item for sl in nested_list if list_condition(sl) \
            for item in sl if item_condition(item))
for item in gen:
    # do something...
```

循环嵌套

```
##不推荐
for x in x_list:
    for y in y_list:
        for z in z_list:
            # do something for x &amp;amp; y  

##推荐
from itertools import product
for x, y, z in product(x_list, y_list, z_list):
    # do something for x, y, z
```

尽量使用生成器代替列表

```
##不推荐
def my_range(n):
    i = 0
    result = []
    while i &amp;lt; n:
        result.append(fn(i))
        i += 1
    return result  #  返回列表

##推荐
def my_range(n):
    i = 0
    result = []
    while i &amp;lt; n:
        yield fn(i)  #  使用生成器代替列表
        i += 1
*尽量用生成器代替列表，除非必须用到列表特有的函数。
```



## 字符串处理方法
'+' 连接  
'*' 重复  
<string&gt;[ ] 索引  
<string&gt;[ : ] 剪切  
len(<string&gt;) 长度  
<string&gt;.upper() 字符串中字母大写  
<string&gt;.lower() 字符串中字母小写  
<string&gt;.strip() 去两边空格及去指定字符  
<string&gt;.split() 按指定字符分割字符串为数组  
<string&gt;.join() 连接两个字符串序列  
<string&gt;.find() 搜索指定字符串  
<string&gt;.replace() 字符串替换  
for <var&gt; in <string&gt; 字符串迭代  

| capitalize()                             | 把字符串的第一个字符改为大写                           |
| ---------------------------------------- | ---------------------------------------- |
| casefold()                               | 把整个字符串的所有字符改为小写                          |
| center(width)                            | 将字符串居中，并使用空格填充至长度 width 的新字符串            |
| count(sub[, start[, end]])               | 返回 sub 在字符串里边出现的次数，start 和 end 参数表示范围，可选。 |
| encode(encoding='utf-8', errors='strict') | 以 encoding 指定的编码格式对字符串进行编码。              |
| endswith(sub[, start[, end]])            | 检查字符串是否以 sub 子字符串结束，如果是返回 True，否则返回 False。start 和 end 参数表示范围，可选。 |
| expandtabs([tabsize=8])                  | 把字符串中的 tab 符号（\t）转换为空格，如不指定参数，默认的空格数是 tabsize=8。 |
| find(sub[, start[, end]])                | 检测 sub 是否包含在字符串中，如果有则返回索引值，否则返回 -1，start 和 end 参数表示范围，可选。 |
| index(sub[, start[, end]])               | 跟 find 方法一样，不过如果 sub 不在 string 中会产生一个异常。 |
| isalnum()                                | 如果字符串至少有一个字符并且所有字符都是字母或数字则返回 True，否则返回 False。 |
| isalpha()                                | 如果字符串至少有一个字符并且所有字符都是字母则返回 True，否则返回 False。 |
| isdecimal()                              | 如果字符串只包含十进制数字则返回 True，否则返回 False。        |
| isdigit()                                | 如果字符串只包含数字则返回 True，否则返回 False。           |
| islower()                                | 如果字符串中至少包含一个区分大小写的字符，并且这些字符都是小写，则返回 True，否则返回 False。 |
| isnumeric()                              | 如果字符串中只包含数字字符，则返回 True，否则返回 False。       |
| isspace()                                | 如果字符串中只包含空格，则返回 True，否则返回 False。         |
| istitle()                                | 如果字符串是标题化（所有的单词都是以大写开始，其余字母均小写），则返回 True，否则返回 False。 |
| isupper()                                | 如果字符串中至少包含一个区分大小写的字符，并且这些字符都是大写，则返回 True，否则返回 False。 |
| join(sub)                                | 以字符串作为分隔符，插入到 sub 中所有的字符之间。              |
| ljust(width)                             | 返回一个左对齐的字符串，并使用空格填充至长度为 width 的新字符串。     |
| lower()                                  | 转换字符串中所有大写字符为小写。                         |
| lstrip()                                 | 去掉字符串左边的所有空格                             |
| partition(sub)                           | 找到子字符串 sub，把字符串分成一个 3 元组 (pre_sub, sub, fol_sub)，如果字符串中不包含 sub 则返回 ('原字符串', '', '') |
| replace(old, new[, count])               | 把字符串中的 old 子字符串替换成 new 子字符串，如果 count 指定，则替换不超过 count 次。 |
| rfind(sub[, start[, end]])               | 类似于 find() 方法，不过是从右边开始查找。                |
| rindex(sub[, start[, end]])              | 类似于 index() 方法，不过是从右边开始。                 |
| rjust(width)                             | 返回一个右对齐的字符串，并使用空格填充至长度为 width 的新字符串。     |
| rpartition(sub)                          | 类似于 partition() 方法，不过是从右边开始查找。           |
| rstrip()                                 | 删除字符串末尾的空格。                              |
| split(sep=None, maxsplit=-1)             | 不带参数默认是以空格为分隔符切片字符串，如果 maxsplit 参数有设置，则仅分隔 maxsplit 个子字符串，返回切片后的子字符串拼接的列表。 |
| splitlines(([keepends]))                 | 按照 '\n' 分隔，返回一个包含各行作为元素的列表，如果 keepends 参数指定，则返回前 keepends 行。 |
| startswith(prefix[, start[, end]])       | 检查字符串是否以 prefix 开头，是则返回 True，否则返回 False。start 和 end 参数可以指定范围检查，可选。 |
| strip([chars])                           | 删除字符串前边和后边所有的空格，chars 参数可以定制删除的字符，可选。    |
| swapcase()                               | 翻转字符串中的大小写。                              |
| title()                                  | 返回标题化（所有的单词都是以大写开始，其余字母均小写）的字符串。         |
| translate(table)                         | 根据 table 的规则（可以由 str.maketrans('a', 'b') 定制）转换字符串中的字符。 |
| upper()                                  | 转换字符串中的所有小写字符为大写。                        |
| zfill(width)                             | 返回长度为 width 的字符串，原字符串右对齐，前边用 0 填充。       |

## 字典
字典的遍历  
+ 遍历字典的键key  
  for key in dictionaryName.keys(): print.(key)
+ 遍历字典的值value  
  for value in dictionaryName.values(): print.(value)
+ 遍历字典的项  
  for item in dicitonaryName.items(): print.(item)
+ 遍历字典的key-value  
  for item，value in adict.items(): print(item, value)

字典方法  
* keys():tuple 返回一个包含字典所有Key的列表
* values():tuple 返回一个包含字典所有value的列表
* Items():tuple 返回一一个包含所有键值的列表
* clear():None 删除字典中的所有项目
* get(key):value 返回字典中key对应的值
* pop(key):val 删除并返回字典中key对应的值
* update(字典) 将字典中的键值添加到字典中

字典键值列表

```
##不推荐
for key in my_dict.keys():
    #  my_dict[key] ...  

##推荐
for key in my_dict:
    #  my_dict[key] ...

# 只有当循环中需要更改key值的情况下，我们需要使用 my_dict.keys()
# 生成静态的键值列表。
```

字典键值判断

```
##不推荐
if my_dict.has_key(key):
    # ...do something with d[key]  

##推荐
if key in my_dict:
    # ...do something with d[key]
```

字典 get 和 setdefault 方法

```
##不推荐
navs = {}
for (portfolio, equity, position) in data:
    if portfolio not in navs:
            navs[portfolio] = 0
    navs[portfolio] += position * prices[equity]
##推荐
navs = {}
for (portfolio, equity, position) in data:
    # 使用 get 方法
    navs[portfolio] = navs.get(portfolio, 0) + position * prices[equity]
    # 或者使用 setdefault 方法
    navs.setdefault(portfolio, 0)
    navs[portfolio] += position * prices[equity]
```


## 集合

**集合类型内建方法总结**

| **集合（s）.方法名**                    | **等价符号** | **方法说明**                                 |
| -------------------------------- | -------- | ---------------------------------------- |
| s.issubset(t)                    | s <= t   | 子集测试（允许不严格意义上的子集）：s 中所有的元素都是 t 的成员       |
|                                  | s < t    | 子集测试（严格意义上）：s != t 而且 s 中所有的元素都是 t 的成员   |
| s.issuperset(t)                  | s >= t   | 超集测试（允许不严格意义上的超集）：t 中所有的元素都是 s 的成员       |
|                                  | s > t    | 超集测试（严格意义上）：s != t 而且 t 中所有的元素都是 s 的成员   |
| s.union(t)                       | s \| t   | 合并操作：s "或" t 中的元素                        |
| s.intersection(t)                | s & t    | 交集操作：s "与" t 中的元素                        |
| s.difference                     | s - t    | 差分操作：在 s 中存在，在 t 中不存在的元素                 |
| s.symmetric_difference(t)        | s ^ t    | 对称差分操作：s "或" t 中的元素，但不是 s 和 t 共有的元素      |
| s.copy()                         |          | 返回 s 的拷贝（浅复制）                            |
| **以下方法仅适用于可变集合**                 |          |                                          |
| s.update                         | s \|= t  | 将 t 中的元素添加到 s 中                          |
| s.intersection_update(t)         | s &= t   | 交集修改操作：s 中仅包括 s 和 t 中共有的成员               |
| s.difference_update(t)           | s -= t   | 差修改操作：s 中包括仅属于 s 但不属于 t 的成员              |
| s.symmetric_difference_update(t) | s ^= t   | 对称差分修改操作：s 中包括仅属于 s 或仅属于 t 的成员           |
| s.add(obj)                       |          | 加操作：将 obj 添加到 s                          |
| s.remove(obj)                    |          | 删除操作：将 obj 从 s 中删除，如果 s 中不存在 obj，将引发异常   |
| s.discard(obj)                   |          | 丢弃操作：将 obj 从 s 中删除，如果 s 中不存在 obj，也没事儿^_^ |
| s.pop()                          |          | 弹出操作：移除并返回 s 中的任意一个元素                    |
| s.clear()                        |          | 清除操作：清除 s 中的所有元素                         |