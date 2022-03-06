## 1.Extended unpacking (Python 3 only)
>>> a, *b, c = [1, 2, 3, 4, 5]
>>> a
1
>>> b
[2, 3, 4]
>>> c
5

## 2.Naming slices (slice(start, end, step))
>>> a = [0, 1, 2, 3, 4, 5]
>>> LASTTHREE = slice(-3, None)
>>> LASTTHREE
slice(-3, None, None)
>>> a[LASTTHREE]
[3, 4, 5]

## 3.Zipping and unzipping lists and iterables
>>> a = [1, 2, 3]
>>> b = ['a', 'b', 'c']
>>> z = zip(a, b)
>>> z
[(1, 'a'), (2, 'b'), (3, 'c')]
>>> zip(*z)
[(1, 2, 3), ('a', 'b', 'c')]

## 4.Grouping adjacent list items using zip
>>> a = [1, 2, 3, 4, 5, 6]

>>> # Using iterators
>>> group_adjacent = lambda a, k: zip(*([iter(a)] * k))
>>> group_adjacent(a, 3)
[(1, 2, 3), (4, 5, 6)]
>>> group_adjacent(a, 2)
[(1, 2), (3, 4), (5, 6)]
>>> group_adjacent(a, 1)
[(1,), (2,), (3,), (4,), (5,), (6,)]


>>> # Using slices
>>> from itertools import islice
>>> group_adjacent = lambda a, k: zip(*(islice(a, i, None, k) for i in range(k)))
>>> group_adjacent(a, 3)
[(1, 2, 3), (4, 5, 6)]
>>> group_adjacent(a, 2)
[(1, 2), (3, 4), (5, 6)]
>>> group_adjacent(a, 1)
[(1,), (2,), (3,), (4,), (5,), (6,)]

## 5.Sliding windows (n-grams) using zip and iterators
>>> from itertools import islice
>>> def n_grams(a, n):
...     z = (islice(a, i, None) for i in range(n))
...     return zip(*z)
...
>>> a = [1, 2, 3, 4, 5, 6]
>>> n_grams(a, 3)
[(1, 2, 3), (2, 3, 4), (3, 4, 5), (4, 5, 6)]
>>> n_grams(a, 2)
[(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
>>> n_grams(a, 4)
[(1, 2, 3, 4), (2, 3, 4, 5), (3, 4, 5, 6)]

## 6.Inverting a dictionary using zip
>>> m = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
>>> m.items()
[('a', 1), ('c', 3), ('b', 2), ('d', 4)]
>>> zip(m.values(), m.keys())
[(1, 'a'), (3, 'c'), (2, 'b'), (4, 'd')]
>>> mi = dict(zip(m.values(), m.keys()))
>>> mi
{1: 'a', 2: 'b', 3: 'c', 4: 'd'}

## 7.Flattening lists:
>>> a = [[1, 2], [3, 4], [5, 6]]
>>> list(itertools.chain.from_iterable(a))
[1, 2, 3, 4, 5, 6]

>>> sum(a, [])
[1, 2, 3, 4, 5, 6]

>>> [x for l in a for x in l]
[1, 2, 3, 4, 5, 6]

>>> a = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
>>> [x for l1 in a for l2 in l1 for x in l2]
[1, 2, 3, 4, 5, 6, 7, 8]

>>> a = [1, 2, [3, 4], [[5, 6], [7, 8]]]
>>> flatten = lambda x: [y for l in x for y in flatten(l)] if type(x) is list else [x]
>>> flatten(a)
[1, 2, 3, 4, 5, 6, 7, 8]
Note: according to Python's documentation on sum, itertools.chain.from_iterable is the preferred method for this.

## 8.Generator expressions
>>> g = (x ** 2 for x in xrange(10))
>>> next(g)
0
>>> next(g)
1
>>> next(g)
4
>>> next(g)
9
>>> sum(x ** 3 for x in xrange(10))
2025
>>> sum(x ** 3 for x in xrange(10) if x % 3 == 1)
408

## 9.Named tuples (collections.namedtuple)
>>> Point = collections.namedtuple('Point', ['x', 'y'])
>>> p = Point(x=1.0, y=2.0)
>>> p
Point(x=1.0, y=2.0)
>>> p.x
1.0
>>> p.y
2.0

## 10.Inheriting from named tuples:
>>> class Point(collections.namedtuple('PointBase', ['x', 'y'])):
...     __slots__ = ()
...     def __add__(self, other):
...             return Point(x=self.x + other.x, y=self.y + other.y)
...
>>> p = Point(x=1.0, y=2.0)
>>> q = Point(x=2.0, y=3.0)
>>> p + q
Point(x=3.0, y=5.0)

## 11.Multisets and multiset operations (collections.Counter)
>>> A = collections.Counter([1, 2, 2])
>>> B = collections.Counter([2, 2, 3])
>>> A
Counter({2: 2, 1: 1})
>>> B
Counter({2: 2, 3: 1})
>>> A | B
Counter({2: 2, 1: 1, 3: 1})
>>> A & B
Counter({2: 2})
>>> A + B
Counter({2: 4, 1: 1, 3: 1})
>>> A - B
Counter({1: 1})
>>> B - A
Counter({3: 1})

## 12.Most common elements in an iterable (collections.Counter)
>>> A = collections.Counter([1, 1, 2, 2, 3, 3, 3, 3, 4, 5, 6, 7])
>>> A
Counter({3: 4, 1: 2, 2: 2, 4: 1, 5: 1, 6: 1, 7: 1})
>>> A.most_common(1)
[(3, 4)]
>>> A.most_common(3)
[(3, 4), (1, 2), (2, 2)]

## 13.Double-ended queue (collections.deque)
>>> Q = collections.deque()
>>> Q.append(1)
>>> Q.appendleft(2)
>>> Q.extend([3, 4])
>>> Q.extendleft([5, 6])
>>> Q
deque([6, 5, 2, 1, 3, 4])
>>> Q.pop()
4
>>> Q.popleft()
6
>>> Q
deque([5, 2, 1, 3])
>>> Q.rotate(3)
>>> Q
deque([2, 1, 3, 5])
>>> Q.rotate(-3)
>>> Q
deque([5, 2, 1, 3])

## 14.Double-ended queue with maximum length (collections.deque)
>>> last_three = collections.deque(maxlen=3)
>>> for i in xrange(10):
...     last_three.append(i)
...     print ', '.join(str(x) for x in last_three)
...
0
0, 1
0, 1, 2
1, 2, 3
2, 3, 4
3, 4, 5
4, 5, 6
5, 6, 7
6, 7, 8
7, 8, 9

## 15.Ordered dictionaries (collections.OrderedDict)
>>> m = dict((str(x), x) for x in range(10))
>>> print ', '.join(m.keys())
1, 0, 3, 2, 5, 4, 7, 6, 9, 8
>>> m = collections.OrderedDict((str(x), x) for x in range(10))
>>> print ', '.join(m.keys())
0, 1, 2, 3, 4, 5, 6, 7, 8, 9
>>> m = collections.OrderedDict((str(x), x) for x in range(10, 0, -1))
>>> print ', '.join(m.keys())
10, 9, 8, 7, 6, 5, 4, 3, 2, 1

## 16.
>>> m = dict()
>>> m['a']
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
KeyError: 'a'
>>>
>>> m = collections.defaultdict(int)
>>> m['a']
0
>>> m['b']
0
>>> m = collections.defaultdict(str)
>>> m['a']
''
>>> m['b'] += 'a'
>>> m['b']
'a'
>>> m = collections.defaultdict(lambda: '[default value]')
>>> m['a']
'[default value]'
>>> m['b']
'[default value]'

## 17.Using default dictionaries to represent simple trees
>>> import json
>>> tree = lambda: collections.defaultdict(tree)
>>> root = tree()
>>> root['menu']['id'] = 'file'
>>> root['menu']['value'] = 'File'
>>> root['menu']['menuitems']['new']['value'] = 'New'
>>> root['menu']['menuitems']['new']['onclick'] = 'new();'
>>> root['menu']['menuitems']['open']['value'] = 'Open'
>>> root['menu']['menuitems']['open']['onclick'] = 'open();'
>>> root['menu']['menuitems']['close']['value'] = 'Close'
>>> root['menu']['menuitems']['close']['onclick'] = 'close();'
>>> print json.dumps(root, sort_keys=True, indent=4, separators=(',', ': '))
{
    "menu": {
    ​    "id": "file",
    ​    "menuitems": {
    ​        "close": {
    ​            "onclick": "close();",
    ​            "value": "Close"
    ​        },
    ​        "new": {
    ​            "onclick": "new();",
    ​            "value": "New"
    ​        },
    ​        "open": {
    ​            "onclick": "open();",
    ​            "value": "Open"
    ​        }
    ​    },
    ​    "value": "File"
    }
}
(See https://gist.github.com/hrldcpr/2012250 for more on this.)

## 18.Mapping objects to unique counting numbers (collections.defaultdict)
>>> import itertools, collections
>>> value_to_numeric_map = collections.defaultdict(itertools.count().next)
>>> value_to_numeric_map['a']
0
>>> value_to_numeric_map['b']
1
>>> value_to_numeric_map['c']
2
>>> value_to_numeric_map['a']
0
>>> value_to_numeric_map['b']
1

## 19.Largest and smallest elements (heapq.nlargest and heapq.nsmallest)
>>> a = [random.randint(0, 100) for __ in xrange(100)]
>>> heapq.nsmallest(5, a)
[3, 3, 5, 6, 8]
>>> heapq.nlargest(5, a)
[100, 100, 99, 98, 98]
1.33   Cartesian products (itertools.product)
>>> for p in itertools.product([1, 2, 3], [4, 5]):
(1, 4)
(1, 5)
(2, 4)
(2, 5)
(3, 4)
(3, 5)
>>> for p in itertools.product([0, 1], repeat=4):
...     print ''.join(str(x) for x in p)
...
0000
0001
0010
0011
0100
0101
0110
0111
1000
1001
1010
1011
1100
1101
1110
1111

## 20. Combinations and combinations with replacement (itertools.combinations and itertools.combinations_with_replacement)
>>> for c in itertools.combinations([1, 2, 3, 4, 5], 3):
...     print ''.join(str(x) for x in c)
...
123
124
125
134
135
145
234
235
245
345
>>> for c in itertools.combinations_with_replacement([1, 2, 3], 2):
...     print ''.join(str(x) for x in c)
...
11
12
13
22
23
33

## 21.Permutations (itertools.permutations)
>>> for p in itertools.permutations([1, 2, 3, 4]):
...     print ''.join(str(x) for x in p)
...
1234
1243
1324
1342
1423
1432
2134
2143
2314
2341
2413
2431
3124
3142
3214
3241
3412
3421
4123
4132
4213
4231
4312
4321

## 22.Chaining iterables (itertools.chain)
>>> a = [1, 2, 3, 4]
>>> for p in itertools.chain(itertools.combinations(a, 2), itertools.combinations(a, 3)):
...     print p
...
(1, 2)
(1, 3)
(1, 4)
(2, 3)
(2, 4)
(3, 4)
(1, 2, 3)
(1, 2, 4)
(1, 3, 4)
(2, 3, 4)
>>> for subset in itertools.chain.from_iterable(itertools.combinations(a, n) for n in range(len(a) + 1))
...     print subset
...
()
(1,)
(2,)
(3,)
(4,)
(1, 2)
(1, 3)
(1, 4)
(2, 3)
(2, 4)
(3, 4)
(1, 2, 3)
(1, 2, 4)
(1, 3, 4)
(2, 3, 4)
(1, 2, 3, 4)

## 23. Grouping rows by a given key (itertools.groupby)
>>> from operator import itemgetter
>>> import itertools
>>> with open('contactlenses.csv', 'r') as infile:
...     data = [line.strip().split(',') for line in infile]
...
>>> data = data[1:]
>>> def print_data(rows):
...     print '\n'.join('\t'.join('{: <16}'.format(s) for s in row) for row in rows)
...

>>> print_data(data)
young               myope                   no                      reduced                 none
young               myope                   no                      normal                  soft
young               myope                   yes                     reduced                 none
young               myope                   yes                     normal                  hard
young               hypermetrope            no                      reduced                 none
young               hypermetrope            no                      normal                  soft
young               hypermetrope            yes                     reduced                 none
young               hypermetrope            yes                     normal                  hard
pre-presbyopic      myope                   no                      reduced                 none
pre-presbyopic      myope                   no                      normal                  soft
pre-presbyopic      myope                   yes                     reduced                 none
pre-presbyopic      myope                   yes                     normal                  hard
pre-presbyopic      hypermetrope            no                      reduced                 none
pre-presbyopic      hypermetrope            no                      normal                  soft
pre-presbyopic      hypermetrope            yes                     reduced                 none
pre-presbyopic      hypermetrope            yes                     normal                  none
presbyopic          myope                   no                      reduced                 none
presbyopic          myope                   no                      normal                  none
presbyopic          myope                   yes                     reduced                 none
presbyopic          myope                   yes                     normal                  hard
presbyopic          hypermetrope            no                      reduced                 none
presbyopic          hypermetrope            no                      normal                  soft
presbyopic          hypermetrope            yes                     reduced                 none
presbyopic          hypermetrope            yes                     normal                  none

>>> data.sort(key=itemgetter(-1))
>>> for value, group in itertools.groupby(data, lambda r: r[-1]):
...     print '-----------'
...     print 'Group: ' + value
...     print_data(group)
...
-----------
Group: hard
young               myope                   yes                     normal                  hard
young               hypermetrope            yes                     normal                  hard
pre-presbyopic      myope                   yes                     normal                  hard
presbyopic          myope                   yes                     normal                  hard
-----------
Group: none
young               myope                   no                      reduced                 none
young               myope                   yes                     reduced                 none
young               hypermetrope            no                      reduced                 none
young               hypermetrope            yes                     reduced                 none
pre-presbyopic      myope                   no                      reduced                 none
pre-presbyopic      myope                   yes                     reduced                 none
pre-presbyopic      hypermetrope            no                      reduced                 none
pre-presbyopic      hypermetrope            yes                     reduced                 none
pre-presbyopic      hypermetrope            yes                     normal                  none
presbyopic          myope                   no                      reduced                 none
presbyopic          myope                   no                      normal                  none
presbyopic          myope                   yes                     reduced                 none
presbyopic          hypermetrope            no                      reduced                 none
presbyopic          hypermetrope            yes                     reduced                 none
presbyopic          hypermetrope            yes                     normal                  none
-----------
Group: soft
young               myope                   no                      normal                  soft
young               hypermetrope            no                      normal                  soft
pre-presbyopic      myope                   no                      normal                  soft
pre-presbyopic      hypermetrope            no                      normal                  soft
presbyopic          hypermetrope            no                      normal                  soft

## 24.Start a static HTTP server in any directory
[10:26] $ python -m SimpleHTTPServer 5000  
Serving HTTP on 0.0.0.0 port 5000 ...

## 25.Learn the Zen of Python
>>> import this  

The Zen of Python, by Tim Peters

Beautiful is better than ugly.   
Explicit is better than implicit.  
Simple is better than complex.  
Complex is better than complicated.  
Flat is better than nested.  
Sparse is better than dense.  
Readability counts.  
Special cases aren't special enough to break the rules.  
Although practicality beats purity.  
Errors should never pass silently.  
Unless explicitly silenced.  
In the face of ambiguity, refuse the temptation to guess.  
There should be one-- and preferably only one --obvious way to do it.  
Although that way may not be obvious at first unless you're Dutch.  
Now is better than never.  
Although never is often better than *right* now.  
If the implementation is hard to explain, it's a bad idea.  
If the implementation is easy to explain, it may be a good idea.  
Namespaces are one honking great idea -- let's do more of those!  

## 26.Use C-Style Braces Instead of Indentation to Denote Scopes
>>> from __future__ import braces

## 中间结果尽量使用imap/ifilter代替map/filter

```
##不推荐
reduce(rf, filter(ff, map(mf, a_list)))

##推荐
from itertools import ifilter, imap
reduce(rf, ifilter(ff, imap(mf, a_list)))
*lazy evaluation 会带来更高的内存使用效率，特别是当处理大数据操作的时候。
```

## 使用any/all函数

```
##不推荐
found = False
for item in a_list:
    if condition(item):
        found = True
        break
if found:
    # do something if found...  

##推荐
if any(condition(item) for item in a_list):
    # do something if found...
```

## 属性(property)

```
##不推荐
class Clock(object):
    def __init__(self):
        self.__hour = 1
    def setHour(self, hour):
        if 25 &amp;gt; hour &amp;gt; 0: self.__hour = hour
        else: raise BadHourException
    def getHour(self):
        return self.__hour

##推荐
class Clock(object):
    def __init__(self):
        self.__hour = 1
    def __setHour(self, hour):
        if 25 &amp;gt; hour &amp;gt; 0: self.__hour = hour
        else: raise BadHourException
    def __getHour(self):
        return self.__hour
    hour = property(__getHour, __setHour)
```

## 使用 with 忽视异常(仅限Python 3)

```
##不推荐
try:
    os.remove("somefile.txt")
except OSError:
    pass

##推荐
from contextlib import ignored  # Python 3 only

with ignored(OSError):
    os.remove("somefile.txt")
```

## 使用 with 处理加锁

```
##不推荐
import threading
lock = threading.Lock()

lock.acquire()
try:
    # 互斥操作...
finally:
    lock.release()

##推荐
import threading
lock = threading.Lock()

with lock:
    # 互斥操作...
```

