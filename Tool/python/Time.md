在 Python 中，与时间处理有关的模块包括：time，datetime 以及 calendar

必要说明：

- 虽然这个模块总是可用，但并非所有的功能都适用于各个平台。
- 该模块中定义的大部分函数是调用 C 平台上的同名函数实现，所以各个平台上实现可能略有不同。

一些术语和约定的解释：

- 时间戳（timestamp）的方式：通常来说，时间戳表示的是从 1970 年 1 月 1 日 00:00:00 开始按秒计算的偏移量（time.gmtime(0)）此模块中的函数无法处理 1970 纪元年以前的日期和时间或太遥远的未来（处理极限取决于 C 函数库，对于 32 位系统来说，是 2038 年）
- UTC（Coordinated Universal Time，世界协调时）也叫格林威治天文时间，是世界标准时间。在中国为 UTC+8
- DST（Daylight Saving Time）即夏令时的意思
- 一些实时函数的计算精度可能低于它们建议的值或参数，例如在大部分 Unix 系统，时钟一秒钟“滴答”50~100 次

时间元祖（time.struct_time）：

gmtime()，localtime() 和 strptime() 以时间元祖（struct_time）的形式返回。

| **索引值（Index）** | **属性（Attribute）** | **值（Values）**      |
| -------------- | ----------------- | ------------------ |
| 0              | tm_year（年）        | （例如：2015）          |
| 1              | tm_mon（月）         | 1 ~ 12             |
| 2              | tm_mday（日）        | 1 ~ 31             |
| 3              | tm_hour（时）        | 0 ~ 23             |
| 4              | tm_min（分）         | 0 ~ 59             |
| 5              | tm_sec（秒）         | 0 ~ 61（见下方注1）      |
| 6              | tm_wday（星期几）      | 0 ~ 6（0 表示星期一）     |
| 7              | tm_yday（一年中的第几天）  | 1 ~ 366            |
| 8              | tm_isdst（是否为夏令时）  | 0， 1， -1（-1 代表夏令时） |

注1：范围真的是 0 ~ 61（你没有看错哦`^_^`）；60 代表闰秒，61 是基于历史原因保留。

time.altzone

返回格林威治西部的夏令时地区的偏移秒数；如果该地区在格林威治东部会返回负值（如西欧，包括英国）；对夏令时启用地区才能使用。

time.asctime([t])

接受时间元组并返回一个可读的形式为"Tue Dec 11 18:07:14 2015"（2015年12月11日 周二 18时07分14秒）的 24 个字符的字符串。

time.clock()

用以浮点数计算的秒数返回当前的 CPU 时间。用来衡量不同程序的耗时，比 time.time() 更有用。

Python 3.3 以后不被推荐，由于该方法依赖操作系统，建议使用 perf_counter() 或 process_time() 代替（一个返回系统运行时间，一个返回进程运行时间，请按照实际需求选择）

time.ctime([secs]) 

作用相当于 asctime(localtime(secs))，未给参数相当于 asctime()

time.gmtime([secs])

接收时间辍（1970 纪元年后经过的浮点秒数）并返回格林威治天文时间下的时间元组 t（注：t.tm_isdst 始终为 0）

time.daylight

如果夏令时被定义，则该值为非零。

time.localtime([secs])

接收时间辍（1970 纪元年后经过的浮点秒数）并返回当地时间下的时间元组 t（t.tm_isdst 可取 0 或 1，取决于当地当时是不是夏令时）

time.mktime(t)

接受时间元组并返回时间辍（1970纪元后经过的浮点秒数）

time.perf_counter()

返回计时器的精准时间（系统的运行时间），包含整个系统的睡眠时间。由于返回值的基准点是未定义的，所以，只有连续调用的结果之间的差才是有效的。

time.process_time() 

返回当前进程执行 CPU 的时间总和，不包含睡眠时间。由于返回值的基准点是未定义的，所以，只有连续调用的结果之间的差才是有效的。

time.sleep(secs)

推迟调用线程的运行，secs 的单位是秒。

time.strftime(format[, t]) 

把一个代表时间的元组或者 struct_time（如由 time.localtime() 和 time.gmtime() 返回）转化为格式化的时间字符串。如果 t 未指定，将传入 time.localtime()。如果元组中任何一个元素越界，将会抛出 ValueError 异常。

format 格式如下：

| **格式** | **含义**                                   | **备注** |
| ------ | ---------------------------------------- | ------ |
| %a     | 本地（locale）简化星期名称                         |        |
| %A     | 本地完整星期名称                                 |        |
| %b     | 本地简化月份名称                                 |        |
| %B     | 本地完整月份名称                                 |        |
| %c     | 本地相应的日期和时间表示                             |        |
| %d     | 一个月中的第几天（01 - 31）                        |        |
| %H     | 一天中的第几个小时（24 小时制，00 - 23）                |        |
| %l     | 一天中的第几个小时（12 小时制，01 - 12）                |        |
| %j     | 一年中的第几天（001 - 366）                       |        |
| %m     | 月份（01 - 12）                              |        |
| %M     | 分钟数（00 - 59）                             |        |
| %p     | 本地 am 或者 pm 的相应符                         | *注1*   |
| %S     | 秒（01 - 61）                               | *注2*   |
| %U     | 一年中的星期数（00 - 53 星期天是一个星期的开始）第一个星期天之前的所有天数都放在第 0 周 | *注3*   |
| %w     | 一个星期中的第几天（0 - 6，0 是星期天）                  | *注3*   |
| %W     | 和 %U 基本相同，不同的是 %W 以星期一为一个星期的开始           |        |
| %x     | 本地相应日期                                   |        |
| %X     | 本地相应时间                                   |        |
| %y     | 去掉世纪的年份（00 - 99）                         |        |
| %Y     | 完整的年份                                    |        |
| %z     | 用 +HHMM 或 -HHMM 表示距离格林威治的时区偏移（H 代表十进制的小时数，M 代表十进制的分钟数） |        |
| %Z     | 时区的名字（如果不存在为空字符）                         |        |
| `%%`     | %号本身                                     |        |

注1：“%p”只有与“%I”配合使用才有效果。

注2：范围真的是 0 ~ 61（你没有看错哦^_^）；60 代表闰秒，61 是基于历史原因保留。

注3：当使用 strptime() 函数时，只有当在这年中的周数和天数被确定的时候 %U 和 %W 才会被计算。

举个例子：

```python
# I love FishC.com!
>>> import time as t
>>> t.strftime("a, %d %b %Y %H:%M:%S +0000", t.gmtime())
'a, 24 Aug 2014 14:15:03 +0000'
```

**time.strptime(string[, format])**

把一个格式化时间字符串转化为 struct_time。实际上它和 strftime() 是逆操作。

举个例子：

```python
# I really love FishC.com!
>>> import time as t
>>> t.strptime("30 Nov 14", "%d %b %y")
time.struct_time(tm_year=2014, tm_mon=11, tm_mday=30, tm_hour=0, tm_min=0, tm_sec=0, tm_wday=6, tm_yday=334, tm_isdst=-1)
```

**time.time()**

返回当前时间的时间戳（1970 纪元年后经过的浮点秒数）


**time.timezone**

time.timezone 属性是当地时区（未启动夏令时）距离格林威治的偏移秒数（美洲 >0；大部分欧洲，亚洲，非洲 <= 0）


**time.tzname**

time.tzname 属性是包含两个字符串的元组：第一是当地非夏令时区的名称，第二个是当地的 DST 时区的名称。

```python
import time as t

class MyTimer():
    def __init__(self):
        self.unit = ['年', '月', '天', '小时', '分钟', '秒']
        self.prompt = '未开始计时！'
        self.lasted = []
        self.begin = 0 # 不能用start,方法名和属性名重名的时候属性会覆盖方法
        self.end = 0
    
    # __str__使得调用print(MyTimer())时输出return的值
    def __str__(self):
        return self.prompt

    # __repr__使得调用MyTimer()时输出return的值
    __repr__ = __str__

    def __add__(self, other):
        prompt = '总共运行了'
        result = []
        for index in range(6):
            result.append(self.lasted[index] + other.lasted[index])
            if result[index]:
                prompt += (str(result[index]) + self.unit[index])
        return prompt
    
    # 开始计时
    def start(self):
        self.begin = t.localtime()
        self.prompt = '提示：请先调用stop()停止计时！'
        print('计时开始...')

    # 停止计时
    def stop(self):
        if not self.begin:
            print('提示：请先调用start()进行计时！')
        else:
            self.end = t.localtime()
            self._calc()
            print('计时结束！')

    # 内部方法，计算运行时间
    def _calc(self):
        self.lasted = []
        self.prompt = '总共运行了'
        # 分别指年月日时分秒
        for index in range(6):
            self.lasted.append(self.end[index] - self.begin[index])
            # 为0的时间不放入prompt
            if self.lasted[index]:
                self.prompt += (str(self.lasted[index]) + self.unit[index])
        # 为下一轮计时初始化变量
        self.begin = 0
        self.end = 0
```

## datetime模块详解

datetime 模块详解 -- 基本的日期和时间类型

datetime 模块提供了各种类用于操作日期和时间，该模块侧重于高效率的格式化输出

在 Python 中，与时间处理有关的模块包括：

time

，datetime 以及 calendar

datetime 模块定义了两个常量：

- datetime.MINYEAR - date 和 datetime 对象所能支持的最小年份，object.MINYEAR 的值为 1
- datetime.MAXYEAR - date 和 datetime 对象所能支持的最大年份，object.MAXYEAR 的值为 9999

datetime 模块中定义的类（前四个下方有详解）：

- datetime.date - 表示日期的类，常用属性：year, month, day
- datetime.time - 表示时间的类，常用属性：hour, minute, second, microsecond, tzinfo
- datetime.datetime - 表示日期和时间的类，常用属性： year, month, day, hour, minute, second, microsecond, tzinfo
- datetime.timedelta - 表示时间间隔，即两个时间点（date，time，datetime）之间的长度
- datetime.tzinfo - 表示时区的基类，为上方的 time 和 datetime 类提供调整的基准
- datetime.timezone - 表示 UTC 时区的固定偏移，是 tzinfo 基类的实现

注：上边这些类的对象是不可变的

上边这些类的从属关系：

```
object
    timedelta
    tzinfo
        timezone
    time
    date
        datetime
```

**timedelta 对象**

timedelta 对象表示两个日期或时间之间的间隔

```
datetime.timedelta(days=0, seconds=0, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0) 
```

以上所有的参数都是可选的（默认为 0），参数的可以是整数或浮点数，正数或负数。

内部的存储单位只有 days（天）、seconds（秒）、microseconds（毫秒），其他单位均先转换后再存储：

- 1 millisecond -> 1000 microseconds
- 1 minutes -> 60 seconds
- 1 hours -> 3600 seconds
- 1 weeks -> 7 days

而 days、seconds 和 microseconds 为了不产生时间表示上的歧义，将根据以下范围自动“进位”：

- 0 <= microseconds < 1000000
- 0 <= seconds < 3600 * 24（1小时的秒数 * 24小时）
- -999999999 <= days <= 999999999

timedelta 类属性：

- timedelta.min - timedelta 对象负值的极限，timedelta(-999999999)
- timedelta.max - timedelta 对象正值的极限，timedelta(days=999999999, hours=23, minutes=59, seconds=59, microseconds=999999)
- timedelta.resolution - 两个 timedelta 不相等的对象之间最小的差值，timedelta(microseconds=1)

请注意，在正常情况下，timedelta.max > -timedelta.min，-timedelta.max 无意义。

timedelta 实例属性（只读）：

| **属性**                 | **取值范围**               |
| ---------------------- | ---------------------- |
| timedelta.days         | -999999999 ~ 999999999 |
| timedelta.seconds      | 0 ~ 86399              |
| timedelta.microseconds | 0 ~ 999999             |

timedelta 对象支持的操作：

| **操作**                       | **结果**                                   |
| ---------------------------- | ---------------------------------------- |
| t1 = t2 + t3                 | t2 和 t3 的和，随后：t1 - t2 == t3 and t1 - t3 == t2 为 True（注1） |
| t1 = t2 - t3                 | t2 和 t3 的差，随后：t1 == t2 - t3 and t2 == t1 + t3 为 True(注1） |
| t1 = t2 \* i 或 t1 = i \* t2    | 对象乘以一个整数，随后：t1 // i == t2 为 true；且 i != 0 |
| t1 = t2 \* f 或 t1 = f \* t2    | 对象乘以一个浮点数，结果四舍五入到精度 timedelta.resolution（注1） |
| f = t2 / t3                  | t2 和 t3 的商（注3），返回一个 float 对象           |
| t1 = t2 / f 或 t1 = t2 / i    | 对象除以一个整数或浮点数，结果四舍五入到精度 timedelta.resolution |
| t1 = t2 // i 或 t1 = t2 // t3 | 对象地板除一个整数或浮点数，结果舍去小数，返回一个整数*（注3）        |
| t1 = t2 % t3                 | t2 和 t3 的余数，返回一个 timedelta 对象（注3）*      |
| q, r = divmod(t1, t2)        | 计算 t1 和 t2 的商和余数，q = t1 // t2（注3），r = t1 % t2，q 是一个整数，r 是一个 timedelta 对象 |
| +t1                          | 返回一个 timedelta 对象，且值相同（注2）             |
| -t1                          | 等同于 timedelta(-t1.days, -t1.seconds, -t1.microseconds)，并且相当于 t1  -1（注1、4） |
| abs(t)                       | 当 t.days >= 0 时，等同于 +t；当 t.days < = 时，等同于 -t（注2） |
| str(t)                       | 返回一个字符串，按照此格式：[D day[ s ], ][H]H:MM:SS[.UUUUUU] |
| repr(t)                      | 返回一个字符串，按照此格式：datetime.timedelta(D[, S[, U]]) |

注1：这是准确的，但可能会溢出

注2：这是准确的，并且不会溢出

注3：除数为 0 会引发 ZeroDivisionError 异常

注4：-timedelta.max 是无意义的

timedelta 实例方法：

timedelta.total_seconds()

\- 返回 timedelta 对象所包含的总秒数，相当于 td / timedelta(seconds=1)

请注意，对于非常大的时间间隔（在大多数平台上是大于270年），这种方法将失去微秒（microsecond）精度

timedelta 用法示例：

```
# 爱学习，爱鱼C工作室
>>> from datetime import timedelta
>>> year = timedelta(days=365)
>>> another_year = timedelta(weeks=40, days=84, hours=23,
...                          minutes=50, seconds=600)  # adds up to 365 days
>>> year.total_seconds()
31536000.0
>>> year == another_year
True
>>> ten_years = 10 * year
>>> ten_years, ten_years.days // 365
(datetime.timedelta(3650), 10)
>>> nine_years = ten_years - year
>>> nine_years, nine_years.days // 365
(datetime.timedelta(3285), 9)
>>> three_years = nine_years // 3;
>>> three_years, three_years.days // 365
(datetime.timedelta(1095), 3)
>>> abs(three_years - ten_years) == 2 * three_years + year
True
```

**date 对象**

date 对象表示一个日期，在一个理想化的日历里，日期由 year（年）、month（月）、day（日）组成

```
datetime.date(year, month, day)
```

所有的参数都是必需的，参数可以是整数，并且在以下范围内：

- MINYEAR <= year <= MAXYEAR（也就是 1 ~ 9999）
- 1 <= month <= 12
- 1 <= day <= 根据 year 和 month 来决定（例如 2015年2月 只有 28 天）

date 类方法（classmethod）：

- date.today() - 返回一个表示当前本地日期的 date 对象
- date.fromtimestamp(timestamp) - 根据给定的时间戮，返回一个 date 对象
- date.fromordinal(ordinal) - 将 Gregorian 日历时间转换为 date 对象（Gregorian Calendar：一种日历表示方法，类似于我国的农历，西方国家使用比较多）

date 类属性：

- date.min - date 对象所能表示的最早日期，date(MINYEAR, 1, 1)
- date.max - date 对象所能表示的最晚日期，date(MAXYEAR, 12, 31)
- date.resolution - date 对象表示日期的最小单位，在这里是 1 天，timedelta(days=1)

date 实例属性（只读）：

| **属性**     | **取值范围**                                 |
| ---------- | ---------------------------------------- |
| date.year  | MINYEAR ~ MAXYEAR（1 ~ 9999）              |
| date.month | 1 ~ 12                                   |
| date.day   | 1 ~ 根据 year 和 month 来决定（例如 2015年2月 只有 28 天） |

date 对象支持的操作：

| **操作**                    | **结果**                                   |
| ------------------------- | ---------------------------------------- |
| date2 = date1 + timedelta | 日期加上一个时间间隔，返回一个新的日期对象*（注1）*              |
| date2 = date1 - timedelta | 日期减去一个时间间隔，相当于 date2 + timedelta == date1*（注2）* |
| timedelta = date1 - date2 | *（注3）*                                   |
| date1 < date2             | 当 date1 的日期在 date2 之前时，我们认为 date1 < date2*（注4）* |

注1：timedelta.day > 0 或 timedelta.day < 0 决定 date2 日期增长的方向；随后，date2 - date1 == timedelta.days；timedelta.seconds 和 timedelta.microseconds 被忽略；如果 date2.year < MINYEAR 或 date2.year > MAXYEAR，引发 OverflowError 异常
注2：这并不等同于 date1 + (-timedelta)，因为单独的 -timedelta 可能会溢出，而 date1 - timedelta 则不会溢出；timedelta.seconds 和 timedelta.microseconds 被忽略
注3：这是准确的，并且不会溢出；timedelta.seconds 和 timedelta.microseconds 都为 0，然后 date2 + timedelta == date1
注4：换句话说，当且仅当 date1.toordinal() < date2.toordinal()，才有 date1 < date2

date 实例方法：

date.replace(year, month, day)

\- 生成一个新的日期对象，用参数指定的年、月、日代替原有对象中的属性

date.timetuple()

\- 返回日期对应的 time.struct_time 对象（类似于 

time 模块

的 time.localtime()）

date.toordinal()

\- 返回日期对应的 Gregorian Calendar 日期

date.weekday()

\- 返回 0 ~ 6 表示星期几（星期一是 0，依此类推）

date.isoweekday()

\- 返回 1 ~ 7 表示星期几（星期一是1， 依此类推）

date.isocalendar()

\- 返回一个三元组格式 (year, month, day)

date.isoformat()

\- 返回一个 ISO 8601 格式的日期字符串，如 "YYYY-MM-DD" 的字符串

date.\__str\__()

\- 对于 date 对象 d 来说，str(d) 相当于 d.isoformat()

date.ctime()

\- 返回一个表示日期的字符串，相当于 

time 模块

的 time.ctime(time.mktime(d.timetuple()))

date.strftime(format)

\- 返回自定义格式化字符串表示日期，下面有详解

date.\__format\__(format)

\- 跟 date.strftime(format) 一样，这使得调用 str.format() 时可以指定 data 对象的字符串

以下是计算天数的例子：

```
# You may say I'm the dreamer. But I'm not the only one! 

>>> import time
>>> from datetime import date
>>> today = date.today()
>>> today
datetime.date(2014, 8, 31)
>>> today == date.fromtimestamp(time.time())
True
>>> my_birthday = date(today.year, 6, 24)
>>> if my_birthday < today:
        my_birthday = my_birthday.replace(year = today.year + 1)

>>> my_birthday
datetime.date(2015, 6, 24)
>>> time_to_birthday = abs(my_birthday - today)
>>> time_to_birthday.days
297
```

**关于 date 的综合应用：**

```
# Follow FishC. Follow your dream!

>>> from datetime import date
>>> d = date.fromordinal(735678)  # 自日期 1.1.0001 之后的第 735678 天
>>> d
datetime.date(2015, 3, 21)
>>> t = d.timetuple()
>>> for i in t:
        print(i)
        
2015
3
21
0
0
0
5
80
-1
>>> ic = d.isocalendar()
>>> for i in ic:
        print(i)
        
2015
12
6
>>> d.isoformat()
'2015-03-21'
>>> d.strftime("%d/%m/%y")
'21/03/15'
>>> d.strftime("%A %d. %B %Y")
'Saturday 21. March 2015'
>>> 'The {1} is {0:%d}, the {2} is {0:%B}.'.format(d, "day", "month")
'The day is 21, the month is March.'
```

**time 对象**

time 对象表示一天中的一个时间，并且可以通过 tzinfo 对象进行调整

```
datetime.time(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
```

所有的参数都是可选的；tzinfo 可以是 None 或者 tzinfo 子类的实例对象；其余的参数可以是整数，并且在以下范围内：

- 0 <= hour < 24
- 0 <= minute < 60
- 0 <= second < 60
- 0 <= microsecond < 1000000

注：如果参数超出范围，将引发 ValueError 异常

**time 类属性**

- time.min - time 对象所能表示的最早时间，time(0, 0, 0, 0)
- time.max - time 对象所能表示的最晚时间，time(23, 59, 59, 999999)
- time.resolution - time 对象表示时间的最小单位，在这里是 1 毫秒，timedelta(microseconds=1)

time 实例属性（只读）：

| **属性**           | **取值范围**            |
| ---------------- | ------------------- |
| time.hour        | 0 ~ 23              |
| time.minute      | 0 ~ 59              |
| time.second      | 0 ~ 59              |
| time.microsecond | 0 ~ 999999          |
| time.tzinfo      | 通过构造函数的 tzinfo 参数赋值 |

time 实例方法：

time.replace([hour[, minute[, second[, microsecond[, tzinfo]]]]]) 

\- 生成一个新的时间对象，用参数指定时间代替原有对象中的属性

time.isoformat()

\- 返回一个 ISO 8601 格式的日期字符串，如 "HH:MM:SS.mmmmmm" 的字符串

time.\__str\__()

\- 对于 time 对象 t 来说，str(t) 相当于 t.isoformat()

time.strftime(format)

\- 返回自定义格式化字符串表示时间，下面有详解

time.\__format\__(format)

\- 跟 time.strftime(format) 一样，这使得调用 str.format() 时可以指定 time 对象的字符串

time.utcoffset()

\- 如果 tzinfo 属性是 None，则返回 None；否则返回 self.tzinfo.utcoffset(self)

time.dst()

\- 如果 tzinfo 属性是 None，则返回 None；否则返回 self.tzinfo.dst(self)

time.tzname()

\- 如果 tzinfo 属性是 None，则返回 None；否则返回 self.tzinfo.tzname(self)

关于 time 的综合应用：

```
# 学编程，到鱼C
>>> from datetime import time, timedelta, tzinfo
>>> class GMT1(tzinfo):
        def utcoffset(self, dt):
                return timedelta(hours=1)
        def dst(self, dt):
                return timedelta(0)
        def tzname(self, dt):
                return "欧洲/布拉格"

>>> t = time(14, 10, 30, tzinfo=GMT1())
>>> t
datetime.time(14, 10, 30, tzinfo=<__main__.GMT1 object at 0x02D7FE90>)
>>> gmt = GMT1()
>>> t.isoformat()
'14:10:30+01:00'
>>> t.dst()
datetime.timedelta(0)
>>> t.tzname()
'欧洲/布拉格'
>>> t.strftime("%H:%M:%S %Z")
'14:10:30 欧洲/布拉格'
>>> 'The {} is {:%H:%M}.'.format("time", t)
'The time is 14:10.'
```

**datetime 对象**

datetime 对象是 date 对象和 time 对象的结合体，并且包含他们的所有信息

```
datetime.datetime(year, month, day, hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
```

必须的参数是 year（年）、month（月）、day（日）；tzinfo 可以是 None 或者 tzinfo 子类的实例对象；其余的参数可以是整数，并且在以下范围内：

- MINYEAR <= year <= MAXYEAR（也就是 1 ~ 9999）
- 1 <= month <= 12
- 1 <= day <= 根据 year 和 month 来决定（例如 2015年2月 只有 28 天）
- 0 <= hour < 24
- 0 <= minute < 60
- 0 <= second < 60
- 0 <= microsecond < 1000000

注：如果参数超出范围，将引发 ValueError 异常

datetime 类方法（classmethod）：

datetime.today()

\- 返回一个表示当前本地时间的 datetime 对象，等同于 datetime.fromtimestamp(time.time())

datetime.now(tz=None)

\- 返回一个表示当前本地时间的 datetime 对象；如果提供了参数 tz，则获取 tz 参数所指时区的本地时间

datetime.utcnow()

\- 返回一个当前 UTC 时间的 datetime 对象

datetime.fromtimestamp(timestamp, tz=None)

\- 根据时间戮创建一个 datetime 对象，参数 tz 指定时区信息

datetime.utcfromtimestamp(timestamp)

\- 根据时间戮创建一个 UTC 时间的 datetime 对象

datetime.fromordinal(ordinal)

\- 返回对应 Gregorian 日历时间对应的 datetime 对象

datetime.combine(date, time)

\- 根据参数 date 和 time，创建一个 datetime 对象

datetime.strptime(date_string, format)

\- 将格式化字符串转换为 datetime 对象

datetime 类属性：

- datetime.min - datetime 对象所能表示的最早日期，datetime(MINYEAR, 1, 1, tzinfo=None)
- datetime.max - datetime 对象所能表示的最晚日期，datetime(MAXYEAR, 12, 31, 23, 59, 59, 999999, tzinfo=None)
- datetime.resolution - datetime 对象表示日期的最小单位，在这里是 1 毫秒，timedelta(microseconds=1)

datetime 实例属性（只读）：

| **属性**               | **取值范围**                                 |
| -------------------- | ---------------------------------------- |
| datetime.year        | MINYEAR ~ MAXYEAR（1 ~ 9999）              |
| datetime.month       | 1 ~ 12                                   |
| datetime.day         | 1 ~ 根据 year 和 month 来决定（例如 2015年2月 只有 28 天） |
| datetime.hour        | 0 ~ 23                                   |
| datetime.minute      | 0 ~ 59                                   |
| datetime.second      | 0 ~ 59                                   |
| datetime.microsecond | 0 ~ 999999                               |
| datetime.tzinfo      | 通过构造函数的 tzinfo 参数赋值                      |

datetime 对象支持的操作：

| **操作**                            | **结果**                                   |
| --------------------------------- | ---------------------------------------- |
| datetime2 = datetime1 + timedelta | 日期加上一个时间间隔，返回一个新的日期对象*（注1）*              |
| datetime2 = datetime1 - timedelta | 日期减去一个时间间隔，相当于 datetime2 + timedelta == datetime1*（注2）* |
| timedelta = datetime1 - datetime2 | 两个日期相减得到一个时间间隔*（注3）*                     |
| datetime1 < datetime2             | 当 datetime1 的日期在 datetime2 之前时，我们认为 datetime1 < datetime2 |

注1：timedelta.day > 0 或 timedelta.day < 0 决定 datetime2 日期增长的方向；计算结果 datetime2 的 tzinfo 属性和 datetime1 相同；如果 date2.year < MINYEAR 或 date2.year > MAXYEAR，引发 OverflowError 异常
注2：计算结果 datetime2 的 tzinfo 属性和 datetime1 相同；这并不等同于 date1 + (-timedelta)，因为单独的 -timedelta 可能会溢出，而 date1 - timedelta 则不会溢出
注3：如果 datetime1 和 datetime2 的 tzinfo 属性一样（指向同一个时区），则 tzinfo 属性被忽略，计算结果为一个 timedelta 对象 t，则 datetime2 + t == datetime1（不用进行时区调整）；如果 datetime1 和 datetime2 的 tzinfo 属性不一样（指向不同时区），则 datetime1 和 datetime2 会先被转换为 UTC 时区时间，在进行计算，(datetime1.replace(tzinfo=None) - datetime1.utcoffset()) - (datetime2.replace(tzinfo=None) - datetime2.utcoffset())

datetime 实例方法：

datetime.date()

\- 返回一个 date 对象datetime.time() - 返回一个 time 对象（tzinfo 属性为 None）

datetime.timetz()

\- 返回一个 time() 对象（带有 tzinfo 属性）

datetime.replace([year[, month[, day[, hour[, minute[, second[, microsecond[, tzinfo]]]]]]]])

\- 生成一个新的日期对象，用参数指定日期和时间代替原有对象中的属性

datetime.astimezone(tz=None)

\- 传入一个新的 tzinfo 属性，返回根据新时区调整好的 datetime 对象

datetime.utcoffset()

\- 如果 tzinfo 属性是 None，则返回 None；否则返回 self.tzinfo.utcoffset(self)

datetime.dst()

\- 如果 tzinfo 属性是 None，则返回 None；否则返回 self.tzinfo.dst(self)

datetime.tzname()

\- 如果 tzinfo 属性是 None，则返回 None；否则返回 self.tzinfo.tzname(self)

datetime.timetuple()

-  返回日期对应的 time.struct_time 对象（类似于 

time 模块

的 time.localtime()）

datetime.utctimetuple()

\- 返回 UTC 日期对应的 time.struct_time 对象

datetime.toordinal()

\- 返回日期对应的 Gregorian Calendar 日期（类似于 self.date().toordinal()）

datetime.timestamp()

\- 返回当前时间的时间戳（类似于 

time 模块

的 time.time()）

datetime.weekday()

\- 返回 0 ~ 6 表示星期几（星期一是 0，依此类推）

datetime.isoweekday()

 

\- 返回 1 ~ 7 表示星期几（星期一是1， 依此类推）

datetime.isocalendar() 

\- 返回一个三元组格式 (year, month, day)

datetime.isoformat(sep='T')

\- 返回一个 ISO 8601 格式的日期字符串，如 "YYYY-MM-DD" 的字符串

datetime.__str__()

\- 对于 date 对象 d 来说，str(d) 相当于 d.isoformat()

datetime.ctime()

\- 返回一个表示日期的字符串，相当于 

time 模块

的 time.ctime(time.mktime(d.timetuple()))

datetime.strftime(format)

\- 返回自定义格式化字符串表示日期，下面有详解

datetime.__format__(format)

\- 跟 datetime.strftime(format) 一样，这使得调用 str.format() 时可以指定 data 对象的字符串

关于 datetime 的综合应用：

```python
# I love FishC.com!
>>> from datetime import datetime, date, time

# 使用 datetime.combine()
>>> d = date(2015, 8, 1)
>>> t = time(12, 30)
>>> datetime.combine(d, t)
datetime.datetime(2015, 8, 1, 12, 30)

# 使用 datetime.now() 或 datetime.utcnow()
>>> datetime.now()
datetime.datetime(2014, 8, 31, 18, 13, 40, 858954)
>>> datetime.utcnow()
datetime.datetime(2014, 8, 31, 10, 13, 49, 347984)

# 使用 datetime.srptime()
>>> dt = datetime.strptime("21/11/14 16:30", "%d/%m/%y %H:%M")
>>> dt
datetime.datetime(2014, 11, 21, 16, 30)

# 使用 datetime.timetuple()
>>> tt = dt.timetuple()
>>> for it in tt:
        print(it)

2014
11
21
16
30
0
4
325
-1

# ISO 格式的日期
>>> ic = dt.isocalendar()
>>> for it in ic:
        print(it)

2014
47
5

# 格式化 datetime 对象
>>> dt.strftime("%A, %d. %B %Y %I:%M%p")
'Friday, 21. November 2014 04:30PM'
>>> 'The {1} is {0:%d}, the {2} is {0:%B}, the {3} is {0:%I:%M%p}.'.format(dt, "day", "month", "time")
'The day is 21, the month is November, the time is 04:30PM.'
```

**带有 tzinfo 的 datetime 综合演示：**

```python
# 嘿，都能看到这里来了，毅力不错哈^_^
>>> from datetime import timedelta, datetime, tzinfo
>>> class GMT1(tzinfo):
        def utcoffset(self, dt):
                return timedelta(hours=1) + self.dst(dt)
        def dst(self, dt):
                # DST 开始于三月最后一个星期天
                # 结束于十月最后一个星期天
                d = datetime(dt.year, 4, 1)
                self.dston = d - timedelta(days=d.weekday() + 1)
                d = datetime(dt.year, 11, 1)
                self.dstoff = d - timedelta(days=d.weekday() + 1)
                if self.dston <= dt.replace(tzinfo=None) < self.dstoff:
                        return timedelta(hours=1)
                else:
                        return timedelta(0)
        def tzname(self, dt):
                return "GMT +1"
        
>>> class GMT2(tzinfo):
        def utcoffset(self, dt):
                return timedelta(hours=2) + self.dst(dt)
        def dst(self, dt):
                d = datetime(dt.year, 4, 1)
                self.dston = d - timedelta(days=d.weekday() + 1)
                d = datetime(dt.year, 11, 1)
                self.dstoff = d - timedelta(days=d.weekday() + 1)
                if self.dston <=  dt.replace(tzinfo=None) < self.dstoff:
                        return timedelta(hours=1)
                else:
                        return timedelta(0)
        def tzname(self, dt):
                return "GMT +2"
        
>>> gmt1 = GMT1()

# 夏令时
>>> dt1 = datetime(2014, 11, 21, 16, 30, tzinfo=gmt1)
>>> dt1.dst()
datetime.timedelta(0)
>>> dt1.utcoffset()
datetime.timedelta(0, 3600)
>>> dt2 = datetime(2014, 6, 14, 13, 0, tzinfo=gmt1)
>>> dt2.dst()
datetime.timedelta(0, 3600)
>>> dt2.utcoffset()
datetime.timedelta(0, 7200)

# 将 datetime 转换到另一个时区
>>> dt3 = dt2.astimezone(GMT2())
>>> dt3
datetime.datetime(2014, 6, 14, 14, 0, tzinfo=<__main__.GMT2 object at 0x036C0F70>)
>>> dt2
datetime.datetime(2014, 6, 14, 13, 0, tzinfo=<__main__.GMT1 object at 0x036C0B10>)
>>> dt2.utctimetuple() == dt3.utctimetuple()
True
```

格式化字符串：strftime() 和 strptime()

date, datetime, 和 time 对象均支持使用 strftime(format) 方法，将指定的日期或时间转换为自定义的格式化字符串

相反的，datetime.strptime() 类方法却是把格式化字符串转换为 datetime 对象

| **格式化指令** | **含义**                                   |
| --------- | ---------------------------------------- |
| %a        | 星期的简写（星期一 ~ 天：Mon, Tue, Wed, Thu, Fri, Sat, Sun） |
| %A        | 星期的全写（星期一 ~ 天：Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday） |
| %w        | 在一个星期中的第几天（ 0 表示星期天 ... 6 表示星期六）         |
| %d        | 在一个月中的第几天（01, 02, ..., 31）               |
| %b        | 月份的简写（一月 ~ 十二月：Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec） |
| %B        | 月份的全写（一月 ~ 十二月：January, February, March, April, May, June, July, August, September, October, November, December） |
| %m        | 月份（01, 02, ..., 12）                      |
| %y        | 用两个数字表示年份（例如 2014年 == 14）                |
| %Y        | 用四个数字表示年份                                |
| %H        | 二十四小时制（00, 01, ..., 23）                  |
| %I        | 十二小时制（01, 02, ..., 11）                   |
| %p        | AM 或者 PM                                 |
| %M        | 分钟（00, 01, ..., 59）                      |
| %S        | 秒（00, 01, ..., 59）                       |
| %f        | 微秒（000000, 000001, ..., 999999）          |
| %z        | 与 UTC 时间的间隔 ；如果是本地时间，返回空字符串（(empty), +0000, -0400, +1030） |
| %Z        | 时区名称；如果是本地时间，返回空字符串（(empty), UTC, EST, CST） |
| %j        | 在一年中的第几天（001, 002, ..., 366）             |
| %U        | 在一年中的第几周，星期天作为第一天（00, 01, ..., 53）       |
| %W        | 在一年中的第几周，星期一作为第一天（00, 01, ..., 53）       |
| %c        | 用字符串表示日期和时间（Tue Aug 16 21:30:00 2014）    |
| %x        | 用字符串表示日期（08/16/14）                       |
| %X        | 用字符串表示时间（21:30:00）                       |
| `%%`        | 表示百分号                                    |

格式化字符串综合演示：

```
>>> from datetime import datetime
>>> dt = datetime.now()
>>> print('(%Y-%m-%d %H:%M:%S %f): ', dt.strftime('%Y-%m-%d %H:%M:%S %f'))
(%Y-%m-%d %H:%M:%S %f):  2014-08-31 23:54:58 379804
>>> print('(%Y-%m-%d %H:%M:%S %p): ', dt.strftime('%y-%m-%d %I:%M:%S %p'))
(%Y-%m-%d %H:%M:%S %p):  14-08-31 11:54:58 PM
>>> print('%%a: %s ' % dt.strftime('%a'))
%a: Sun 
>>> print('%%A: %s ' % dt.strftime('%A'))
%A: Sunday 
>>> print('%%b: %s ' % dt.strftime('%b'))
%b: Aug 
>>> print('%%B: %s ' % dt.strftime('%B'))
%B: August 
>>> print('日期时间%%c: %s ' % dt.strftime('%c'))
日期时间%c: 08/31/14 23:54:58 
>>> print('日期%%x：%s ' % dt.strftime('%x'))
日期%x：08/31/14 
>>> print('时间%%X：%s ' % dt.strftime('%X'))
时间%X：23:54:58 
>>> print('今天是这周的第%s天 ' % dt.strftime('%w'))
今天是这周的第0天 
>>> print('今天是今年的第%s天 ' % dt.strftime('%j'))
今天是今年的第243天 
>>> print('今周是今年的第%s周 ' % dt.strftime('%U'))
今周是今年的第35周
```

## timeit模块详解

**timeit 模块详解 -- 准确测量小段代码的执行时间**

timeit 模块提供了测量 Python 小段代码执行时间的方法。它既可以在命令行界面直接使用，也可以通过导入模块进行调用。该模块灵活地避开了测量执行时间所容易出现的错误。

以下例子是命令行界面的使用方法：

```
$ python -m timeit '"-".join(str(n) for n in range(100))'
10000 loops, best of 3: 40.3 usec per loop
$ python -m timeit '"-".join([str(n) for n in range(100)])'
10000 loops, best of 3: 33.4 usec per loop
$ python -m timeit '"-".join(map(str, range(100)))'
10000 loops, best of 3: 25.2 usec per loop
```

以下例子是 IDLE 下调用的方法：

```
>>> import timeit
>>> timeit.timeit('"-".join(str(n) for n in range(100))', number=10000)
0.8187260627746582
>>> timeit.timeit('"-".join([str(n) for n in range(100)])', number=10000)
0.7288308143615723
>>> timeit.timeit('"-".join(map(str, range(100)))', number=10000)
0.5858950614929199
```

需要注意的是，只有当使用命令行界面时，timeit 才会自动确定重复的次数。


**timeit 模块**

该模块定义了三个实用函数和一个公共类。


**timeit.timeit(stmt='pass', setup='pass', timer=<default timer>, number=1000000)**

创建一个 Timer 实例，参数分别是 stmt（需要测量的语句或函数），setup（初始化代码或构建环境的导入语句），timer（计时函数），number（每一次测量中语句被执行的次数）

*注：由于 timeit() 正在执行语句，语句中如果存在返回值的话会阻止 timeit() 返回执行时间。timeit() 会取代原语句中的返回值。*


**timeit.repeat(stmt='pass', setup='pass', timer=<default timer>, repeat=3, number=1000000)**

创建一个 Timer 实例，参数分别是 stmt（需要测量的语句或函数），setup（初始化代码或构建环境的导入语句），timer（计时函数），repeat（重复测量的次数），number（每一次测量中语句被执行的次数）


**timeit.default_timer()**

默认的计时器，一般是 time.perf_counter()，time.perf_counter() 方法能够在任一平台提供最高精度的计时器（它也只是记录了自然时间，记录自然时间会被很多其他因素影响，例如计算机的负载）。


**class timeit.Timer(stmt='pass', setup='pass', timer=<timer function>) **

计算小段代码执行速度的类，构造函数需要的参数有 stmt（需要测量的语句或函数），setup（初始化代码或构建环境的导入语句），timer（计时函数）。前两个参数的默认值都是 'pass'，timer 参数是平台相关的；前两个参数都可以包含多个语句，多个语句间使用分号（;）或新行分隔开。

第一次测试语句的时间，可以使用 timeit() 方法；repeat() 方法相当于持续多次调用 timeit() 方法并将结果返回为一个列表。

stmt 和 setup 参数也可以是可供调用但没有参数的对象，这将会在一个计时函数中嵌套调用它们，然后被 timeit() 所执行。注意，由于额外的调用，计时开销会相对略到。


**- timeit(number=1000000) **

功能：计算语句执行 number 次的时间。

它会先执行一次 setup 参数的语句，然后计算 stmt 参数的语句执行 number 次的时间，返回值是以秒为单位的浮点数。number 参数的默认值是一百万，stmt、setup 和 timer 参数由 timeit.Timer 类的构造函数传递。

*注意：默认情况下，timeit() 在计时的时候会暂时关闭 Python 的垃圾回收机制。这样做的优点是计时结果更具有可比性，但缺点是 GC（garbage collection，垃圾回收机制的缩写）有时候是测量函数性能的一个重要组成部分。如果是这样的话，GC 可以在 setup 参数执行第一条语句的时候被重新启动，例如：

```
timeit.Timer('for i in range(10): oct(i)', 'gc.enable()').timeit()
```

**- repeat(repeat=3, number=1000000) **

功能：重复调用 timeit()。

repeat() 方法相当于持续多次调用 timeit() 方法并将结果返回为一个列表。repeat 参数指定重复的次数，number 参数传递给 timeit() 方法的 number 参数。

*注意：人们很容易计算出平均值和标准偏差，但这并不是非常有用。在典型的情况下，最低值取决于你的机器可以多快地运行给定的代码段；在结果中更高的那些值通常不是由于 Python 的速度导致，而是因为其他进程干扰了你的计时精度。所以，你所应感兴趣的只有结果的最低值（可以用 min() 求出）。*


**- print_exc(file=None) **

功能：输出计时代码的回溯（Traceback）

典型的用法：

```
t = Timer(...)       # outside the try/except
try:
    t.timeit(...)    # or t.repeat(...)
except Exception:
    t.print_exc()
```

标准回溯的优点是在编译模板中，源语句行会被显示出来。可选的 file 参数指定将回溯发送的位置，默认是发送到 sys.stderr。


**命令行界面**

当被作为命令行程序调用时，可以使用下列选项：

```
python -m timeit [-n N] [-r N] [-s S] [-t] [-c] [-h] [statement ...]
```

各个选项的含义：

| 选项                | 原型         | 含义                                       |
| ----------------- | ---------- | ---------------------------------------- |
| -n N              | --number=N | 执行指定语句（段）的次数                             |
| -r N              | --repeat=N | 重复测量的次数（默认 3 次）                          |
| -s S              | --setup=S  | 指定初始化代码或构建环境的导入语句（默认是 pass）              |
| -p                | --process  | 测量进程时间而不是实际执行时间（使用 time.process_time() 代替默认的 time.perf_counter()） |
| 以下是 Python3.3 新增： |            |                                          |
| -t                | --time     | 使用 time.time()（不推荐）                      |
| -c                | --clock    | 使用 time.clock()（不推荐）                     |
| -v                | --verbose  | 打印原始的计时结果，输出更大精度的数值                      |
| -h                | --help     | 打印一个简短的用法信息并退出                           |

示例

以下演示如果在开始的时候设置初始化语句：

命令行：

```
$ python -m timeit -s 'text = "I love FishC.com!"; char = "o"'  'char in text'
10000000 loops, best of 3: 0.0877 usec per loop
$ python -m timeit -s 'text = "I love FishC.com!"; char = "o"'  'text.find(char)'
1000000 loops, best of 3: 0.342 usec per loop
```

使用 timeit 模块：

```
>>> import timeit
>>> timeit.timeit('char in text', setup='text = "I love FishC.com!"; char = "o"')
0.41440500499993504
>>> timeit.timeit('text.find(char)', setup='text = "I love FishC.com!"; char = "o"')
1.7246671520006203
```

使用 Timer 对象：

```
>>> import timeit
>>> t = timeit.Timer('char in text', setup='text = "I love FishC.com!"; char = "o"')
>>> t.timeit()
0.3955516149999312
>>> t.repeat()
[0.40193588800002544, 0.3960157959998014, 0.39594301399984033]
```

以下演示包含多行语句如何进行测量：

（我们通过 hasattr() 和 try/except 两种方法测试属性是否存在，并且比较它们之间的效率）

命令行：

```
$ python -m timeit 'try:' '  str.__bool__' 'except AttributeError:' '  pass'
100000 loops, best of 3: 15.7 usec per loop
$ python -m timeit 'if hasattr(str, "__bool__"): pass'
100000 loops, best of 3: 4.26 usec per loop

$ python -m timeit 'try:' '  int.__bool__' 'except AttributeError:' '  pass'
1000000 loops, best of 3: 1.43 usec per loop
$ python -m timeit 'if hasattr(int, "__bool__"): pass'
100000 loops, best of 3: 2.23 usec per loop
```

使用 timeit 模块：

```
>>> import timeit
>>> # attribute is missing
>>> s = """\
... try:
...     str.__bool__
... except AttributeError:
...     pass
... """
>>> timeit.timeit(stmt=s, number=100000)
0.9138244460009446
>>> s = "if hasattr(str, '__bool__'): pass"
>>> timeit.timeit(stmt=s, number=100000)
0.5829014980008651
>>>
>>> # attribute is present
>>> s = """\
... try:
...     int.__bool__
... except AttributeError:
...     pass
... """
>>> timeit.timeit(stmt=s, number=100000)
0.04215312199994514
>>> s = "if hasattr(int, '__bool__'): pass"
>>> timeit.timeit(stmt=s, number=100000)
0.08588060699912603
```

**为了使 timeit 模块可以测量你的函数，你可以在 setup 参数中通过 import 语句导入：**

```
def test():
    """Stupid test function"""
    L = [i for i in range(100)]

if __name__ == '__main__':
    import timeit
    print(timeit.timeit("test()", setup="from __main__ import test"))
```

## 基本用法

```python
import datetime, calendar
now_time = datetime.datetime.now() # 表示当前时间
day_num = datetime.datetime.now().isoweekday() # 返回1-7，代表周一到周日，当前时间所在本周第几天；
datetime.datetime.now().weekday() # 返回的0-6，代表周一到周日
monday = (now_time - datetime.timedelta(days=day_num-1)) # 本周周一时间
```

### 获取今天是星期几和本月的天数

```python
import datetime, calendar
def test1():
  # 获取当前日期
    now_time = datetime.datetime.now()
    # 获取当前时间的星期数和月数
    week, days_num = calendar.monthrange(now_time.year, now_time.month)
     # 返回周几和当月的天数
    return week, days_num

```

### 本周x

```python
def get_week(day=1):
    """
    本周x
    :param day:
    :return:
    """
    d = datetime.now()
    dayscount = timedelta(days=d.isoweekday())
    dayto = d - dayscount
    sixdays = timedelta(days=-day)
    dayfrom = dayto - sixdays
    date_from = datetime(dayfrom.year, dayfrom.month, dayfrom.day, 0, 0, 0)
    return str(date_from)[0:4] + '年' + str(date_from)[5:7] + '月' + str(date_from)[8:10] + '日'
```

### 获取下个月第一天和上个月最后一天

```python
import datetime, calendar
def test2():
  # 获取当前日期
    now_time = datetime.datetime.now()
    # 获取当前时间的星期数和天数
    week, days_num = calendar.monthrange(now_time.year, now_time.month)
    # 获取本月的最后一天
    end_day_in_mouth = now_time.replace(day=days_num)
    # 获取下月的第一天
    next_mouth = end_day_in_mouth + datetime.timedelta(days=1)
    date_from = datetime(next_mouth.year, next_mouth.month, next_mouth.day, 0, 0, 0)
    return str(date_from)[0:4] + '年' + str(date_from)[5:7] + '月' + str(date_from)[8:10] + '日'

```

上个月最后一天

```python

def test2():
  # 获取当前日期
    now_time = datetime.datetime.now()
    # 获取本月的第一天
    end_day_in_mouth = now_time.replace(day=1)
    # 获取上月的最后一天
    next_mouth = end_day_in_mouth - datetime.timedelta(days=1)
    # 返回上月的月份
    return next_mouth.month

```

### 上周x和下周x

```python
def get_lastweek(day=1):
    """
    周几
    :param day:
    :return:
    """
    d = datetime.now()
    dayscount = timedelta(days=d.isoweekday())
    dayto = d - dayscount
    sixdays = timedelta(days=7 - day)
    dayfrom = dayto - sixdays
    date_from = datetime(dayfrom.year, dayfrom.month, dayfrom.day, 0, 0, 0)
    return str(date_from)[0:4] + '年' + str(date_from)[5:7] + '月' + str(date_from)[8:10] + '日'
    
    
def get_nextweek(day=1):
    """
    周几
    :param day:
    :return:
    """
    d = datetime.now()
    dayscount = timedelta(days=d.isoweekday())
    dayto = d - dayscount
    sixdays = timedelta(days=-7 - day)
    dayfrom = dayto - sixdays
    date_from = datetime(dayfrom.year, dayfrom.month, dayfrom.day, 0, 0, 0)
    return str(date_from)[0:4] + '年' + str(date_from)[5:7] + '月' + str(date_from)[8:10] + '日' 
 
```
