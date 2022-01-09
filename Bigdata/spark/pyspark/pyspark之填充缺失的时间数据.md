## pyspark之填充缺失的时间数据

这里的场景是，原始数据有两个表示时间的字段：日期和小时，以及对应时间的数据值(比如某个网站的访问量，在凌晨一般没有，白天有)。只有数据值不为0的时候才会记录，因此数据值为0的时间段是没有的。但我们可能需要这些数据，因此就要用到填充功能。

下面会举一个例子来说明。

首先导入需要用到的包，这里的pyspark版本是2.2.0，python版本是2.7。

```
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from pyspark.sql import SparkSession, SQLContext
import pyspark.sql.functions as F
from pyspark.sql.types import *
from pyspark.sql.window import Window
```

创建一个spark会话（如果使用的是shell，不需要此步骤）：

```
spark = SparkSession.builder.appName("test").enableHiveSupport().getOrCreate()
```

创建一个dataframe，有5列6行数据：

```
df = spark.createDataFrame(
    [(1, "a", 10, "2019-09-20", "1"), (2, "a", 20, "2019-09-20", "3"), (3, "a", 5, "2019-09-21", "5"), (4, "b", 8, "2019-09-20", "7"), (5, "b", 9, "2019-09-21", "9"), (6, "b", 16, "2019-09-21", "11")], 
    ["id", "category", "num", "d", "h"])
```

原始数据表示时间的两列是日期(d)和小时(h)，将其转换为时间戳，在此之前，先将h列转换成整数类型：

```
df = df.withColumn("h", df["h"].cast(IntegerType()))
df = df.withColumn("time", F.from_unixtime((F.unix_timestamp(F.col("d"), "yyyy-MM-dd")+F.col("h")*3600)).cast("timestamp"))
```

看一看数据：

```
df.take(6)
```

输出：

```
[Row(id=1, category=u'a', num=10, d=u'2019-09-20', h=1, time=datetime.datetime(2019, 9, 20, 9, 0)),
 Row(id=2, category=u'a', num=20, d=u'2019-09-20', h=3, time=datetime.datetime(2019, 9, 20, 11, 0)),
 Row(id=3, category=u'a', num=5, d=u'2019-09-21', h=5, time=datetime.datetime(2019, 9, 21, 13, 0)),
 Row(id=4, category=u'b', num=8, d=u'2019-09-20', h=7, time=datetime.datetime(2019, 9, 20, 15, 0)),
 Row(id=5, category=u'b', num=9, d=u'2019-09-21', h=9, time=datetime.datetime(2019, 9, 21, 17, 0)),
 Row(id=6, category=u'b', num=16, d=u'2019-09-21', h=11, time=datetime.datetime(2019, 9, 21, 19, 0))]
```

可以发现，time列显示的时间并不是我们希望得到的时间，不过没关系，这个不影响我们数据的填充。

下面介绍两种方法来填充数据。

第一种方法，根据数据中的最小时间和最大时间，生成这个时间段的所有时间数据，再和原始表做left outer join。

先获取最小时间和最大时间

```
# 得到数据中的最小时间和最大时间，这里得到的minp和maxp是(1568941200, 1569063600)，可以用python代码转换一下
minp, maxp = df.select(F.min("time").cast("long"), F.max("time").cast("long")).first()
# print(datetime.datetime.utcfromtimestamp(1568941200))
# 2019-09-20 01:00:00
# 结果和原始时间一样！神奇不！
```

根据最小时间和最大时间，以小时为单位，生成这个时间段的所有时间数据：

```
# 时间间隔，这里是以小时为单位，所以是60*60，即3600秒
step = 60 * 60  
reference = spark.range((minp / step) * step, ((maxp / step) + 1) * step, step).select(F.col("id").cast("timestamp").alias("time"))
reference.take(3)
```

输出：

```
[Row(time=datetime.datetime(2019, 9, 20, 9, 0)),
 Row(time=datetime.datetime(2019, 9, 20, 10, 0)),
 Row(time=datetime.datetime(2019, 9, 20, 11, 0))]
```

这里有两个category，a和b，假如我们希望对于每个category，都有完整的时间数据，要怎么做呢？那就要用到笛卡尔积了：

```
# 我们希望对于每个category，都有每个时间段的数据，因此需要将时间与category做笛卡尔积
cate = dftest.select('category').distinct()
reference2 = cate.crossJoin(reference)   # 笛卡尔积
```

笛卡尔积的结果就是所有我们需要的时间段数据，再将其与原始表做left outer join，就能得到我们想要的结果

```
df1 = reference2.join(df, ["category", "time"], "leftouter")
```



此时df1的前几行是这样的：

```
[Row(category=u'a', time=datetime.datetime(2019, 9, 20, 9, 0), id=1, num=10, d=u'2019-09-20', h=1),
 Row(category=u'a', time=datetime.datetime(2019, 9, 20, 10, 0), id=None, num=None, d=None, h=None),
 Row(category=u'a', time=datetime.datetime(2019, 9, 20, 11, 0), id=2, num=20, d=u'2019-09-20', h=3),
 Row(category=u'a', time=datetime.datetime(2019, 9, 20, 12, 0), id=None, num=None, d=None, h=None),
 Row(category=u'a', time=datetime.datetime(2019, 9, 20, 13, 0), id=None, num=None, d=None, h=None)]
```

发现填充的数据的id，num，d，h都是空的，那么就需要补充这些数据的值了：

```
# 补id、num、d、h
df1 = df1.withColumn("d", F.to_date(F.col("time")).cast(StringType()))
df1 = df1.withColumn("h", F.hour(F.col("time")).cast(IntegerType()))
df1 = df1.fillna(0, subset=['num'])
df1 = df1.fillna(0, subset=['id'])
```

再来看看df1的前5行：

```
[Row(category=u'a', time=datetime.datetime(2019, 9, 20, 9, 0), id=1, num=10, d=u'2019-09-20', h=1),
 Row(category=u'a', time=datetime.datetime(2019, 9, 20, 10, 0), id=0, num=0, d=u'2019-09-20', h=2),
 Row(category=u'a', time=datetime.datetime(2019, 9, 20, 11, 0), id=2, num=20, d=u'2019-09-20', h=3),
 Row(category=u'a', time=datetime.datetime(2019, 9, 20, 12, 0), id=0, num=0, d=u'2019-09-20', h=4),
 Row(category=u'a', time=datetime.datetime(2019, 9, 20, 13, 0), id=0, num=0, d=u'2019-09-20', h=5)]
```

 可以发现，转换为d和h后，时间就变成我们想要的了，time和d、h看起来不是一个时间。。。事实上，写入hive表后，time也会变成我们想要的时间，这里显示的时间不准确，可能是有别的未可知原因。

 用df1.count()查看，有70列数据。可以想一下，为什么是70行。

方法一需要生成所有category的所有时间段的数据，再和原始表join，在数据量很大的时候，效率比较低。方法二则是生成原始表所没有的数据，再和原始表做union。

方法二的主要思想是，通过每个category下time的排序，找出相邻两个time之间的缺失time，然后生成缺失time的数据。

首先针对每个category下的数据，得到该行数据对应的time的上一个已有的time：

```
# 同一个category的上一个时间
tempDf = df.withColumn('pre_time', F.lag(df['time']).over(Window.partitionBy("category").orderBy("time")))
```

得到此时间与上一个时间的时间差：

```
# 时间差
tempDf = tempDf.withColumn('diff', F.unix_timestamp(F.col("time"), "yyyy-MM-dd HH:mm:ss")-F.unix_timestamp(F.col("pre_time"), "yyyy-MM-dd HH:mm:ss"))
```

这里的时间差是以秒为单位的，当时间差为3600秒时，说明两个时间之间没有缺失的时间，大于3600秒时才有。因此，要针对这部分数据找出缺失的时间：

```
fill_dates = F.udf(lambda x,z:[x-y for y in range(3600, z, 3600)], ArrayType(IntegerType()))
tempDf = tempDf.filter(F.col("diff") > 3600)\
    .withColumn("next_dates", fill_dates(F.unix_timestamp(F.col("time")), F.col("diff")))
```

这里的fill_dates是一个udf函数，输入当前时间x，以及时间差z，以3600秒为步长，得到当前时间与上一个时间之间缺失的那些时间，即这里的next_dates，它是一个list。可以用explode函数将这个list拆分，得到多行数据：

```
tempDf = tempDf.withColumn("time", F.explode(F.col("next_dates")))
```

再做一些格式转换，以及d和h的生成，num和id的补充：

```
tempDf = tempDf.withColumn("time", F.col("time").cast(TimestampType()))\
    .withColumn("d", F.to_date(F.col("time")).cast(StringType()))\
    .withColumn("h", F.hour(F.col("time")).cast(IntegerType()))\
    .withColumn("num", F.lit("0")).withColumn("id", F.lit("0"))
```

看两行数据：

```
[Row(id=u'0', category=u'a', num=u'0', d=u'2019-09-20', h=2, time=datetime.datetime(2019, 9, 20, 10, 0), pre_time=datetime.datetime(2019, 9, 20, 9, 0), diff=7200, next_dates=[1568944800]),
 Row(id=u'0', category=u'a', num=u'0', d=u'2019-09-21', h=4, time=datetime.datetime(2019, 9, 21, 12, 0), pre_time=datetime.datetime(2019, 9, 20, 11, 0), diff=93600, next_dates=[1569038400, 1569034800, 1569031200, 1569027600, 1569024000, 1569020400, 1569016800, 1569013200, 1569009600, 1569006000, 1569002400, 1568998800, 1568995200, 1568991600, 1568988000, 1568984400, 1568980800, 1568977200, 1568973600, 1568970000, 1568966400, 1568962800, 1568959200, 1568955600, 1568952000])]
```

next_dates是时间戳格式。

再将这个表和原始表union一下就好了，注意要drop不需要的列：

```
tempDf = tempDf.drop(*['next_dates', 'diff', 'pre_time'])
df2 = df.union(tempDf)
df2.orderBy('category', 'time').select('category', 'time','id','num','d','h').take(5)
```

输出：

```
[Row(category=u'a', time=datetime.datetime(2019, 9, 20, 9, 0), id=u'1', num=u'10', d=u'2019-09-20', h=1),
 Row(category=u'a', time=datetime.datetime(2019, 9, 20, 10, 0), id=u'0', num=u'0', d=u'2019-09-20', h=2),
 Row(category=u'a', time=datetime.datetime(2019, 9, 20, 11, 0), id=u'2', num=u'20', d=u'2019-09-20', h=3),
 Row(category=u'a', time=datetime.datetime(2019, 9, 20, 12, 0), id=u'0', num=u'0', d=u'2019-09-20', h=4),
 Row(category=u'a', time=datetime.datetime(2019, 9, 20, 13, 0), id=u'0', num=u'0', d=u'2019-09-20', h=5)]
```

df2.count()后发现只有58行。

原来这里会针对每个category有一个最大时间和最小时间，所以得到的结果数是比方法一少的。疏忽了！

如果想得到和方法一一样的结果，可以这么写：

```
minp, maxp = df.select(F.min("time").cast("long"), F.max("time").cast("long")).first()
newRow = spark.createDataFrame([(minp,),(maxp,)], ["time"])
newRow = newRow.withColumn('time', F.col("time").cast("timestamp"))
cate = df.select('category').distinct()
newRow = cate.crossJoin(newRow)   # 笛卡尔积
newRow.take(10)
```

先针对每个category，生成最小时间和最大时间的数据。这里有两个category，所以会有2*2=4行数据

输出：

```
[Row(category=u'a', time=datetime.datetime(2019, 9, 20, 9, 0)),
 Row(category=u'b', time=datetime.datetime(2019, 9, 20, 9, 0)),
 Row(category=u'a', time=datetime.datetime(2019, 9, 21, 19, 0)),
 Row(category=u'b', time=datetime.datetime(2019, 9, 21, 19, 0))]
```

然后将生成的数据和原始表left join，得到其他字段(id, num, d, h)的值，这是为了保证对于df中已有的数据，newRow的相应行是一样的，后续union的时候可以去掉重复数据：

```
newRow = newRow.join(df, ['category', 'time'], "left")
newdf = df.select('category', 'time', 'id', 'num', 'd', 'h').union(newRow.select('category', 'time', 'id', 'num', 'd', 'h'))
newdf = newdf.distinct()
newdf = newdf.fillna(0, subset=['num'])
newdf = newdf.fillna(0, subset=['id'])
newdf = newdf.withColumn("d", F.to_date(F.col("time")).cast(StringType()))\
    .withColumn("h", F.hour(F.col("time")).cast(IntegerType()))
newdf.take(10)
```

输出：

```
[Row(category=u'a', time=datetime.datetime(2019, 9, 20, 11, 0), id=2, num=20, d=u'2019-09-20', h=3),
 Row(category=u'a', time=datetime.datetime(2019, 9, 20, 9, 0), id=1, num=10, d=u'2019-09-20', h=1),
 Row(category=u'b', time=datetime.datetime(2019, 9, 20, 15, 0), id=4, num=8, d=u'2019-09-20', h=7),
 Row(category=u'a', time=datetime.datetime(2019, 9, 21, 13, 0), id=3, num=5, d=u'2019-09-21', h=5),
 Row(category=u'b', time=datetime.datetime(2019, 9, 21, 17, 0), id=5, num=9, d=u'2019-09-21', h=9),
 Row(category=u'b', time=datetime.datetime(2019, 9, 21, 19, 0), id=6, num=16, d=u'2019-09-21', h=11),
 Row(category=u'a', time=datetime.datetime(2019, 9, 21, 19, 0), id=0, num=0, d=u'2019-09-21', h=11),
 Row(category=u'b', time=datetime.datetime(2019, 9, 20, 9, 0), id=0, num=0, d=u'2019-09-20', h=1)]
```

这样，每个category下都有了最小时间和最大时间的数据了。

再用和之前一样的方法：

```
fill_dates = F.udf(lambda x,z:[x-y for y in range(3600, z, 3600)], ArrayType(IntegerType()))
# 同一个category的上一个时间
tempDf = newdf.withColumn('pre_time', F.lag(newdf['time']).over(Window.partitionBy("category").orderBy("time")))
#时间差
tempDf = tempDf.withColumn('diff', F.unix_timestamp(F.col("time"), "yyyy-MM-dd HH:mm:ss")-F.unix_timestamp(F.col("pre_time"), "yyyy-MM-dd HH:mm:ss"))
tempDf = tempDf.filter(F.col("diff") > 3600)\
    .withColumn("next_dates", fill_dates(F.unix_timestamp(F.col("time")), F.col("diff")))\
    .withColumn("time", F.explode(F.col("next_dates")))\
    .withColumn("time", F.col("time").cast(TimestampType()))\
    .withColumn("d", F.to_date(F.col("time")).cast(StringType()))\
    .withColumn("h", F.hour(F.col("time")).cast(IntegerType()))\
    .withColumn("num", F.lit("0")).withColumn("id", F.lit("0"))
tempDf = tempDf.drop(*['next_dates', 'diff', 'pre_time'])
df3 = newdf.select('category', 'time', 'id', 'num', 'd', 'h').union(tempDf.select('category', 'time', 'id', 'num', 'd', 'h'))
```

此时df3.count()就是70行啦！



如果我们想要计算每个category的每一个时间点的前后1小时这个时间段（一共3个小时）的平均num，就可以这么做：

```
# 计算前后各1小时的平均num值，必须严格前后1小时
windowSpec = Window.partitionBy("category").orderBy("d", "h").rowsBetween(-1, 1)
df3 = df3.withColumn("movavg_sameday", F.avg("num").over(windowSpec))\
    .withColumn("movavg_sameday_data", F.collect_list("num").over(windowSpec))
df3.take(5)
```

注意这里要partitionBy，也就是分区计算，不然会出现两个category的时间混在一起被计算。

输出：

```
[Row(category=u'a', time=datetime.datetime(2019, 9, 20, 9, 0), id=u'1', num=u'10', d=u'2019-09-20', h=1, movavg_sameday=5.0, movavg_sameday_data=[u'10', u'0']),
 Row(category=u'a', time=datetime.datetime(2019, 9, 20, 10, 0), id=u'0', num=u'0', d=u'2019-09-20', h=2, movavg_sameday=10.0, movavg_sameday_data=[u'10', u'0', u'20']),
 Row(category=u'a', time=datetime.datetime(2019, 9, 20, 11, 0), id=u'2', num=u'20', d=u'2019-09-20', h=3, movavg_sameday=6.666666666666667, movavg_sameday_data=[u'0', u'20', u'0']),
 Row(category=u'a', time=datetime.datetime(2019, 9, 20, 12, 0), id=u'0', num=u'0', d=u'2019-09-20', h=4, movavg_sameday=6.666666666666667, movavg_sameday_data=[u'20', u'0', u'0']),
 Row(category=u'a', time=datetime.datetime(2019, 9, 20, 13, 0), id=u'0', num=u'0', d=u'2019-09-20', h=5, movavg_sameday=0.0, movavg_sameday_data=[u'0', u'0', u'0'])]
```

Window是一个很有用的函数，可以用于取想要的窗口数据。

上述代码中的rowsBetween是指从当前行算起(当前行是第0行)，某两行之间的窗口，比如这里是-1和1，也就是当前行的前一行和后一行之间的这三行。

还有一个方法是rangeBetween(x,y)，是指当前行的某个字段，比如这里的num，取这个字段某个区间的那些数据，即num值处于[num+x, num+y]这个区间的那些行。



参考资料：

<https://stackoverflow.com/questions/42411184/filling-gaps-in-timeseries-spark>