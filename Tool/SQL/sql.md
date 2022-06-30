##基本概念

**1、SQL是一种声明式语言**

SQL 语言声明的是结果集的属性，计算机会根据 SQL 所声明的内容来从数据库中挑选出符合声明的数据，而不是像传统编程思维去指示计算机如何操作

**2、语法顺序和执行顺序不一样**

语法顺序：

- Select [distinct]

- from

- where 

- group by

- having

- union

- order by

语句执行顺序：

- from

- where

- group by

- having

- Select

- distinct

- union

- order by

where条件中不能跟聚合函数，而having后面可以；执行顺序where>聚合函数(sum,min,max,avg,count)>having 

https://blog.csdn.net/moguxiansheng1106/article/details/44258499

**3、 SQL 语言的核心是对表的引用（table references）**

根据 SQL 标准，FROM 语句被定义为：

> <from clause> ::= FROM <table reference> [ { <comma> <table reference> }... ]

FROM 语句的“输出”是一张联合表，来自于所有引用的表在某一维度上的联合。我们们慢慢来分析：

> FROM a, b

上面这句 FROM 语句的输出是一张联合表，联合了表 a 和表 b 。如果 a 表有三个字段， b 表有 5 个字段，那么这个“输出表”就有 8 （ =5+3）个字段。

这个联合表里的数据是 a*b，即 a 和 b 的笛卡尔积。换句话说，也就是 a 表中的每一条数据都要跟 b 表中的每一条数据配对。如果 a 表有3 条数据， b 表有 5 条数据，那么联合表就会有 15 （ =5*3）条数据。

**4、 灵活引用表能使 SQL 语句变得更强大**

灵活引用表能使 SQL 语句变得更强大。一个简单的例子就是 JOIN 的使用。严格的说 JOIN 语句并非是 SELECT 中的一部分，而是一种特殊的表引用语句。 SQL 语言标准中表的连接定义如下：

```
<table reference> ::=
    <table name>
  | <derived table>
  | <joined table>
```

就拿之前的例子来说：

> FROM a, b

a 可能输如下表的连接：

> a1 JOIN a2 ON a1.id = a2.id

将它放到之前的例子中就变成了：

> FROM a1 JOIN a2 ON a1.id = a2.id, b

尽管将一个连接表用逗号跟另一张表联合在一起并不是常用作法，但是你的确可以这么做。结果就是，最终输出的表就有了 a1+a2+b 个字段了。

**5、 SQL 语句中推荐使用表连接**

尽量不要使用逗号来代替 JOIN 进行表的连接，这样会提高你的 SQL 语句的可读性，并且可以避免一些错误。

- 安全。 JOIN 和要连接的表离得非常近，这样就能避免错误。
- 更多连接的方式，JOIN 语句能去区分出来外连接和内连接等。

**6、 SQL 语句中不同的连接操作**

SQL 语句中，表连接的方式从根本上分为五种：

- EQUI JOIN
- SEMI JOIN
- ANTI JOIN
- CROSS JOIN
- DIVISION

**EQUI JOIN**

这是一种最普通的 JOIN 操作，它包含两种连接方式：

- INNER JOIN（或者是 JOIN ）：如果表中有至少一个匹配，则返回行
- OUTER JOIN（包括： LEFT 、 RIGHT、 FULL OUTER JOIN）
  - LEFT JOIN: 即使右表中没有匹配，也从左表返回所有的行
  - RIGHT JOIN: 即使左表中没有匹配，也从右表返回所有的行
  - FULL JOIN: 只要其中一个表中存在匹配，就返回行

**SEMI JOIN**

这种连接关系在 SQL 中有两种表现方式：使用 IN，或者使用 EXISTS。“ SEMI ”在拉丁文中是“半”的意思。这种连接方式是只连接目标表的一部分。这是什么意思呢？再想一下上面关于作者和书名的连接。我们想象一下这样的情况：我们不需要作者 / 书名这样的组合，只是需要那些在书名表中的书的作者信息。那我们就能这么写：

> -- Using IN
> FROM author
> WHERE author.id IN (SELECT book.author_id FROM book)
>
> -- Using EXISTS
> FROM author
> WHERE EXISTS (SELECT 1 FROM book WHERE book.author_id = author.id)

- IN比 EXISTS 的可读性更好
- EXISTS 比IN 的表达性更好（更适合复杂的语句）
- 二者之间性能没有差异（但对于某些数据库来说性能差异会非常大）

因为使用 INNER JOIN 也能得到书名表中书所对应的作者信息，所以很多初学者机会认为可以通过 DISTINCT 进行去重，然后将 SEMI JOIN 语句写成这样：

> -- Find only those authors who also have books
> SELECT DISTINCT first_name, last_name
> FROM author
> JOIN book ON author.id = book.author_id

这是一种很糟糕的写法，原因如下：

- SQL 语句性能低下：因为去重操作（ DISTINCT ）需要数据库重复从硬盘中读取数据到内存中。（译者注： DISTINCT 的确是一种很耗费资源的操作，但是每种数据库对于 DISTINCT 的操作方式可能不同）。
- 这么写并非完全正确：尽管也许现在这么写不会出现问题，但是随着 SQL 语句变得越来越复杂，你想要去重得到正确的结果就变得十分困难。

更多的关于滥用 DISTINCT 的危害可以参考这篇博文：http://blog.jooq.org/2013/07/30/10-common-mistakes-java-developers-make-when-writing-sql/。

**ANTI JOIN**

这种连接的关系跟 SEMI JOIN 刚好相反。在 IN 或者 EXISTS 前加一个 NOT 关键字就能使用这种连接。举个例子来说，我们列出书名表里没有书的作者：

> -- Using IN
> FROM author
> WHERE author.id NOT IN (SELECT book.author_id FROM book)
>
> -- Using EXISTS
> FROM author
> WHERE NOT EXISTS (SELECT 1 FROM book WHERE book.author_id = author.id)

关于性能、可读性、表达性等特性也完全可以参考 SEMI JOIN。

这篇博文介绍了在使用 NOT IN 时遇到 NULL 应该怎么办：http://blog.jooq.org/2012/01/27/sql-incompatibilities-not-in-and-null-values/。

**CROSS JOIN**

这个连接过程就是两个连接的表的乘积：即将第一张表的每一条数据分别对应第二张表的每条数据。我们之前见过，这就是逗号在 FROM 语句中的用法。在实际的应用中，很少有地方能用到 CROSS JOIN，但是一旦用上了，你就可以用这样的 SQL语句表达：

> -- Combine every author with every book
> author CROSS JOIN book

**DIVISION**

DIVISION 的确是一个怪胎。简而言之，如果 JOIN 是一个乘法运算，那么 DIVISION 就是 JOIN 的逆过程。DIVISION 的关系很难用 SQL 表达出来。三篇文章：

- http://blog.jooq.org/2012/03/30/advanced-sql-relational-division-in-jooq/
- http://en.wikipedia.org/wiki/Relational_algebra#Division
- https://www.simple-talk.com/sql/t-sql-programming/divided-we-stand-the-sql-of-relational-division/。

SQL 是对表的引用， JOIN 则是一种引用表的复杂方式。但是 SQL 语言的表达方式和实际我们所需要的逻辑关系之间是有区别的，并非所有的逻辑关系都能找到对应的 JOIN 操作，所以这就要我们在平时多积累和学习关系逻辑，这样你就能在以后编写 SQL 语句中选择适当的 JOIN 操作了。

**7、 SQL 中如同变量的派生表**

在这之前，我们学习到过 SQL 是一种声明性的语言，并且 SQL 语句中不能包含变量。但是你能写出类似于变量的语句，这些就叫做派生表：

说白了，所谓的派生表就是在括号之中的子查询：

> -- A derived table
> FROM (SELECT * FROM author)

需要注意的是有些时候我们可以给派生表定义一个相关名（即我们所说的别名）。

> -- A derived table with an alias
> FROM (SELECT * FROM author) a

派生表可以有效的避免由于 SQL 逻辑而产生的问题。举例来说：如果你想重用一个用 SELECT 和 WHERE 语句查询出的结果，这样写就可以（以 Oracle 为例）：

> -- Get authors' first and last names, and their age in days
> SELECT first_name, last_name, age
> FROM (
>   SELECT first_name, last_name, current_date - date_of_birth age
>   FROM author
> )
> -- If the age is greater than 10000 days
> WHERE age > 10000

需要我们注意的是：在有些数据库，以及 SQL ： 1990 标准中，派生表被归为下一级——通用表语句（ common table experssion）。这就允许你在一个 SELECT 语句中对派生表多次重用。上面的例子就（几乎）等价于下面的语句：

> WITH a AS (
>   SELECT first_name, last_name, current_date - date_of_birth age
>   FROM author
> )
> SELECT *
> FROM a
> WHERE age > 10000

当然了，你也可以给“ a ”创建一个单独的视图，这样你就可以在更广泛的范围内重用这个派生表了。更多信息可以阅读下面的文章（http://en.wikipedia.org/wiki/View_%28SQL%29）。

**我们学到了什么？**

我们反复强调，大体上来说 SQL 语句就是对表的引用，而并非对字段的引用。要好好利用这一点，不要害怕使用派生表或者其他更复杂的语句。

**8、 SQL 语句中 GROUP BY 是对表的引用进行的操作**

让我们再回想一下之前的 FROM 语句：

> FROM a, b

现在，我们将 GROUP BY 应用到上面的语句中：

> GROUP BY A.x, A.y, B.z

上面语句的结果就是产生出了一个包含三个字段的新的表的引用。我们来仔细理解一下这句话：当你应用 GROUP BY 的时候， SELECT 后没有使用聚合函数的列，都要出现在 GROUP BY 后面。（译者注：原文大意为“当你是用 GROUP BY 的时候，你能够对其进行下一级逻辑操作的列会减少，包括在 SELECT 中的列”）。

- 需要注意的是：其他字段能够使用聚合函数：

> SELECT A.x, A.y, SUM(A.z)
> FROM A
> GROUP BY A.x, A.y

- 还有一点值得留意的是： MySQL 并不坚持这个标准，这的确是令人很困惑的地方。但是不要被 MySQL 所迷惑。 GROUP BY 改变了对表引用的方式。你可以像这样既在 SELECT 中引用某一字段，也在 GROUP BY 中对其进行分组。

**我们学到了什么？**

GROUP BY，再次强调一次，是在表的引用上进行了操作，将其转换为一种新的引用方式。

**9、 SQL 语句中的 SELECT 实质上是对关系的映射**

我个人比较喜欢“映射”这个词，尤其是把它用在关系代数上。（译者注：原文用词为 projection ，该词有两层含义，第一种含义是预测、规划、设计，第二种意思是投射、映射，经过反复推敲，我觉得这里用映射能够更直观的表达出 SELECT 的作用）。一旦你建立起来了表的引用，经过修改、变形，你能够一步一步的将其映射到另一个模型中。 SELECT 语句就像一个“投影仪”，我们可以将其理解成一个将源表中的数据按照一定的逻辑转换成目标表数据的函数。

通过 SELECT语句，你能对每一个字段进行操作，通过复杂的表达式生成所需要的数据。

SELECT 语句有很多特殊的规则，至少你应该熟悉以下几条：

1. 你仅能够使用那些能通过表引用而得来的字段；

2. 如果你有 GROUP BY 语句，你只能够使用 GROUP BY 语句后面的字段或者聚合函数；

3. 当你的语句中没有 GROUP BY 的时候，可以使用开窗函数代替聚合函数；

4. 当你的语句中没有 GROUP BY 的时候，你不能同时使用聚合函数和其它函数；

5. 有一些方法可以将普通函数封装在聚合函数中；

   ……

一些更复杂的规则多到足够写出另一篇文章了。比如：为何你不能在一个没有 GROUP BY 的 SELECT 语句中同时使用普通函数和聚合函数？（上面的第 4 条）

原因如下：

- 凭直觉，这种做法从逻辑上就讲不通。
- 如果直觉不能够说服你，那么语法肯定能。 SQL : 1999 标准引入了 GROUPING SETS，SQL： 2003 标准引入了 group sets : GROUP BY() 。无论什么时候，只要你的语句中出现了聚合函数，而且并没有明确的 GROUP BY 语句，这时一个不明确的、空的 GROUPING SET 就会被应用到这段 SQL 中。因此，原始的逻辑顺序的规则就被打破了，映射（即 SELECT ）关系首先会影响到逻辑关系，其次就是语法关系。（译者注：这段话原文就比较艰涩，可以简单理解如下：在既有聚合函数又有普通函数的 SQL 语句中，如果没有 GROUP BY 进行分组，SQL 语句默认视整张表为一个分组，当聚合函数对某一字段进行聚合统计的时候，引用的表中的每一条 record 就失去了意义，全部的数据都聚合为一个统计值，你此时对每一条 record 使用其它函数是没有意义的）。

**我们学到了什么？**

SELECT 语句可能是 SQL 语句中最难的部分了，尽管他看上去很简单。其他语句的作用其实就是对表的不同形式的引用。而 SELECT 语句则把这些引用整合在了一起，通过逻辑规则将源表映射到目标表，而且这个过程是可逆的，我们可以清楚的知道目标表的数据是怎么来的。

想要学习好 SQL 语言，就要在使用 SELECT 语句之前弄懂其他的语句，虽然 SELECT 是语法结构中的第一个关键词，但它应该是我们最后一个掌握的。

**10、 SQL 语句中的几个简单的关键词： DISTINCT ， UNION ， ORDER BY 和 OFFSET**

在学习完复杂的 SELECT 语句之后，我们再来看点简单的东西：

- 集合运算（ DISTINCT 和 UNION ）
- 排序运算（ ORDER BY，OFFSET…FETCH）

**集合运算（ set operation）：**

集合运算主要操作在于集合上，事实上指的就是对表的一种操作。从概念上来说，他们很好理解：

- DISTINCT 在映射之后对数据进行去重
- UNION 将两个子查询拼接起来并去重
- UNION ALL 将两个子查询拼接起来但不去重
- EXCEPT 将第二个字查询中的结果从第一个子查询中去掉
- INTERSECT 保留两个子查询中都有的结果并去重

**排序运算（ ordering operation）：**

排序运算跟逻辑关系无关。这是一个 SQL 特有的功能。排序运算不仅在 SQL 语句的最后，而且在 SQL 语句运行的过程中也是最后执行的。使用 ORDER BY 和 OFFSET…FETCH 是保证数据能够按照顺序排列的最有效的方式。其他所有的排序方式都有一定随机性，尽管它们得到的排序结果是可重现的。

OFFSET…SET是一个没有统一确定语法的语句，不同的数据库有不同的表达方式，如 MySQL 和 PostgreSQL 的 LIMIT…OFFSET、SQL Server 和 Sybase 的 TOP…START AT 等。具体关于 OFFSET..FETCH 的不同语法可以参考这篇文章：
http://www.jooq.org/doc/3.1/manual/sql-building/sql-statements/select-statement/limit-clause/。

## 基础语句

```sql
SQL SELECT Column_ FROM Mytable
SQL DISTINCT(放在SELECT后使用)
SQL WHERE(设置条件)
SQL AND & OR(逻辑与&逻辑或)
SQL ORDER BY(排序操作，BY跟排序字段)
SQL INSERT INTO VALUES(插入数据)
SQL UPDATE SET(更新数据)
SQL DELETE FROM(删除数据)
```

## 高级

```sql
SQL LIMIT(取第N到第M条记录)
SQL IN(用于子查询)
SQL BETWEEN AND(设置区间)
SQL LIKE(匹配通配符)
SQL GROUP BY(按组查询)
SQL HAVING(跟在“GROUP BY”语句后面的设置条件语句)
SQL ALIAS(AS)(可以为表或列取别名)
SQL LEFT JOIN/RIGHT/FULL JOIN(左连接/右连接/全连接)
SQL OUT/INNER JOIN(内连接/外连接)
SQL UNION/UNION ALL(并集，后者不去重)
SQL INTERSECT(交集)
SQL EXCEPT(差集)
SQL SELECT INTO(查询结果赋给变量或表)
SQL CREATE TABLE(创建表)
SQL CREATE VIEW AS(创建视图)
SQL CREATE INDEX(创建索引)
SQL CREATE PROCEDURE BEGIN END(创建存储过程)
SQL CREATE TRIGGER T_name BEFORE/AFTER INSERT/UPDATE/DELETE ON MyTable FOR (创建触发器)
SQL ALTER TABLE ADD/MODIFY COLUMN/DROP(修改表:增加字段/修改字段属性/删除字段)
SQL UNIQUE(字段、索引的唯一性约束)
SQL NOT NULL(定义字段值非空)
SQL AUTO_INCREMENT(字段定义为自动添加类型)
SQL PRIMARY KEY(字段定义为主键)
SQL FOREIGN KEY(创建外键约束)
SQL CHECK(限制字段值的范围)
SQL DROP TABLE/INDEX/VIEW/PROCEDURE/TRIGGER (删除表/索引/视图/存储过程/触发器)
SQL TRUNCATE TABLE(删除表数据，不删表结构)
```

## 函数

###常用的文本处理函数

```sql
SQL Length(str)(返回字符串str长度)
SQL Locate(substr,str)(返回子串substr在字符串str第一次出现的位置)
SQL LTrim(str)(移除字符串str左边的空格)
SQL RTrim(str)(移除字符串str右边的空格)
SQL Trim(str)(移除字符串str左右两边的空格)
SQL Left(str,n)(返回字符串str最左边的n个字符)
SQL Right(str,n)(返回字符串str最右边的n个字符)
SQL Soundex()
SQL SubString(str,pos,len)/Substr()(从pos位置开始截取str字符串中长度为的字符串)
SQL Upper(str)/Ucase(str)(小写转化为大写)
SQL Lower(str)/Lcase(str)(大写转化为小写)
```

### 常用的日期与时间处理函数

```sql
SQL AddDate()(增加一个日期，天、周等)
SQL AddTime()(增加一个时间，天、周等)
SQL CurDate()(返回当前日期)
SQL CurTime()(返回当前时间)
SQL Date()(返回日期时间的日期部分)
SQL DateDiff()(计算两个日期之差)
SQL Date_Add()(高度灵活的日期运算函数)
SQL Date_Format()(返回一个格式化的日期或时间串)
SQL Day()(返回一个日期的天数部分)
SQL DayOfWeek()(返回一个日期对应的星期几)
SQL Hour()(返回一个时间的小时部分)
SQL Minute()(返回一个时间的分钟部分)
SQL Month()(返回一个日期的月份部分)
SQL Now()(返回当前日期和时间)
SQL Second()(返回一个时间的秒部分)
SQL Time()(返回一个日期时间的时间部分)
SQL Year()(返回一个日期的年份部分)
```

### 常用的数值处理函数

```sql
SQL Avg()(求均值)
SQL Max()(求最大值)
SQL Min()(求最小值)
SQL Sum()(求和)
SQL Count()(统计个数)
SQL Abs()(求绝对值)
SQL Cos()(求一个角度的余弦值)
SQL Exp(n)(求e^n)
SQL Mod()(求余)
SQL Pi()(求圆周率)
SQL Rand()(返回一个随机数)
SQL Sin()(求一个角度的正弦值)
SQL Sqrt()(求一个数的开方)
SQL Tan()(求一个角度的正切值)
SQL Mid(ColumnName,Start,[,length])(得到字符串的一部分)
SQL Round(n,m)(以m位小数来对n四舍五入)
SQL Convert(xxx,TYPE)/Cast(xxx AS TYPE) (把xxx转为TYPE类型的数据)
SQL Format() (用来格式化数值)
SQL First(ColumnName)(返回指定字段中第一条记录)
SQL Last(ColumnName)(返回指定字段中最后一条记录)
```

## 优化技巧

https://zhuanlan.zhihu.com/p/27540896

1、应尽量避免在 where 子句中使用!=或<>操作符，否则将引擎放弃使用索引而进行全表扫描。

2、对查询进行优化，应尽量避免全表扫描，首先应考虑在 where 及 order by 涉及的列上建立索引。

3、应尽量避免在 where 子句中对字段进行 null 值判断，否则将导致引擎放弃使用索引而进行全表扫描，如：

select id from t where num is null

可以在num上设置默认值0，确保表中num列没有null值，然后这样查询：

select id from t where num=0

4、尽量避免在 where 子句中使用 or 来连接条件，否则将导致引擎放弃使用索引而进行全表扫描，如：

select id from t where num=10 or num=20

可以这样查询：

select id from t where num=10

union all

select id from t where num=20

5、下面的查询也将导致全表扫描：(不能前置百分号)

select id from t where name like ‘%c%’

若要提高效率，可以考虑全文检索。

6、in 和 not in 也要慎用，否则会导致全表扫描，如：

select id from t where num in(1,2,3)

对于连续的数值，能用 between 就不要用 in 了：

select id from t where num between 1 and 3

7、如果在 where 子句中使用参数，也会导致全表扫描。因为SQL只有在运行时才会解析局部变量，但优化程序不能将访问计划的选择推迟到运行时；它必须在编译时进行选择。然 而，如果在编译时建立访问计划，变量的值还是未知的，因而无法作为索引选择的输入项。如下面语句将进行全表扫描：

select id from t where num=@num

可以改为强制查询使用索引：

select id from t with(index(索引名)) where num=@num

8、应尽量避免在 where 子句中对字段进行表达式操作，这将导致引擎放弃使用索引而进行全表扫描。如：

select id from t where num/2=100

应改为:

select id from t where num=100*2

9、应尽量避免在where子句中对字段进行函数操作，这将导致引擎放弃使用索引而进行全表扫描。如：

select id from t where substring(name,1,3)=’abc’–name以abc开头的id

select id from t where datediff(day,createdate,’2005-11-30′)=0–’2005-11-30′生成的id

应改为:

select id from t where name like ‘abc%’

select id from t where createdate>=’2005-11-30′ and createdate<’2005-12-1′

10、不要在 where 子句中的“=”左边进行函数、算术运算或其他表达式运算，否则系统将可能无法正确使用索引。

11、在使用索引字段作为条件时，如果该索引是复合索引，那么必须使用到该索引中的第一个字段作为条件时才能保证系统使用该索引，否则该索引将不会被使 用，并且应尽可能的让字段顺序与索引顺序相一致。

12、不要写一些没有意义的查询，如需要生成一个空表结构：

`select col1,col2 into #t from t where 1=0`

这类代码不会返回任何结果集，但是会消耗系统资源的，应改成这样：

`create table #t(…)`

13、很多时候用 exists 代替 in 是一个好的选择：

select num from a where num in(select num from b)

用下面的语句替换：

select num from a where exists(select 1 from b where num=a.num)

14、并不是所有索引对查询都有效，[SQL](https://link.zhihu.com/?target=http%3A//cda.pinggu.org/view/22577.html)是根据表中数据来进行查询优化的，当索引列有大量数据重复时，SQL查询可能不会去利用索引，如一表中有字段 sex，male、female几乎各一半，那么即使在sex上建了索引也对查询效率起不了作用。

15、索引并不是越多越好，索引固然可以提高相应的 select 的效率，但同时也降低了 insert 及 update 的效率，因为 insert 或 update 时有可能会重建索引，所以怎样建索引需要慎重考虑，视具体情况而定。一个表的索引数最好不要超过6个，若太多则应考虑一些不常使用到的列上建的索引是否有 必要。

16.应尽可能的避免更新 clustered 索引数据列，因为 clustered 索引数据列的顺序就是表记录的物理存储顺序，一旦该列值改变将导致整个表记录的顺序的调整，会耗费相当大的资源。若应用系统需要频繁更新 clustered 索引数据列，那么需要考虑是否应将该索引建为 clustered 索引。

17、尽量使用数字型字段，若只含数值信息的字段尽量不要设计为字符型，这会降低查询和连接的性能，并会增加存储开销。这是因为引擎在处理查询和连接时会 逐个比较字符串中每一个字符，而对于数字型而言只需要比较一次就够了。

18、尽可能的使用 varchar/nvarchar 代替 char/nchar ，因为首先变长字段存储空间小，可以节省存储空间，其次对于查询来说，在一个相对较小的字段内搜索效率显然要高些。

19、任何地方都不要使用 select * from t ，用具体的字段列表代替“*”，不要返回用不到的任何字段。

20、尽量使用表变量来代替临时表。如果表变量包含大量数据，请注意索引非常有限（只有主键索引）。

21、避免频繁创建和删除临时表，以减少系统表资源的消耗。

22、临时表并不是不可使用，适当地使用它们可以使某些例程更有效，例如，当需要重复引用大型表或常用表中的某个数据集时。但是，对于一次性事件，最好使 用导出表。

23、在新建临时表时，如果一次性插入数据量很大，那么可以使用 select into 代替 create table，避免造成大量 log ，以提高速度；如果数据量不大，为了缓和系统表的资源，应先create table，然后insert。

24、如果使用到了临时表，在存储过程的最后务必将所有的临时表显式删除，先 truncate table ，然后 drop table ，这样可以避免系统表的较长时间锁定。

25、尽量避免使用游标，因为游标的效率较差，如果游标操作的数据超过1万行，那么就应该考虑改写。

26、使用基于游标的方法或临时表方法之前，应先寻找基于集的解决方案来解决问题，基于集的方法通常更有效。

27、与临时表一样，游标并不是不可使用。对小型数据集使用 FAST_FORWARD 游标通常要优于其他逐行处理方法，尤其是在必须引用几个表才能获得所需的数据时。在结果集中包括“合计”的例程通常要比使用游标执行的速度快。如果开发时 间允许，基于游标的方法和基于集的方法都可以尝试一下，看哪一种方法的效果更好。

28、在所有的存储过程和触发器的开始处设置 SET NOCOUNT ON ，在结束时设置 SET NOCOUNT OFF 。无需在执行存储过程和触发器的每个语句后向客户端发送 DONE_IN_PROC 消息。

29、尽量避免向客户端返回大数据量，若[数据](https://link.zhihu.com/?target=http%3A//cda.pinggu.org/)量过大，应该考虑相应需求是否合理。

30、尽量避免大事务操作，提高系统并发能力。

##操作

**条件选择 and， or，in**

```
select * from DataAnalyst
where (city = '上海' and positionName = '数据分析师') 
   or (city = '北京' and positionName = '数据产品经理')
```

```
select * from DataAnalyst
where city in ('北京','上海','广州','深圳','南京')
```

**区间数值，between and**

```
select * from DataAnalyst
where companyId between 10000 and 20000
```

between and 包括数值两端的边界，等同于 companyId >=10000 and companyId <= 20000。 

**模糊查找，like**

```
select * from DataAnalyst
where positionName like '%数据分析%'
```

where name like ’A%’ or ’B%’（错误），where name like('A%' or 'B%' )（错误），

WHERE name LIKE 'A%' OR name LIKE 'B%';（正确）。

**%代表的是通配符 **

**not，代表逻辑的逆转，常见not in、not like、not null等。 **

in的对立面并不是NOT IN！not in等价的含义是<> all，例如In(‘A’,’B’)：A或者B；not in (‘A’,’B’)：不是A且B。 

**group by**

```
select city,count(1) from DataAnalyst
group by city
# 去重
select city,count(distinct positionId) from DataAnalyst
group by city
# 多维度
select city,workYear,count(distinct positionId) from DataAnalyst
group by city,workYear
```

上述语句，使用count函数，统计计数了每个城市拥有的职位数量。括号里面的1代表以第一列为计数标准。 

除了count，还有max，min，sum，avg等函数，也叫做聚合函数。 

**逻辑判断**

统计各个城市中有多少数据分析职位，其中，电商领域的职位有多少，在其中的占比？ 

```
select if(industryField like '%电子商务%',1,0) from DataAnalyst
```

利用if判断出哪些是电商行业的数据分析师，哪些不是。if函数中间的字段代表为true时返回的值，不过因为包含重复数据，我们需要将其改成positionId。图片中第二个count我漏加distinct了。之后，用它与group by 组合就能达成目的了。 

```
select city,
       count(distinct positionId),
       count(distinct if(industryField like '%电子商务%',positionId,null)) 
from DataAnalyst
group by city
```

第一列数字是职位总数，第二列是电商领域的职位数，相除就是占比。记住，**count是不论0还是1都会纳入计数，所以第三个参数需要写成null**，代表不是电商的职位就排除在计算之外。 

找出各个城市，数据分析师岗位数量在500以上的城市有哪些，应该怎么计算？有两种方法，第一种，是使用having语句，它对聚合后的数据结果进行过滤。 

```
select city,count(distinct positionId) from DataAnalyst
group by city having count(distinct positionId) >= 500 
```

第二种，是利用嵌套子查询 

```
select * from(
    select city,count(distinct positionId) as counts from DataAnalyst
    group by city) as t1
where counts>=500
```

**时间**

```
select now()    # 获得当前的系统时间，精确到秒
select date(now())
# 它代表的是获得当前日期，week函数获得当前第几周，month函数获得当前第几个月。其余还包括，quarter，year，day，hour，minute。
select week(now(),0)
# 除了以上的日期表达，也可以使用dayofyear、weekofyear 的形式计算。
```

```
# 时间的加减
select date_add(date(now()) ,interval 1 day)
```

我们可以改变1为负数，达到减法的目的，也能更改day为week、year等，进行其他时间间隔的运算。如果是求两个时间的间隔，则是datediff(date1,date2)或者timediff(time1,time2)。 

**数据清洗类**

```
select left(salary,1) from DataAnalyst
```

MySQL支持left、right、mid等函数，和Excel一样。 

首先利用locate函数查找第一个k所在的位置。 

```
select locate("k",salary),salary from DataAnalyst
```

然后使用left函数截取薪水的下限。

```
select left(salary,locate("k",salary)-1),salary from DataAnalyst
```

为了获得薪水的上限，要用substr函数，或者mid，两者等价。

> substr（字符串，从哪里开始截，截取的长度）

再然后计算不同城市不同工作年限的平均薪资。 

```
select city,workYear,avg((bottomSalary+topSalary)/2) as avgSalary
from (select left(salary,locate("K",salary)-1) as bottomSalary,
             substr(salary,locate("-",salary)+1,length(salary)- locate("-",salary)-1) as topSalary,
             city,positionId,workYear
      from DataAnalyst
      where salary not like '%以上%') as t1
group by city,workYear
order by city,avgSalary 
```

一些雷区要注意：

①在不用聚合函数的时候，单独用group by，group by 子句中必须包含所有的列，否则会报错，但此时虽然成功执行了，group by在这里并没有发挥任何的作用，完全可以不用；若不用聚合函数，就是按照group by后面字段的顺序，把相同内容归纳在一起

③如果只有聚合函数，而没有group by，则聚合函数用于聚合整个结果集 (匹配WHERE子句的所有行)，相当于只分一组。

④where后面不能放聚合函数！无论是count还是sum。那么如何解决呢，使用HAVING关键字！例如：having
sum(amount) >100

⑤order by 后面是可以跟聚合函数的，即可以用聚合函数排序。

另外，除了Count(*)函数外，所有的聚合函数都忽略NULL值。

两个典型小问题的解决方法，看了很受启发。一是，最后排序时若要将某一类放在最前或最后，可以利用case when，巧妙的引用辅助列，帮助排序。例如：

①ORDER BY (case
when subject in ('Physics','Chemistry') then 1 else 0 end ), subject, winner

结果：科目为(‘Physics’,’Chemistry’)
的排在最后，其余科目按subject升序排列，

②ORDER BY (case
when subject in ('Physics','Chemistry') then 1 else 0 end ) desc, yr desc, winner

结果：将(‘Physics’,’Chemistry’)
排在最前；同一科目种类时，按年份从新到老；同一科目、同一年份时，按获奖者名字升序排列。

二是，一个经典问题：分组后取每组的前几条记录。这里看一个例子吧。

例：已知一个表， StudentGrade
(stuid--学号, subid--课程号, grade--成绩)。PRIMARY KEY
(stuid, subid)。

想要：查询每门课程的前2名成绩。

方法①：

select distinct * from
studentgrade as t1

where stuid in

(select top 2 stuid from
studentgrade as t2

where t1.subid=t2.subid

order by t2.grade desc) order by subid, grade
desc

思路：相同的表格自联结，第二个表格将相同学科的所有学生按成绩排序-倒序，选取前二。注意，mysql不支持select top n的语法！但是mysql可用limit来实现相关功能。

方法②：

select * from StudentGrade a

where (select count(1) from
studentGrade b

where b.subId=a.subId and b.grade
\>= a.grade) <=2

思路：第一个>=号，限制了查询条件是相同科目下成绩从大往小排，第二个<=号，表示筛选个数是2个（从1开始的）。

注意，这里大于等于、小于等于容易弄错，尤其是第二个。

方法③：

select * from StudentGrade a

where (select count(1) from
StudentGrade b

where b.subid=a.subid and
b.grade> a.grade) <=1

order by subId, grade desc

思路：这两张表思路相同：相同表格自联结，返回相同学科并且成绩大于a表的影响行数。这就是查询条件，再按 subId,grade 排序。