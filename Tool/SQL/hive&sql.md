# 数据库

## hive常见问题

科学计数法问题的解决：cast(cast(poiid as double) as bigint)

逗号=inner join=join，不写条件就是笛卡尔积

left join=left outer join

right join=right outer join

full join=full outer join包含左右表的所有行



[hive lateral view 与 explode详解](https://blog.csdn.net/bitcarmanlee/article/details/51926530)

## hive字符串用法

[https://zhuanlan.zhihu.com/p/82601425](https://zhuanlan.zhihu.com/p/82601425)

[hive中split(),explode()和lateral row](https://blog.csdn.net/yahahassr/article/details/97911676)

字符串类型的：

```SQL
select productid, split(depart,':')[0], split(depart,':')[1],district 
from xxx demo
lateral view explode(split(departcityprices,',')) demo as depart
lateral view explode(split(districtids,',')) demo as district
where businesstype in (6, 7);
```


`array<struct>`类型的

### 行列转换(explode array)

```sql
with
loc as
(  select explode(array('en-AU','en-GB','en-HK','en-MY','en-SG','en-US','ja-JP','ko-KR','zh-HK','zh-TW')) as locale
),
poi a
(  select id as officialpoiid, poitype, poitypecode,coverimageid,sourcetype,districtid,publishstatus
 ,districtidpath,threecode,rating,locale
    , case when locale='en-AU' then name_en_au when locale='en-GB' then name_en_gb
      when locale='en-HK' then name_en_hk when locale='en-MY' then name_en_my
      when locale='en-SG' then name_en_sg when locale='en-US' then name_en_us
      when locale='ja-JP' then name_ja_jp when locale='ko-KR' then name_ko_kr
      when locale='zh-HK' then name_zh_hk when locale='zh-TW' then name_zh_tw
      else null end as name
 , case when locale='en-AU' then reviewcount_en_au when locale='en-GB' then reviewcount_en_gb
      when locale='en-HK' then reviewcount_en_hk when locale='en-MY' then reviewcount_en_my
      when locale='en-SG' then reviewcount_en_sg when locale='en-US' then reviewcount_en_us
      when locale='ja-JP' then reviewcount_ja_jp when locale='ko-KR' then reviewcount_ko_kr
      when locale='zh-HK' then reviewcount_zh_hk when locale='zh-TW' then reviewcount_zh_tw
      else null end as reviewcount
 , case when locale='en-AU' then photocount_en_au when locale='en-GB' then photocount_en_gb
      when locale='en-HK' then photocount_en_hk when locale='en-MY' then photocount_en_my
      when locale='en-SG' then photocount_en_sg when locale='en-US' then photocount_en_us
      when locale='ja-JP' then photocount_ja_jp when locale='ko-KR' then photocount_ko_kr
      when locale='zh-HK' then photocount_zh_hk when locale='zh-TW' then photocount_zh_tw
      else null end as photocount
  , case when locale='en-AU' then score_en_au when locale='en-GB' then score_en_gb
      when locale='en-HK' then score_en_hk when locale='en-MY' then score_en_my
      when locale='en-SG' then score_en_sg when locale='en-US' then score_en_us
      when locale='ja-JP' then score_ja_jp when locale='ko-KR' then score_ko_kr
      when locale='zh-HK' then score_zh_hk when locale='zh-TW' then score_zh_tw
      else null end as score
 from xxx p,loc
)
```




### json字符串

```sql
hive> select explode(json_array('[{"website":"www.baidu.com","name":"百度"},{"website":"google.com"name":"谷歌"}]'));
OK
{"website":"www.baidu.com","name":"百度"}
{"website":"google.com","name":"谷歌"}
Time taken: 10.427 seconds, Fetched: 2 row(s)
```


```sql
hive> select json_tuple(json, 'website', 'name') from (SELECT explode(json_array('[{"website":"www.baidu.com","name":"百度"},{"website":"google.com","name":"谷歌"}]')) as json) test;
OK
www.baidu.com   百度
google.com      谷歌
Time taken: 0.265 seconds, Fetched: 2 row(s)

select json_tuple(json, 'website') as website from 
(SELECT explode(json_array('[{"website":"www.baidu.com","name":"百度"},{"website":"google.com","name":"谷歌"}]')) as json) test;


```


### 正则表达式

`regexp`

```sql
-- 不含中文的
select * from table where name not regexp '[\\u4E00-\\u9FFF]+'
```




[https://www.cnblogs.com/yfb918/p/10644262.html](https://www.cnblogs.com/yfb918/p/10644262.html)

## hive变量

### hivevar

`set hivevar:poitype=(2,3,66);`，用法：`poitype in ${poitype}`

`set hivevar:initial_date=2020-04-28;`，用法：`where d='${initial_date}'`

`set hivevar:score_base=2.5;`，用法：`'${score_base_cnt}'`

### hiveconf

`set vers=(select MAX(version) from dw_youdb.ta_sync_poi WHERE version IS NOT NULL);`

用法：`where version=${hiveconf:vers}`

## hive归一化

```sql
(lncommenttotalscore-min(lncommenttotalscore) over ()) / 
(max(lncommenttotalscore) over ()-min(lncommenttotalscore) over ()) as commentnorm
```


## mysql常用命令

`mysql -u root -p`

## mysql索引

应尽量避免在 where 子句中使用 or 来连接条件，否则将导致引擎放弃使用索引而进行全表扫描，如： select id from t where num=10 or num=20 可以这样查询： select id from t where num=10 union all select id from t where num=20

like keyword%    索引有效，其它的like语句索引无效。如果是前缀like，可以考虑reverse。

in会走索引



[MySQL百万级数据查询优化](https://juejin.cn/post/6854573209485770765)

# [Mysql索引（一篇就够le）](https://www.cnblogs.com/zsql/p/13808417.html)

[多个单列索引和联合索引的区别详解](https://blog.csdn.net/Abysscarry/article/details/80792876)

[sql like与索引（后模糊匹配才能让索引有效）](https://blog.csdn.net/lan12334321234/article/details/70048833)

## mysql常见问题

mysql卸载：[https://zhuanlan.zhihu.com/p/68190605](https://zhuanlan.zhihu.com/p/68190605)

安装：[https://zhuanlan.zhihu.com/p/37152572](https://zhuanlan.zhihu.com/p/37152572)

3306端口被占用：[https://blog.csdn.net/qq_28325423/article/details/80549018](https://blog.csdn.net/qq_28325423/article/details/80549018)

my.ini文件在路径C:\ProgramData\MySQL\MySQL Server 8.0下。修改datadir路径，将原路径下的文件拷贝到新路径，注意my.ini文件的编码方式是ANSI，自动保存会更改编码方式，导致mysql无法启动。

MYSQL导出数据：[https://blog.csdn.net/fdipzone/article/details/78634992](https://blog.csdn.net/fdipzone/article/details/78634992)

远程访问：[https://blog.csdn.net/sgrrmswtvt/article/details/82344183](https://blog.csdn.net/sgrrmswtvt/article/details/82344183)

python3 mysql错误 pymysql.err.OperationalError: (2013, 'Lost connection to MySQL server during query'）：[https://blog.csdn.net/whatday/article/details/104098336](https://blog.csdn.net/whatday/article/details/104098336)

MySQL 各种超时参数的含义：[https://www.cnblogs.com/xiaoboluo768/p/6222862.html](https://www.cnblogs.com/xiaoboluo768/p/6222862.html)

MySQL索引：[https://blog.csdn.net/u014745069/article/details/80466917](https://blog.csdn.net/u014745069/article/details/80466917)

## [Mysql死锁如何排查：insert on duplicate死锁一次排查分析过程](https://www.cnblogs.com/jay-huaxiao/p/11456921.html)

