# 【Task4】MapReduce+MapReduce执行过程
1. MR原理

hadoop streaming 原理
![图片](https://uploader.shimo.im/f/C3e59L3B4aMHQ5Gf.png!thumbnail)

作业思路
a. 准备数据
b. 写mr
c. 执行并验证答案

2. 使用Hadoop Streaming -python写出WordCount

Python版本
2.1 准备数据 
vim wc_input.txt
```
hello
word
hello
world
hello
python
java
python
```
```bash
hadoop fs -mkdir -p /user/dw/wc/input/
hadoop fs -mkdir -p /user/dw/wc/output/
hadoop fs -put wc_input.txt /user/dw/wc/input/
``` 
2.2 编写map/reduce/run.sh(注意修改/path/to)
mapper
```python
import sys


for line in sys.stdin:
	word = line.strip()
	print(word+'\t'+'1')

```

reducer.py

```python
import sys

cur_word = None
sum = 0
for line in sys.stdin:
	word,val = line.strip().split('\t')
	
	if cur_word==None:
		cur_word = word
	if cur_word!=word:
		print('%s\t%s'%(cur_word,sum)) 
		cur_word = word
		sum = 0
	sum+=int(val)
print('%s\t%s'%(cur_word,sum))	
```

run.sh
```bash
HADOOP_CMD="/path/to/hadoop"
STREAM_JAR_PATH="/path/to/hadoop-streaming-2.6.1.jar"

INPUT_FILE_PATH="/user/dw/wc/input/wc_input.txt"
OUTPUT_PATH="/user/dw/wc/output"

$HADOOP_CMD jar $STREAM_JAR_PATH \
    -input $INPUT_FILE_PATH \
    -output $OUTPUT_PATH \
    -mapper "python mapper.py" \
    -reducer "python reducer.py" \
    -file ./mapper.py \
    -file ./reducer.py
```
2.3. 设置输入输出路径
2.4. 执行，sh run.sh
2.5. check答案
```bash
hadoop fs -cat /user/dw/wc/output/part-00000
```

1 使用mr计算movielen中每个用户的平均评分。
1.1 主备数据
```bash
wget  wget http://files.grouplens.org/datasets/movielens/ml-100k.zip
unzip ml-100k.zip
cd ml-100k
mv u.data ml_input.txt
hadoop fs -mkdir -p /user/dw/avgscore/input
hadoop fs -mkdir -p /user/dw/avgscore/output
hadoop fs -put ml_input.txt /user/dw/avgscore/input/
```
1.2 编写map/reduce/run.sh
mapper.py
```python
import sys

for line in sys.stdin:
    item = line.strip().split('\t')
    print(item[1]+'\t'+item[2])

```
reduce.py
```python
import sys

item_score = {}

for line in sys.stdin:
    line = line.strip()
    item, score = line.split('\t')

    if item in item_score:
        item_score[item].append(int(score))
    else:
        item_score[item] = []
        item_score[item].append(int(score))

for item in item_score.keys():
    ave_score = sum(item_score[item])*1.0 / len(item_score[item])
    print '%s\t%s'% (item, ave_score)

```
run.sh 略，参考以上run.sh
1.3 check结果的正确性，自行编写python代码验证

2. 实现merge功能
2.1 准备数据ml-100k
u.data
u.item
自行使用map先将二者分隔符统一，例如全部使用＃作为分隔符

2.2 编写map/reduce/run
mapper.py
```python
import sys

for line in sys.stdin:
    line = line.strip()
    line = line.split(" ")

    user = "-1"
    item = "-1"
    score = "-1"
    item_name = "-1"
    item_time = "-1"



    if len(line) ==4:
        user = line[0]
        item = line[1]
        score = line[2]
    else:
        item = line[0]
        item_name = line[1]
        item_score = line[2]


    print '%s\t%s\t%s\t%s\t%s' % (user, item, score, item_name, item_time)

```
reducer.py
```python
import sys

item_dict ={}
ui_dict={}
for line in sys.stdin:
    line = line.strip()
    user,item,score,item_name,item_time = line.split('\t')

    if user == "-1":
        item_dict[item] = [item_name,item_time]
    else:
        ui_dict[user] = [item,score]

for user in ui_dict.keys():
    item_name = item_dict[ui_dict[user][0]]
    item_time = item_dict[ui_dict[user][1]]
    item = customer_dict[id][1]
    score = customer_dict[id][2]

    print '%s\t%s\t%s\t%s'% (user, item, score, item_name, item_time)

```

3. 使用mr实现去重任务。
3.1 准备数据
```
1
2
3
4
5
6
1
2
3
3
```
其他步骤同上
3.2 map/reduce/run

mapper.py
```python
import sys

for line in sys.stdin:
    print(line+'\t'+' ')
```
reducer.py
```python
import sys

last_key = None
for line in sys.stdin:
    this_key = line.split('\t')[0].strip()
    if this_key == last_key:
        pass
    else:
        if last_key:
            print(last_key)
        last_key = this_key
print(this_key)
```

4. 使用mr实现排序。
4.1 数据，使用上述计算的电影平均分作为输入

4.2 编写map/reduce/run
```python
import sys
 
for line in sys.stdin:
	line = line.strip()
	print('{0}'.format(line))
```

```python
import sys
 
for line in sys.stdin:
	line = line.strip()
	print("{0}".format(line))
```

```bash
HADOOP_CMD="/path/to/hadoop"
STREAM_JAR_PATH="/path/to/hadoop-streaming-2.6.1.jar"

INPUT_FILE_PATH="/user/dw/wc/input/wc_input.txt"
OUTPUT_PATH="/user/dw/wc/output"

$HADOOP_CMD jar $STREAM_JAR_PATH \
    -D stream.map.output.field.separator='\t' \
    -D stream.num.map.output.key.fields=2 \
    -D mapreduce.job.output.key.comparator.class=org.apache.hadoop.mapreduce.lib.partition.KeyFieldBasedComparator \
    -D mapreduce.partition.keycomparator.options=-k2,2nr \ 
    -input $INPUT_FILE_PATH \
    -output $OUTPUT_PATH \
    -mapper "python mapper.py" \
    -reducer "python reducer.py" \
    -file ./mapper.py \
    -file ./reducer.py
```

5. 使用mapreduce实现倒排索引。

mapper.py
```python
import os
import sys

docname = os.environ["map_input_file"]

for line in sys.stdin:
    line = line.strip().split(' ')
    for word in line:
        print('{1}\t{2}'.format(line,docname)

```
reducer.py
```python
import sys

word_doc_dict ={}
for line in sys.stdin:
    line = line.strip()
    word,docname = line.split('\t')

    if word in word_doc_dict:
        word_doc_dict[word].append(docname)
    else:
        word_doc_dict[word] = []
        word_doc_dict[word].append(int(scdocname))

for word in word_doc_dict.keys():
    print('{1}\t{2}'.format(word,','.join(word_doc_dict[word]))

```

6. 使用mapreduce计算Jaccard相似度(参考spark实现)。
7. 使用mapreduce实现PageRank(选做，答案略，自行查阅)。

[Python3调用Hadoop的API](https://www.cnblogs.com/sss4/p/10443497.html)