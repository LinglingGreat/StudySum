## 用Python进行语言检测

最近正好碰到这个需求，总结一下用Python进行语言检测的方法。

1.用unicode编码检测

汉字、韩文、日文等都有对应的unicode字符集范围，只要用正则表达式匹配出来即可。

在判断的时候，往往需要去掉一些特殊字符，例如中英文标点符号。可以用下列方法去除：

```
# 方法一，自定义需要去掉的标点符号，注意这个字符串的首尾出现的[]不是标点符号'[]'，而是正则表达式中的中括号，表示定义匹配的字符范围
remove_nota = u'[’·°–!"#$%&\'()*+,-./:;<=>?@，。?★、…【】（）《》？“”‘’！[\\]^_`{|}~]+'
sentence = '测试。，[].?'
print(re.sub(remove_nota, '', sentence))

# 方法二，只能去掉英文标点符号
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
print(sentence.translate(remove_punctuation_map))
```

输出：

```
测试
测试。，
```



还可以把数字也去掉：

```
# 方法一
sentence = re.sub('[0-9]', '', sentence).strip()

# 方法二
remove_digits = str.maketrans('', '', string.digits)
sentence = sentence.translate(remove_digits)
```

然后就可以进行语言检测了。

这里的思路是匹配句子的相应语言字符，然后替换掉，如果替换后字符串为空，表示这个句子是纯正的该语言(即不掺杂其它语言)。也可以用正则表达式查询出句子中属于该语言的字符

```
s = "English Test"
re_words = re.compile(u"[a-zA-Z]")
res = re.findall(re_words, s)  # 查询出所有的匹配字符串
print(res)

res2 = re.sub('[a-zA-Z]', '', s).strip()
print(res2)    # 空字符串
if len(res2) <= 0:
    print("This is English")
```

输出：

```
['E', 'n', 'g', 'l', 'i', 's', 'h', 'T', 'e', 's', 't']

This is English
```

匹配英文用u"[a-zA-Z]"

中文用u"[\u4e00-\u9fa5]+"

韩文用u"[\uac00-\ud7ff]+"

日文用u"[\u30a0-\u30ff\u3040-\u309f]+" (包括平假名和片假名)



如果想只保留需要的内容，比如保留中英文及数字：

```
# 只保留中文、英文、数字（会去掉法语德语韩语日语等）
rule = re.compile(u"[^a-zA-Z0-9\u4e00-\u9fa5]")
sentence = rule.sub('', sentence)
```



完整代码：

```
import re
import string

remove_nota = u'[’·°–!"#$%&\'()*+,-./:;<=>?@，。?★、…【】（）《》？“”‘’！[\\]^_`{|}~]+'
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
def filter_str(sentence):
    sentence = re.sub(remove_nota, '', sentence)
    sentence = sentence.translate(remove_punctuation_map)
    return sentence.strip()
    
# 判断中日韩英
def judge_language(s):
    # s = unicode(s)   # python2需要将字符串转换为unicode编码，python3不需要
    s = filter_str(s)
    result = []
    s = re.sub('[0-9]', '', s).strip()
    # unicode english
    re_words = re.compile(u"[a-zA-Z]")
    res = re.findall(re_words, s)  # 查询出所有的匹配字符串
    res2 = re.sub('[a-zA-Z]', '', s).strip()
    if len(res) > 0:
        result.append('en')
    if len(res2) <= 0:
        return 'en'

    # unicode chinese
    re_words = re.compile(u"[\u4e00-\u9fa5]+")
    res = re.findall(re_words, s)  # 查询出所有的匹配字符串
    res2 = re.sub(u"[\u4e00-\u9fa5]+", '', s).strip()
    if len(res) > 0:
        result.append('zh')
    if len(res2) <= 0:
        return 'zh'

    # unicode korean
    re_words = re.compile(u"[\uac00-\ud7ff]+")
    res = re.findall(re_words, s)  # 查询出所有的匹配字符串
    res2 = re.sub(u"[\uac00-\ud7ff]+", '', s).strip()
    if len(res) > 0:
        result.append('ko')
    if len(res2) <= 0:
        return 'ko'

    # unicode japanese katakana and unicode japanese hiragana
    re_words = re.compile(u"[\u30a0-\u30ff\u3040-\u309f]+")
    res = re.findall(re_words, s)  # 查询出所有的匹配字符串
    res2 = re.sub(u"[\u30a0-\u30ff\u3040-\u309f]+", '', s).strip()
    if len(res) > 0:
        result.append('ja')
    if len(res2) <= 0:
        return 'ja'
    return ','.join(result)
```

这里的judge_language函数实现的功能是：针对一个字符串，返回其所属语种，如果存在多种语言，则返回多种语种(只能检测出中日英韩)

测试一下效果：

```
s1 = "汉语是世界上最优美的语言，正则表达式是一个很有用的工具"
s2 = "正規表現は非常に役に立つツールテキストを操作することです"
s3 = "あアいイうウえエおオ"
s4 = "정규 표현식은 매우 유용한 도구 텍스트를 조작하는 것입니다"
s5 = "Regular expression is a powerful tool for manipulating text."
s6 = "Regular expression 正则表达式 あアいイうウえエおオ 정규 표현식은"
print(judge_language(s1))
print(judge_language(s2))
print(judge_language(s3))
print(judge_language(s4))
print(judge_language(s5))
print(judge_language(s6))
```

输出：

```
zh
zh,ja
ja
ko
en
en,zh,ko,ja
```

因为s2中包括了汉字，所以输出结果中有zh。



2.用工具包检测

（1）langdetect

```
from langdetect import detect
from langdetect import detect_langs

s1 = "汉语是世界上最优美的语言，正则表达式是一个很有用的工具"
s2 = "正規表現は非常に役に立つツールテキストを操作することです"
s3 = "あアいイうウえエおオ"
s4 = "정규 표현식은 매우 유용한 도구 텍스트를 조작하는 것입니다"
s5 = "Regular expression is a powerful tool for manipulating text."
s6 = "Regular expression 正则表达式 あアいイうウえエおオ 정규 표현식은"

print(detect(s1))
print(detect(s2))
print(detect(s3))
print(detect(s4))
print(detect(s5))
print(detect(s6))     # detect()输出探测出的语言类型
print(detect_langs(s6))    # detect_langs()输出探测出的所有语言类型及其所占的比例
```

输出：

```
zh-cn
ja
ja
ko
en
ca   # 加泰隆语
[ca:0.7142837837746273, ja:0.2857136751343887]
```

emmm...最后一句话识别的不准



（2）langid

```
import langid

s1 = "汉语是世界上最优美的语言，正则表达式是一个很有用的工具"
s2 = "正規表現は非常に役に立つツールテキストを操作することです"
s3 = "あアいイうウえエおオ"
s4 = "정규 표현식은 매우 유용한 도구 텍스트를 조작하는 것입니다"
s5 = "Regular expression is a powerful tool for manipulating text."
s6 = "Regular expression 正则表达式 あアいイうウえエおオ 정규 표현식은"

print(langid.classify(s1))
print(langid.classify(s2))
print(langid.classify(s3))
print(langid.classify(s4))
print(langid.classify(s5))
print(langid.classify(s6))   # langid.classify(s6)输出探测出的语言类型及其confidence score，其confidence score计算方式方法见：https://jblevins.org/log/log-sum-exp
```

输出：

```
('zh', -370.64875650405884)
('ja', -668.9920794963837)
('ja', -213.35927987098694)
('ko', -494.80780935287476)
('en', -56.482327461242676)
('ja', -502.3459689617157)
```

两个包都把最后一句话识别成了英文，他们给出的结果都是ISO 639-1标准的语言代码。

再来看几个其他语言的例子：

```
s = "ру́сский язы́к"    # Russian
print(detect(s))
print(langid.classify(s))

s = "العَرَبِيَّة"    # Arabic
print(detect(s))
print(langid.classify(s))

s = "bonjour"    # French
print(detect(s))
print(langid.classify(s))
```

输出：

```
ru
('ru', -194.25553131103516)
ar
('ar', -72.63771915435791)
hr   # 克罗地亚语
('en', -22.992373943328857)
```

法语没判断出来。langdetect的判断结果依旧比较离谱...

没事可以多玩玩这两个包，O(∩_∩)O哈哈~



参考资料：

<https://blog.csdn.net/gatieme/article/details/43235791>

<https://blog.csdn.net/quiet_girl/article/details/79653037>





