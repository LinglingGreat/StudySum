from urllib.request import urlopen
from bs4 import BeautifulSoup
import re
import string
from collections import OrderedDict

def cleanInput(input):
    # 移除转义字符\n
    input = re.sub('\n+', " ", input)
    # 剔除引用标记（如[1])
    input = re.sub('\[[0-9]*\]', "", input)
    input = re.sub(' +', " ", input)
    input = bytes(input, "UTF-8")
    # 过滤掉unicode字符
    input = input.decode("ascii", "ignore")
    cleanInput = []
    input = input.split(' ')
    for item in input:
        # 单词两端的任何标点符号都会被去掉，但带连字符的单词（连字符在单词内部）仍然会保留
        item = item.strip(string.punctuation)
        if len(item) > 1 or (item.lower() == 'a' or item.lower() == 'i'):
            cleanInput.append(item)
    return cleanInput

def getNgrams(input, n):
    input = cleanInput(input)
    output = dict()
    for i in range(len(input)-n+1):
        newNGram = " ".join(input[i:i+n])
        if newNGram in output:
            output[newNGram] += 1
        else:
            output[newNGram] = 1
    return output

html = urlopen("http://en.wikipedia.org/wiki/Python_(programming_language)")
bsObj = BeautifulSoup(html, "html.parser")
content = bsObj.find("div", {"id":"mw-content-text"}).get_text()
#ngrams = getNgrams(content, 2)
#print(ngrams)
#print("2-grams count is: "+str(len(ngrams)))

ngrams = getNgrams(content, 2)
ngrams = OrderedDict(sorted(ngrams.items(), key=lambda t: t[1], reverse=True))
print(ngrams)