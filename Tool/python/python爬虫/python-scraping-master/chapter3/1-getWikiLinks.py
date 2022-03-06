from urllib.request import urlopen
from bs4 import BeautifulSoup
import datetime
import random
import re

random.seed(datetime.datetime.now())
def getLinks(articleUrl):
    html = urlopen("http://en.wikipedia.org"+articleUrl)
    bsObj = BeautifulSoup(html, "html.parser")
    # 指向词条页面（不是指向其他内容页面）的链接的共同点：
    # 1.他们都在id是bodyContent的div标签里 2.URL链接不包含冒号 3.URL链接都以/wiki/开头
    return bsObj.find("div", {"id":"bodyContent"}).findAll("a", href=re.compile("^(/wiki/)((?!:).)*$"))
# 以某个起始词条为参数调用getLinks，再从返回的URL列表里随机选择一个词条链接，再调用getLinks，
# 直到我们主动停止，或者再新的页面上没有词条链接了，程序才停止运行
links = getLinks("/wiki/Kevin_Bacon")
while len(links) > 0:
    newArticle = links[random.randint(0, len(links)-1)].attrs["href"]
    print(newArticle)
    links = getLinks(newArticle)