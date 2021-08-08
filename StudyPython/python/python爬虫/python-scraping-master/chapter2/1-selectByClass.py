from urllib.request import urlopen
from bs4 import BeautifulSoup

html = urlopen("http://www.pythonscraping.com/pages/warandpeace.html")
bsObj = BeautifulSoup(html, "html.parser")
# 抽取只包含在<span class="green"></span>标签里的文字
nameList = bsObj.findAll("span", {"class":"green"})
# get_text()会把你正在处理的HTML文档中所有的标签都清除，然后返回一个只包含文字的字符串。
for name in nameList:
    print(name.get_text())