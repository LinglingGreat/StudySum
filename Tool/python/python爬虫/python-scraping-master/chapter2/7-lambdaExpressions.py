from urllib.request import urlopen
from bs4 import BeautifulSoup
html = urlopen("http://www.pythonscraping.com/pages/page2.html")
bsObj = BeautifulSoup(html, "html.parser")
# tag.attrs可以获取标签对象的全部属性，myImgTag.attrs["src"]
# 获取有两个属性的标签
tags = bsObj.findAll(lambda tag: len(tag.attrs) == 2)
for tag in tags:
	print(tag)