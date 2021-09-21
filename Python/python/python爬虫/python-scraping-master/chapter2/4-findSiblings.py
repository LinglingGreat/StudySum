from urllib.request import urlopen
from bs4 import BeautifulSoup
html = urlopen("http://www.pythonscraping.com/pages/page3.html")
bsObj = BeautifulSoup(html, "html.parser")

# 打印产品列表里所有行的产品，第一行表格标题除外：因为对象不能把自己作为兄弟标签；这个函数只调用后面的兄弟标签
# 类似的函数还有previous_siblings, next_sibling, previous_sibling
for sibling in bsObj.find("table",{"id":"giftList"}).tr.next_siblings:
    print(sibling) 