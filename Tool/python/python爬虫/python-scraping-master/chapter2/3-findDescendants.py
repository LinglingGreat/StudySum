from urllib.request import urlopen
from bs4 import BeautifulSoup

html = urlopen("http://www.pythonscraping.com/pages/page3.html")
bsObj = BeautifulSoup(html, "html.parser")

# 打印giftList表格中所有产品的数据行。如果你用descendants()函数而不是children()函数，那么就会有二十几个标签打印出来，
# 包括img标签、span标签，以及每个td标签
for child in bsObj.find("table",{"id":"giftList"}).children:
    print(child)