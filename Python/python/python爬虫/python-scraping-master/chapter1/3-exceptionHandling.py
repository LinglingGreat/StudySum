from urllib.request import urlopen
from urllib.error import HTTPError
from bs4 import BeautifulSoup
import sys


def getTitle(url):
    try:
        '''这里可能会出现两种异常：
        网页在服务器上不存在（或者获取页面的时候出现错误），返回HTTP错误
        服务器不存在（链接打不开或者是URL链接写错了），返回一个None对象'''
        html = urlopen(url)
    except HTTPError as e:
        print(e)
        # 返回空值，中断程序，或者执行另一个方案
        return None
    try:
        '''如果想要调用的标签不存在，BeautifulSoup就会返回None对象。不过，如果再调用这个None对象下面的子标签，
        就会发生AttributeError错误。'''
        bsObj = BeautifulSoup(html, "html.parser")
        title = bsObj.body.h1
    except AttributeError as e:
        return None
    return title

title = getTitle("http://www.pythonscraping.com/exercises/exercise1.html")
if title == None:
    print("Title could not be found")
else:
    print(title)
    
    