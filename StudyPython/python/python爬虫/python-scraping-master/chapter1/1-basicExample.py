from urllib.request import urlopen
#Retrieve HTML string from the URL
'''urlopen用来打开并读取一个从网络获取的远程对象
可以轻松读取HTML文件、图像文件，或其他任何文件流'''
# 输出在域名http://www.pythonscraping.com的服务器上<网络应用根地址>/exercises文件夹里的HTML文件exercise1.html的源代码
html = urlopen("http://www.pythonscraping.com/exercises/exercise1.html")
print(html.read())
