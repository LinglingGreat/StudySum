URL的一般格式为（带方括号[]的为可选项）：

protocol :// hostname[:port] / path / \[;parameters][?query]#fragment

URL由三部分组成： 

–第一部分是协议：http，https，ftp，file，ed2k…

–第二部分是存放资源的服务器的域名系统或IP地址（有时候要包含端口号，各种传输协议都有默认的端口号，如http的默认端口为80）。

–第三部分是资源的具体地址，如目录或文件名等。



下载猫的图片

```python
import urllib.request

response = urllib.request.urlopen("http://placekitten.com/g/200/300")
# 相当于下面两句
# req = urllib.request.Request("http://placekitten.com/g/200/300")
# response = urllib.request.urlopen(req)
cat_img = response.read()
# 其它函数：response.geturl(), response.info(), response.getcode()

with open('cat_200_300.jpg', 'wb') as f:
    f.write(cat_img)
```

## 隐藏

通过Request的headers参数修改

通过Request.add_header()方法修改

有道翻译

```python
# Network——header——各种参数以及fromdata
import urllib.request
import urllib.parse
import json
import time

while True:
    content = input('请输入待翻译的内容（输入"q!"退出程序）：')
    if content == 'q!':
        break
    
    url = "http://fanyi.youdao.com/translate?smartresult=dict&smartresult=rule&smartresult=ugc&sessionFrom=http://www.youdao.com/"

    '''
    head = {}
    head['User-Agent'] = 'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.65 Safari/537.36'
    '''

    data = {}
    data['type'] = 'AUTO'
    data['i'] = content
    data['doctype'] = 'json'
    data['xmlVersion'] = '1.6'
    data['keyfrom'] = 'fanyi.web'
    data['ue'] = 'UTF-8'
    data['typoResult'] = 'true'
    data = urllib.parse.urlencode(data).encode('utf-8')   # 编码

    req = urllib.request.Request(url, data)
    req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.65 Safari/537.36')

    response = urllib.request.urlopen(req)
    html = response.read().decode('utf-8')   # 解码成unicode形式

    target = json.loads(html)   # 字典
    target = target['translateResult'][0][0]['tgt']

    print(target)
    time.sleep(5)
```

代理   步骤：

1. 参数是一个字典 {‘类型’:‘代理ip:端口号’}

proxy_support = urllib.request.ProxyHandler({})

2. 定制、创建一个 opener

opener = urllib.request.build_opener(proxy_support)

   3a. 安装 opener，永久性

urllib.request.install_opener(opener)

   3b. 调用 opener，每次调用

opener.open(url)

```python
import urllib.request
import random

url = 'http://www.whatismyip.com.tw'

iplist = ['119.6.144.73:81', '183.203.208.166:8118', '111.1.32.28:81']

proxy_support = urllib.request.ProxyHandler({'http':random.choice(iplist)})

opener = urllib.request.build_opener(proxy_support)
opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.65 Safari/537.36')]

urllib.request.install_opener(opener)

response = urllib.request.urlopen(url)
html = response.read().decode('utf-8')

print(html)
```



## 图片下载实例

```python
import urllib.request
import os
import random


def url_open(url):
    req = urllib.request.Request(url)
    req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.65 Safari/537.36')

    proxies = ['119.6.144.70:81', '111.1.36.9:80', '203.144.144.162:8080']
    proxy = random.choice(proxies)

    proxy_support = urllib.request.ProxyHandler({'http':proxy})
    opener = urllib.request.build_opener(proxy_support)
    urllib.request.install_opener(opener)

    response = urllib.request.urlopen(url)
    html = response.read()

    return html


def get_page(url):
    html = url_open(url).decode('utf-8')

    a = html.find('current-comment-page') + 23
    b = html.find(']', a)   # 从a开始找[

    return html[a:b]


def find_imgs(url):
    html = url_open(url).decode('utf-8')
    img_addrs = []

    a = html.find('img src=')

    while a != -1:
        b = html.find('.jpg', a, a+255)  # 限制图片地址不超过255
        if b != -1:
            img_addrs.append(html[a+9:b+4])
        else:
            b = a + 9

        a = html.find('img src=', b)

    return img_addrs


def save_imgs(folder, img_addrs):
    for each in img_addrs:
        filename = each.split('/')[-1]
        with open(filename, 'wb') as f:
            img = url_open(each)
            f.write(img)


def download_mm(folder='OOXX', pages=10):
    os.mkdir(folder)   # 创建文件夹
    os.chdir(folder)   # 切换到文件夹里

    url = "http://jandan.net/ooxx/"
    page_num = int(get_page(url))

    for i in range(pages):
        page_num -= i
        page_url = url + 'page-' + str(page_num) + '#comments'
        img_addrs = find_imgs(page_url)
        save_imgs(folder, img_addrs)

if __name__ == '__main__':
    download_mm()
```



## 异常处理

处理异常的第一种写法 

```python
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
req = Request(someurl)
try:
    response = urlopen(req)
except HTTPError as e:
    print('The server couldn\'t fulfill the request.')
    print('Error code: ', e.code)
except URLError as e:
    print('We failed to reach a server.')
    print('Reason: ', e.reason)
else:
# everything is fine
```

处理异常的第二种写法 

```python
from urllib.request import Request, urlopen
from urllib.error import URLError
req = Request(someurl)
try:
    response = urlopen(req)
except URLError as e:
    if hasattr(e, 'reason'):
        print('We failed to reach a server.')
        print('Reason: ', e.reason)
    elif hasattr(e, 'code'):
        print('The server couldn\'t fulfill the request.')
        print('Error code: ', e.code)
else:
# everything is fine
```

##Scrapy

使用Scrapy抓取一个网站一共需要四个步骤：

– 创建一个Scrapy项目；

– 定义Item容器；

– 编写爬虫；

– 存储内容。

```
tutorial/
    scrapy.cfg
    tutorial/
        __init__.py
        items.py
        pipelines.py
        settings.py
        spiders/
            __init__.py
            ...
```

