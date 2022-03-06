

```python
re.search(r' ',str)
```

编译正则表达式

```python
pattern = re.compile()

pattern.search()
```



```python
import urllib.request
import re

def open_url(url):
    req = urllib.request.Request(url)
    req.add_header('User-Agent', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36')
    page = urllib.request.urlopen(req)
    html = page.read().decode('utf-8')

    return html

def get_img(html):
    p = r'<img class="BDE_Image" src="([^"]+\.jpg)"'
    # 只输出src=后面的内容，因为这里的()被python看作元组
    imglist = re.findall(p, html)
    '''
    for each in imglist:
        print(each)
    '''
    for each in imglist:
        filename = each.split("/")[-1]
        urllib.request.urlretrieve(each, filename, None)

    
if __name__ == '__main__':    
    url = "http://tieba.baidu.com/p/3563409202"
    get_img(open_url(url))

```





```python
import urllib.request
import re

def open_url(url):
    req = urllib.request.Request(url)
    req.add_header('User-Agent', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36')
    page = urllib.request.urlopen(req)
    html = page.read().decode('utf-8')

    return html

def get_img(html):
    p = r'(?:(?:[0,1]?\d?\d|2[0-4]\d|25[0-5])\.){3}(?:[0,1]?\d?\d|2[0-4]\d|25[0-5])'
    # 加上?:
    iplist = re.findall(p, html)

    for each in iplist:
        print(each)

    
if __name__ == '__main__':    
    url = "http://cn-proxy.com"
    get_img(open_url(url))

```





