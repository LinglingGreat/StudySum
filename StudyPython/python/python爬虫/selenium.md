中文文档http://selenium-python-zh.readthedocs.io/en/latest/index.html

# 安装

pip install selenium

## 1.下载浏览器驱动器

以chrome为例，下载地址为

https://sites.google.com/a/chromium.org/chromedriver/downloads

http://npm.taobao.org/mirrors/chromedriver/   淘宝映像地址

http://blog.csdn.net/huilan_same/article/details/51896672  chromedriver与chrome版本映射表

可以将chromedriver配置在环境变量中，或者放在某个目录下

## 2. 开始使用

```python
from selenium import webdriver

# 如果已经配置环境
browser = webdriver.Chrome()
# 如果没有配置环境
browser = webdriver.Chrome('E:\chromedriver.exe')
browser.get('http://www.baidu.com/')
```

可以增加一些偏好设置

```python
from selenium import webdriver

options = webdriver.ChromeOptions()
# 保存文件的路径
prefs = {'download.default_directory': 'E:\\'}
options.add_experimental_option('prefs', prefs)
browser = webdriver.Chrome(executable_path="\chromedriver.exe", chrome_options=options)
browser.get('http://www.baidu.com/')
```

## 查找元素（定位方法）

一、十八种定位方法

前八种是经常会用到的

1.id定位：find_element_by_id(self, id_)，如browser.find_element_by_id('kw')
2.name定位：find_element_by_name(self, name)
3.class定位：find_element_by_class_name(self, name)
4.tag定位：find_element_by_tag_name(self, name)
5.link定位：find_element_by_link_text(self, link_text)
6.partial_link定位find_element_by_partial_link_text(self, link_text)
7.xpath定位：find_element_by_xpath(self, xpath)
8.css定位：find_element_by_css_selector(self, css_selector）
这八种是复数形式
9.id复数定位find_elements_by_id(self, id_)
10.name复数定位find_elements_by_name(self, name)
11.class复数定位find_elements_by_class_name(self, name)
12.tag复数定位find_elements_by_tag_name(self, name)
13.link复数定位find_elements_by_link_text(self, text)
14.partial_link复数定位find_elements_by_partial_link_text(self, link_text)
15.xpath复数定位find_elements_by_xpath(self, xpath)
16.css复数定位find_elements_by_css_selector(self, css_selector
这两种就是快失传了的
find_element(self, by='id', value=None)
find_elements(self, by='id', value=None)



举例：<input type="text" name="passwd" id="passwd-id" />

```python
element = browser.find_element_by_id("passwd-id")
element = browser.find_element_by_name("passwd")
element = browser.find_element_by_tag_name("input")
element = browser.find_element_by_xpath("//input[@id='passwd-id']")
# 多个条件用and
element = browser.find_element_by_xpath("//input[@id='passwd-id' and @name='passwd]")
```

再比如：

```python
browser.find_element_by_xpath("//select[@class='ui-datepicker-year']/option[@value='" + str(year) + "']").click()  # 多个标签嵌套查找
itemmake_val = browser.find_element_by_class_name('ui-datepicker-month')
allitemmake = itemmake_val.find_elements_by_tag_name('option')
allmonth = [t.get_attribute("value") for t in allitemmake] # 获取value值

browser.find_elements_by_xpath("//td[contains(@class, ' has-data')]//a") # 部分匹配查找
browser.find_element_by_xpath("//a[text()="+str(day)+"]") # 根据标签的文本查找
```

以下内容参考http://blog.csdn.net/eastmount/article/details/48108259

举例：

```html
<html>  
 <body>  
  <form id="loginForm">  
   <input name="username" type="text" />  
   <input name="password" type="password" />  
   <input name="continue" type="submit" value="Login" />  
   <input name="continue" type="button" value="Clear" />  
  </form>  
</body>  
<html>  
```

定位username元素的方法如下：

```python
username = driver.find_element_by_xpath("//form[input/@name='username']")  
username = driver.find_element_by_xpath("//form[@id='loginForm']/input[1]")  
username = driver.find_element_by_xpath("//input[@name='username']")  
```

## 操作元素方法

通常所有的操作与页面交互都将通过WebElement接口，常见的操作元素方法如下：

- clear 清除元素的内容
- send_keys 模拟按键输入
- click 点击元素
- submit 提交表单

举例自动访问FireFox浏览器自动登录163邮箱。

```python
from selenium import webdriver    
from selenium.webdriver.common.keys import Keys    
import time  
  
# Login 163 email  
driver = webdriver.Firefox()    
driver.get("http://mail.163.com/")  
  
elem_user = driver.find_element_by_name("username")  
elem_user.clear  
elem_user.send_keys("15201615157")    
elem_pwd = driver.find_element_by_name("password")  
elem_pwd.clear  
elem_pwd.send_keys("******")    
elem_pwd.send_keys(Keys.RETURN)  
#driver.find_element_by_id("loginBtn").click()  
#driver.find_element_by_id("loginBtn").submit()  
time.sleep(5)    
assert "baidu" in driver.title    
driver.close()    
driver.quit()    
```

首先通过name定位用户名和密码，再调用方法clear()清除输入框默认内容，如“请输入密码”等提示，通过send_keys("**")输入正确的用户名和密码，最后通过click()点击登录按钮或send_keys(Keys.RETURN)相当于回车登录，submit()提交表单。
​        PS：如果需要输入中文，防止编码错误使用send_keys(u"中文用户名")。

## WebElement接口获取值

通过WebElement接口可以获取常用的值，这些值同样非常重要。

- size 获取元素的尺寸
- text 获取元素的文本
- get_attribute(name) 获取属性值
- location 获取元素坐标，先找到要获取的元素，再调用该方法
- page_source 返回页面源码
- driver.title 返回页面标题
- current_url 获取当前页面的URL
- is_displayed() 设置该元素是否可见
- is_enabled() 判断元素是否被使用
- is_selected() 判断元素是否被选中
- tag_name 返回元素的tagName

​        举例代码如下：

```python
from selenium import webdriver    
from selenium.webdriver.common.keys import Keys    
import time  
  
driver = webdriver.PhantomJS(executable_path="G:\phantomjs-1.9.1-windows\phantomjs.exe")     
driver.get("http://www.baidu.com/")  
  
size = driver.find_element_by_name("wd").size  
print size  
#尺寸: {'width': 500, 'height': 22}  
  
news = driver.find_element_by_xpath("//div[@id='u1']/a[1]").text  
print news  
#文本: 新闻  
  
href = driver.find_element_by_xpath("//div[@id='u1']/a[2]").get_attribute('href')  
name = driver.find_element_by_xpath("//div[@id='u1']/a[2]").get_attribute('name')  
print href,name  
#属性值: http://www.hao123.com/ tj_trhao123  
  
location = driver.find_element_by_xpath("//div[@id='u1']/a[3]").location  
print location  
#坐标: {'y': 19, 'x': 498}  
  
print driver.current_url  
#当前链接: https://www.baidu.com/  
print driver.title  
#标题: 百度一下， 你就知道  
  
result = location = driver.find_element_by_id("su").is_displayed()  
print result  
#是否可见: True
```

## 鼠标点击事件

在现实的自动化测试中关于鼠标的操作不仅仅是click()单击操作，还有很多包含在ActionChains类中的操作。如下：

- context_click(elem) 右击鼠标点击元素elem，另存为等行为
- double_click(elem) 双击鼠标点击元素elem，地图web可实现放大功能
- drag_and_drop(source,target) 拖动鼠标，源元素按下左键移动至目标元素释放
- move_to_element(elem) 鼠标移动到一个元素上
- click_and_hold(elem) 按下鼠标左键在一个元素上
- perform() 在通过调用该函数执行ActionChains中存储行为

​        举例如下图所示，获取通过鼠标右键另存为百度图片logo。代码：

```
import time  
from selenium import webdriver  
from selenium.webdriver.common.keys import Keys  
from selenium.webdriver.common.action_chains import ActionChains  
  
driver = webdriver.Firefox()  
driver.get("http://www.baidu.com")  
  
#鼠标移动至图片上 右键保存图片  
elem_pic = driver.find_element_by_xpath("//div[@id='lg']/img")  
print elem_pic.get_attribute("src")  
action = ActionChains(driver).move_to_element(elem_pic)  
action.context_click(elem_pic)  
  
#重点:当右键鼠标点击键盘光标向下则移动至右键菜单第一个选项  
action.send_keys(Keys.ARROW_DOWN)  
time.sleep(3)  
action.send_keys('v') #另存为  
action.perform()  
  
#获取另存为对话框(失败)  
alert.switch_to_alert()  
alert.accept()  
```

## 保存文件

保存文件的时候不能更改文件名，可以用下列方法在每次保存后修改最新保存的文件的文件名

```python
import os

filename = max([filepath + '\\' + f for f in os.listdir(filepath)], key=os.path.getctime)
os.rename(os.path.join(filepath, filename), os.path.join(filepath, newfilename))
```

## 页面等待

隐式等待是等待特定的时间，显式等待是指定某一条件直到这个条件成立时继续执行。

### 显式等待

显式等待指定某个条件，然后设置最长等待时间。如果在这个时间还没有找到元素，那么便会抛出异常了。

```python
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
 
driver = webdriver.Chrome()
driver.get("http://somedomain/url_that_delays_loading")
try:
    element = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "myDynamicElement"))
    )
finally:
    driver.quit()
```

程序默认会 500ms 调用一次来查看元素是否已经生成，如果本来元素就是存在的，那么会立即返回。

下面是一些内置的等待条件，你可以直接调用这些条件，而不用自己写某些等待条件了。

> - title_is
> - title_contains
> - presence_of_element_located
> - visibility_of_element_located
> - visibility_of
> - presence_of_all_elements_located
> - text_to_be_present_in_element
> - text_to_be_present_in_element_value
> - frame_to_be_available_and_switch_to_it
> - invisibility_of_element_located
> - element_to_be_clickable – it is Displayed and Enabled.
> - staleness_of
> - element_to_be_selected
> - element_located_to_be_selected
> - element_selection_state_to_be
> - element_located_selection_state_to_be
> - alert_is_present

```python
from selenium.webdriver.support import expected_conditions as EC
 
wait = WebDriverWait(driver, 10)
element = wait.until(EC.element_to_be_clickable((By.ID,'someid')))
```

### 隐式等待

隐式等待比较简单，就是简单地设置一个等待时间，单位为秒。

```python
from selenium import webdriver
 
driver = webdriver.Chrome()
driver.implicitly_wait(10) # seconds
driver.get("http://somedomain/url_that_delays_loading")
myDynamicElement = driver.find_element_by_id("myDynamicElement")
```

## 键盘操作

在webdriver的Keys类中提供了键盘所有的按键操作，当然也包括一些常见的组合键操作如Ctrl+A(全选)、Ctrl+C(复制)、Ctrl+V(粘贴)。更多键参考官方文档对应的编码。

- send_keys(Keys.ENTER) 按下回车键
- send_keys(Keys.TAB) 按下Tab制表键
- send_keys(Keys.SPACE) 按下空格键space
- send_keys(Kyes.ESCAPE) 按下回退键Esc
- send_keys(Keys.BACK_SPACE) 按下删除键BackSpace
- send_keys(Keys.SHIFT) 按下shift键
- send_keys(Keys.CONTROL) 按下Ctrl键
- send_keys(Keys.ARROW_DOWN) 按下鼠标光标向下按键
- send_keys(Keys.CONTROL,'a') 组合键全选Ctrl+A
- send_keys(Keys.CONTROL,'c') 组合键复制Ctrl+C
- send_keys(Keys.CONTROL,'x') 组合键剪切Ctrl+X
- send_keys(Keys.CONTROL,'v') 组合键粘贴Ctrl+V

```python
#coding=utf-8  
import time  
from selenium import webdriver  
from selenium.webdriver.common.keys import Keys  
  
driver = webdriver.Firefox()  
driver.get("http://www.baidu.com")  
  
#输入框输入内容  
elem = driver.find_element_by_id("kw")  
elem.send_keys("Eastmount CSDN")  
time.sleep(3)  
  
#删除一个字符CSDN 回退键  
elem.send_keys(Keys.BACK_SPACE)  
elem.send_keys(Keys.BACK_SPACE)  
elem.send_keys(Keys.BACK_SPACE)  
elem.send_keys(Keys.BACK_SPACE)  
time.sleep(3)  
  
#输入空格+"博客"  
elem.send_keys(Keys.SPACE)  
elem.send_keys(u"博客")  
time.sleep(3)  
  
#ctrl+a 全选输入框内容  
elem.send_keys(Keys.CONTROL,'a')  
time.sleep(3)  
  
#ctrl+x 剪切输入框内容  
elem.send_keys(Keys.CONTROL,'x')  
time.sleep(3)  
  
#输入框重新输入搜索  
elem.send_keys(Keys.CONTROL,'v')  
time.sleep(3)  
  
#通过回车键替代点击操作  
driver.find_element_by_id("su").send_keys(Keys.ENTER)  
time.sleep(3)  
  
driver.quit()
```

## 常见错误解决

### 定位不到元素

报selenium.common.exceptions.NoSuchElementException错误

一般是Frame/Iframe原因定位不到元素

这个是最常见的原因，首先要理解下frame的实质，frame中实际上是嵌入了另一个页面，而webdriver每次只能在一个页面识别，因此需要先定位到相应的frame，对那个页面里的元素进行定位。

解决方案：

如果iframe有name或id的话，直接使用switch_to_frame("name值")或switch_to_frame("id值")。

```python
Chrome_login = webdriver.Chrome()
Chrome_login.get(url)
iframe = Chrome_login.find_element_by_xpath("//iframe[contains(@src,'/publicweb/quotesdata/memberDealPosiQuotes.html')]")
Chrome_login.switch_to_frame(iframe)
```

参考 https://www.cnblogs.com/yufeihlf/p/5689042.html

