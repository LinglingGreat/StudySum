# 图形用户界面
Graphical User Interface, GUI
Tkinter-Python标准GUI
Graphics-基于Tkinter扩展
Turtle-python内置的图形库
## graphics库
原点在左上角，x轴从坐到右是递增，y轴从上到下是递增
```
from graphics import *
circ = Circle(Point(100, 100), 30)
# 在(100,100)坐标处建立圆
win = GraphWin()
circ.draw(win)
```
两个不同的变量可能表示同一个对象

编写卡通脸的程序，两个眼睛间隔40像素长度
```
win = GraphWin()
leftEye = Circle(Point(80, 80), 5)
leftEye.setFill("yellow")
leftEye.setOutline("red")
rightEye = leftEye
rightEye.move(40, 0)
leftEye.draw(win)
rightEye.draw(win)
```
解决方法  
为左右眼分别创建两个不同的圆对象
```
from graphics import *
win = GraphWin()
face = Circle(Point(100, 95), 50)
leftEye = Circle(Point(80, 80), 5)
leftEye.setFill("yellow")
leftEye.setOutline("red")
rightEye = Circle(Point(120, 80), 5)
rightEye.setFill("yellow")
rightEye.setOutline("red")
mouth = Line(Point(80, 110), Point(120, 110))
face.draw(win)
mouth.draw(win)
leftEye.draw(win)
rightEye.draw(win)
```
### 交互式图形用户接口
连续点击10次鼠标，返回其坐标值
```
from graphics import *
def main():
    win = GraphWin("Click Me!")
    for i in range(10):
        p = win.getMouse()
        print("You clicked at:", p.getX(), p.getY())
if __name__ == '__main__':
    main()
```
在窗口中点击5个点来画一个五边形
```
from graphics import *
win = GraphWin("Draw a polygon",300,300)
win.setCoords(0.0, 0.0, 300.0, 300.0)
message = Text(Point(150, 20), "Click on five points")
message.draw(win)
#获得多边形的5个点
p1 = win.getMouse()
p1.draw(win)
p2 = win.getMouse()
p2.draw(win)
p3 = win.getMouse()
p3.draw(win)
p4 = win.getMouse()
p4.draw(win)
p5 = win.getMouse()
p5.draw(win)
# 使用Polygon对象绘制多边形
polygon = Polygon(p1, p2, p3, p4, p5)
polygon.setFill("peachpuff")
polygon.setOutline("black")
polygon.draw(win)
# 等待响应鼠标事件，退出程序
message.setText("Click anywhere to quit.")
win.getMouse()
```
Text对象：setText()和getText()  
Entry对象：setText()和getText()  
内容可以被用户修改
```
from graphics import *
def main():
    win = GraphWin("Click Me!")
    for i in range(10):
        p = win.getMouse()
        print("You clicked at:", p.getX(), p.getY())
if __name__ == '__main__':
    main()
```
在窗口中点击5个点来画一个五边形
```
from graphics import *
win = GraphWin("Celsius Temperature",400,300)
win.setCoords(0.0, 0.0, 3.0, 4.0)
# 绘制接口
Text(Point(1,3), " Celsius Temperature:").draw(win)
Text(Point(1,1), "Fahrenheit Temperature:").draw(win)
input = Entry(Point(2,3), 5)
input.setText("0.0")
input.draw(win)
output = Text(Point(2,1), "")
output.draw(win)
button = Text(Point(1.5,2.0), "Convert It")
button.draw(win)
Rectangle(Point(1,1.5), Point(2,2.5)).draw(win)
# 等待鼠标点击
win.getMouse()
# 转换输入
celsius = eval(input.getText())
fahrenheit = 9.0/5.0 * celsius + 32.0
# 显示输出，改变按钮
output.setText(fahrenheit)
button.setText("Quit")
# 等待响应鼠标点击，退出程序
win.getMouse()
win.close()
```
### GraphWin对象  
一个程序可以定义任意数量的窗体  
GraphWin()  
默认标题是"Graphics Window"  
默认大小为200*200

GraphWin对象常用方法  
* plot(x, y, color)
在窗口中(x,y)位置绘制像素。颜色参数可选，默认值为黑色。
* plotPixel(x, y, Color)
在“原始”位置(x,y)处绘制像素，忽略* setCoords()方法设置的坐标变换。
* setBackground(color)
将窗口背景颜色设为指定颜色，默认值为灰色。
* close()
关闭屏幕上的窗口。
* getMouse()
程序等待用户在窗口内点击鼠标，返回值为点击处的位置，并以Point对象返回。
* setCoords(xll, yll, xur, yur)
设置窗口的坐标系。左下角是(xll,yll)，右上角是(xur,yur)。所有后面的绘制都以这个坐标系做参照(plotPexil除外)
* 图形对象：点、线段、圆、椭圆、矩形、多边形以及文本
* 默认初始化：黑色边框；没有被填充；

图形对象通用方法
* setFill(color)
设置对象内部填充颜色。
* setOutline(color)
设置对象边框颜色。
* setWidth(pixels)
设置对象的宽度(对Point类不起作用)。
* draw(aGraphWin)
在指定的窗口中绘制对象。
* undraw()
从窗口中删除该对象。如该对象没有在窗口中画出将会报错。
* move(dx,dy)
将对象沿x轴和y轴分别移动dx和dy单位长度。
* clone()返回该对象的副本。

Point对象方法
* Point(x,y)
以指定坐标的值(x, y)构造一点
* getX()
返回该点的x坐标值
* getY()
返回该点的y坐标值

Line对象方法
* Line(point1, point2)
构造一个从点point1到点point2的线段
* setArrow(string)
设置线段的箭头样式。箭头可以绘制在左端，右端，或者两端都有。string参数值为’first’, ’last’, ’both’,或 ’none’，默认值为’none’。
* getCenter()
返回线段中点的坐标值。
* getP1(), getP2()
返回线段相应端点的坐标值。

Circle对象方法
* Circle(centerPoint, radius)
根据给定圆心和半径构建圆
* getCenter()
返回圆心的值
* getRadius()
返回圆的半径长度
* getP1(), getP2()
返回值为该圆边框对应点，对应点指的是该圆外接正方形的对角点。

Rectangle对象方法
* Rectangle(point1, point2)
以point1和point2为对角点创建一个矩形。
* getCenter()
返回矩形的中心点的克隆值。
* getP1(), getP2()
返回构造矩形的对角点的克隆值

Oval对象方法
* Oval(point1, point2)
在点point1和point2指定的边界框中创建一个椭圆。
* getCenter()
返回椭圆的中心点的坐标值
* getP1(), getP2()
返回构造椭圆的对角点的坐标值

Polygon 对象方法
* Polygon
(point1, point2, point3, ...)
根据给定的顶点构造一个多边形。也可以只用一个顶点列表作为参数
* getPoints()
返回构造多边形的顶点值的列表

Text 对象方法
* Text(anchorPoint, string)
以anchorPoint点的位置为中心，构建了一个内容为string的文本对象。
* setText(string)
设置文本对象的内容
* getText()
返回当前文本内容。
* getAnchor()
返回文本显示中间位置点anchor的坐标值。
* setFace(family)
设置文本字体。family可选值为：’helvetica’,’courier’, ’times roman’, 以及 ’arial’.
* setSize(point)
设置字体大小为给定点point的大小。合法数值为5-36。
* setStyle(style)
设置字体的风格。可选值为’normal’, ’bold’, ’italic’,以及 ’bold italic’。
* setTextColor(color)
设置文本颜色。与setFill效果相同。

图形颜色  
* Python中颜色由字符串指定
* 很多颜色具有不同深浅
    * 红色逐渐加深
    * ‘red1’‘red2’‘red3’ ‘red4’

color_rgb(red,green,blue)函数
* 设定颜色数值获得颜色
* 三个参数为0-255范围内的整数
* 返回一个字符串
    * color_rgb(255,0,0) 亮红色，
    * color_rgb(130,0,130) 中度洋红色。

温度转换程度示例
```
from graphics import *
 
win = GraphWin("Celsius Converter", 400, 300)
win.setCoords(0.0, 0.0, 3.0, 4.0)
# 绘制接口
Text(Point(1,3), " Celsius Temperature:").draw(win)
Text(Point(1,1), "Fahrenheit Temperature:").draw(win)
input = Entry(Point(2,3), 5)
input.setText("0.0")
input.draw(win)
output = Text(Point(2,1),"")
output.draw(win)
button = Text(Point(1.5,2.0),"Convert It")
button.draw(win)
Rectangle(Point(1,1.5), Point(2,2.5)).draw(win)
# 等待鼠标点击
win.getMouse()
# 转换输入
celsius = eval(input.getText())
fahrenheit = 9.0/5.0 * celsius + 32.0
# 显示输出，改变按钮
output.setText(fahrenheit)
button.setText("Quit")
# 等待响应鼠标点击，退出程序
win.getMouse()
win.close()
```  
计算温度值设定窗口颜色：  
温度越高，颜色越偏红  
温度越低，颜色越偏蓝  
setBackground(Newcolor)设置窗口背景颜色。  
假定输入温度范围为0-100，  
颜色权重weight=输入温度/100  
newcolor的rgb计算：  
红色分量=255*weight  
绿色分量=66+150(1-weight)  
蓝色分量=255*(1-weight)  
```
from graphics import *
 
def convert(input):
    celsius = eval(input.getText())    # 输入转换
    fahrenheit = 9.0/5.0 * celsius + 32
    return fahrenheit 
def colorChange(win,input):
    cnum = eval(input.getText())
    weight = cnum / 100.0
    newcolor =color_rgb(255*weight,66+150*(1-weight),255*(1-weight))
    win.setBackground(newcolor)
def main():
    win = GraphWin("Celsius Converter", 400, 300)
    win.setCoords(0.0, 0.0, 3.0, 4.0)
    # 绘制输入接口
    Text(Point(1,3),
         " Celsius Temperature:").draw(win)
    Text(Point(2,2.7),
         " (Please input 0.0-100.0 )").draw(win)
    Text(Point(1,1),
         "Fahrenheit Temperature:").draw(win)
    input = Entry(Point(2,3), 5)
    input.setText("0.0")
    input.draw(win)
    output = Text(Point(2,1),"")
    output.draw(win)
    button = Text(Point(1.5,2.0),"Convert It")
    button.draw(win)
    rect = Rectangle(Point(1,1.5), Point(2,2.5))
    rect.draw(win)
    # 等待鼠标点击
    win.getMouse()
    result = convert(input)    # 转换输入
    output.setText(result)    # 显示输出 
    # 改变颜色
    colorChange(win,input)
    # 改变按钮字体
    button.setText("Quit")
    # 等待点击事件，退出程序
    win.getMouse()
    win.close()
 
if __name__ == '__main__':
    main()
```
## Tkinter库
创建GUI程序的基本步骤为：  
导入Tk模块.  
创建GUI应用程序的主窗口.  
添加控件或GUI应用程序.  
进入主事件循环，等待响应用户触发事件.  

15中常见的Tk控件  
Button, Canvas, Checkbutton, Entry, Frame, Label, Listbox, Menubutton, Menu, Message, Radiobutton, Scale Scrollbar, Text, Toplevel, Spinbox PanedWindow, LabelFrame, tkMessageBox

共同属性
* Dimensions ：尺寸
* Colors：颜色
* Fonts：字体
* Anchors：锚
* Relief styles：浮雕式
* Bitmaps：显示位图
* Cursors：光标的外形
特有属性

界面布局  
Tkinter三种几何管理方法  
pack()
grid()
place()  
简单GUI示例
```
from tkinter import *
tk = Tk()
label = Label(tk, text="Welcome to Python Tkinter")
button = Button(tk, text="Click Me")
label.pack()
button.pack()
tk.mainloop()
```
响应用户事件示例
```
from tkinter import *
def processOK():
    print("OK button is clicked")
def processCancel():
    print("Cancel button is clicked")
def main():
    tk = Tk()
    btnOK = Button(tk, text="OK", fg = "red", command = processOK)
    btnCancel = Button(tk, text = "Cancel", bg = "yellow", command = processCancel)
    btnOK.pack()
    btnCancel.pack()
    tk.mainloop()
```
显示文字、图片、绘制图形
```
from tkinter import *
tk = Tk()
canvas = Canvas(tk, width = 200, height = 200)
canvas.pack()
canvas.create_text(100, 40, text = "Welcome to Tkinter", fill = "blue", font = ("Times", 16))
myImage = PhotoImage(file = "python_logo.gif")
canvas.create_image(10, 70, anchor = NW, image = myImage)
canvas.create_rectangle(10, 70, 190, 130)
tk.mainloop()
```
控制图形移动的示例
```
from tkinter import *
tk = Tk()
canvas = Canvas(tk, width = 400, height = 400)
canvas.pack()
def moverectangle(event):
    if event.keysym == "Up":
        canvas.move(1, 0, -5)
    elif event.keysym == "Down":
        canvas.move(1, 0, 5)
    elif event.keysym == "Left":
        canvas.move(1, -5, 0)
    elif event.keysym == "Right":
        canvas.move(1, 5, 0)
canvas.create_rectangle(10,10,50,50,fill="red")
canvas.bind_all("<KeyPress-Up>",moverectangle)
canvas.bind_all("<KeyPress-Down>",moverectangle)
canvas.bind_all("<KeyPress-Left>",moverectangle)
canvas.bind_all("<KeyPress-Right>",moverectangle)
```
基于tkinter库完成聊天窗口GUI  
见ChatWin.py


