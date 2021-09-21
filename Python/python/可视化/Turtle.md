## 五角星的绘制
```
from turtle import *
fillcolor("red")
begin_fill()
while True:
    forward(200)
    right(144)
    if abs(pos())<1:
        break
end_fill()
done()
```
```
# 绘制五角星
p = Turtle()
p.speed(3)
p.pensize(5)
p.color("black", 'yellow')
#p.fillcolor("red")
p.begin_fill()
for i in range(5):
    p.forward(200)
    p.right(144)
p.end_fill()
```
## 太阳花的绘制
```
from turtle import *
color('red','yellow')
begin_fill()
while True:
    forward(200)
    left(170)
    if abs(pos())<1:
        break
end_fill()
done()
```
## 螺旋线绘制
```
import turtle
import time
turtle.speed("fastest")
turtle.pensize(2)
for x in range(100):
    turtle.forward(2*x)
    turtle.left(90)
time.sleep(3)
done()
```
## 彩色螺旋线的绘制
```
import turtle
import time
turtle.pensize(2)
turtle.bgcolor("black")
colors=["red","yellow",'purple','blue']
turtle.tracer(False)
for x in range(400):
    turtle.forward(2*x)
    turtle.color(colors[x % 4])
    turtle.left(91)
turtle.tracer(True)
done()
```
## 蟒蛇的绘制
```
import turtle

## 等边三角形
for i in range(3):
    turtle.seth(i*120)    # 设置当前朝向为angle角度
    turtle.fd(100)    # 沿着当前方向前进指定距离

def drawSnake(rad, angle, len, neckrad):
    for i in range(len):
        turtle.circle(rad, angle)    # 绘制一个指定半径，角度以及绘制步骤的圆
        turtle.circle(-rad, angle)
    # 参数rad描述圆形轨迹半径的位置
    # 这个半径在小乌龟运行的左侧rad远位置处，如果rad为负值，则半径在小乌龟运行的右侧
    # 参数angle表示小乌龟沿着圆形爬行的弧度值
    turtle.circle(rad, angle/2)
    # turtle.fd()函数也可以用turtle.forward()表示乌龟向前直线爬行移动
    # 表示小乌龟向前直线爬行移动，它有一个参数表示爬行的距离
    turtle.fd(rad)
    turtle.circle(neckrad+1, 180)
    turtle.fd(rad*2/3)

# main()函数给出了小乌龟爬行的窗体大小，爬行轨迹颜色和宽度以及初始爬行的方位。   
def main():
    # 开启窗口大小(启动一个图形窗口)
    # turtle.setup(width, height, startx，starty)
    # 启动窗口的宽度和高度,表示窗口启动时，窗口左上角在屏幕中的坐标位置
    turtle.setup(1300, 800, 0, 0)
    # 启动一个1300像素宽、800像素高的窗口，该窗口的左上角是屏幕的左上角。
    # 小乌龟运行轨迹的宽度，为30像素
    pythonsize = 30
    # 设置画笔的宽度
    turtle.pensize(pythonsize)
    # 每个部分用不同颜色？绘制彩色蟒蛇
    # 小乌龟运行轨迹的颜色
    # turtle.pencolor(“#3B9909”)
    turtle.pencolor("blue")
    # 修改seth参数
    # 小乌龟启动时运行的方向，参数是角度值
    # 0表示向东，90度向北，180度向西，270度向南；负值表示相反方向
    # 向东南方向40度
    turtle.seth(-40)
    # 调用drawSnake函数启动绘制蟒蛇功能
    drawSnake(40,80,5,pythonsize/2)
    
main()
```
## 引入方式
```
import turtle
from turtle import *
```
## 控制画笔绘制状态的函数
    pendown()   | pd()     | down()放下画笔
    penup()     | pu()     | up()提起笔，用于另起一个地方绘制时用，与pendown()配对使用
    pensize(wid )          | width(wid)设置画笔线条的粗细为指定大小
## 控制画笔颜色和字体函数
    color()设置画笔的颜色     
    reset()清空当前窗口，并重置位置等状态为默认值
    begin_fill()填充图形前，调用该方法   
    end_fill()填充图形结束 
    filling()返回填充的状态，True 为填充，False 为未填充   
    clear()清空当前窗口，但不改变当前画笔的位置 
    screensize()设置画布的长和宽
    showturtle() | st()显示画笔的turtle 形状
    hideturtle() | ht()隐藏画笔的turtle 形状
    isvisible()如果turtle 可见，则返回True 
    write(arg,move=False,align="left",font =("Arial",8,"normal") )输出font 字体的字符串
## 控制画笔运动的函数
    forward(distance) | fd(distance)沿着当前方向前进指定距离
    backward(distance)| bk(distance)沿着当前相反方向后退指定距离
    |back(distance)
    right(angle) | rt(angle)向右旋转angle角度
    left(angle) | lt(angle)
    setheading(to_angle)设置当前朝向为angle角度
    position() | pos()
    goto(x,y )移动到绝对坐标（x,y）处 
    setposition(x,y ) | setpos(x,y )
    circle(radius,extent ,steps )绘制一个指定半径，角度、以及绘制步骤step 的圆
    dot(size ,*color) radians()绘制一个指定半径r 和颜色color 的圆点
    stamp() speed(speed )
    clearstamp(stamp_id)
    clearstamps(n ) undo()
    speed(speed ) heading()
    towards(x,y ) distance(x,y )
    xcor() ycor() 
    setx(x) sety(y)将当前x或y轴移动到指定位置
    home()设置当前画笔位置为原点，朝向东
    undo()撤销画笔最后一步动作
    degrees(fullcircle = 360.0)
## TurtleScreen/Screen类的函数
    bgcolor(*args)    getcanvas() 
    bgpic(picname )   getshapes()
    clearscreen()     turtles()
    resetscreen()     window_height()
    screensize(cwid ,canvh,bg )   window_width()
    tracer(n ,delay )    bye()
    listen(xdummy ,ydummy )    exitonclick()
    onkey((fun,key)     title(titlestring)
    onkeyrelease((fun,key)     onkeypress(fun,key )
    onscreenclick(fun,btn=1,add )  
    setup(wid=_CFG["wid"],h=_CFG["h"], startx=_CFG["leftright"], starty=_CFG["topbottom"])
## 绘制树或森林
drawtree.py, drawforest.py
## 七段数码管绘制
数码管是一种价格便宜、使用简单的发光电子器件，广泛应用在价格较低的电子类产品中，其中，七段数码管最为常用。七段数码管（seven-segment indicator）由7 段数码管拼接而成，每段有亮或不亮两种情况，改进型的七段数码管还包括一个小数点位置.

七段数码管能形成27=128 种不同状态，其中部分状态能够显示易于人们理解的数字或字母含义，因此被广泛使用。

该问题的IPO 描述如下：  
输入：当前日期的数字形式  
处理：根据每个数字绘制七段数码管表示  
输出：绘制当前日期的七段数码管表示  

DrawSevenSegDisplay.py, DrawSevenSegDisplay1.py
## 应用circle方法绘制图形ColorShapes.py
```
import turtle
 
def main():
    turtle.pensize(3)
    turtle.penup()
    turtle.goto(-200,-50)
    turtle.pendown()
    # 修饰
    turtle.begin_fill()
    turtle.color("red")
    turtle.circle(40, steps=3)
    turtle.end_fill()
 
 
    turtle.penup()
    turtle.goto(-100,-50)
    turtle.pendown()
    # 修饰
    turtle.begin_fill()
    turtle.color("blue")
    turtle.circle(40, steps=4)
    turtle.end_fill()
 
    turtle.penup()
    turtle.goto(0,-50)
    turtle.pendown()
    # 修饰
    turtle.begin_fill()
    turtle.color("green")
    turtle.circle(40, steps=5)
    turtle.end_fill()
 
    turtle.penup()
    turtle.goto(100,-50)
    turtle.pendown()
    # 修饰
    turtle.begin_fill()
    turtle.color("yellow")
    turtle.circle(40, steps=6)
    turtle.end_fill()
 
    turtle.penup()
    turtle.goto(200,-50)
    turtle.pendown()
    # 修饰
    turtle.begin_fill()
    turtle.color("purple")
    turtle.circle(40)
    turtle.end_fill()
 
    turtle.color("green")
    turtle.penup()
    turtle.goto(-100,50)
    turtle.pendown()
    turtle.write(("Cool Colorful shapes"),
        font = ("Times", 18, "bold"))
    turtle.hideturtle()
 
    turtle.done
 
if __name__ == '__main__':
    main()
```
## 时钟模拟
编写Python程序模拟时钟，要求时钟根据计算机系统时间实时动态更新。见clock.py
## Turtle艺术
《雪景-Snowfall》绘制  
随机因素：
雪花位置，雪花颜色，雪花大小，花瓣数目，地面灰色长短，地面灰色位置  
见Snowfall.py  
《Rainbow》绘制  
颜色空间  
* RGB模型：
    * 光的三原色
    * 色相由RGB共同决定
* HSV模型：
    * H色彩、 S深浅、V明暗
    * 色相由H决定
见Rainbow.py
