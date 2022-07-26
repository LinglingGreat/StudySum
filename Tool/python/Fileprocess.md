## 文件的基本处理
打开文件  
- 建立磁盘上的文件与程序中的对象相关联  
- 通过相关的文件对象获得   
```
<variable> = open (<name>, <mode>) 
<name>磁盘文件名   
<mode>打开模式
```
文件打开模式

| **打开模式** | **执行操作**                             |
| -------- | ------------------------------------ |
| 'r'      | 以只读方式打开文件（默认），如果文件不存在，则输出错误          |
| 'w'      | 以写入的方式打开文件，会覆盖已存在的文件（如果文件不存在，则自动创建文件 |
| 'x'      | 如果文件已经存在，使用此模式打开将引发异常                |
| 'a'      | 以写入模式打开，如果文件存在，则在末尾追加写入              |
| 'b'      | 以二进制模式打开文件                           |
| 't'      | 以文本模式打开（默认）                          |
| '+'      | 可读写模式（可添加到其他模式中使用）                   |
| 'U'      | 通用换行符支持                              |

rb 只读二进制文件。如果文件不存在，则输出错误  
wb 只写二进制文件，如果文件不存在，则自动创建文件。  
ab 附加到二进制文件末尾  
r+ 读写      

文件对象方法

| **文件对象方法**                     | **执行操作**                                 |
| ------------------------------ | ---------------------------------------- |
| f.close()                      | 关闭文件                                     |
| f.read([size=-1])              | 从文件读取size个字符，当未给定size或给定负值的时候，读取剩余的所有字符，然后作为字符串返回 |
| f.readline([size=-1])          | 从文件中读取并返回一行（包括行结束符），如果有size有定义则返回size个字符 |
| f.write(str)                   | 将字符串str写入文件                              |
| f.writelines(seq)              | 向文件写入字符串序列seq，seq应该是一个返回字符串的可迭代对象        |
| f.seek(offset, from)           | 在文件中移动文件指针，从from（0代表文件起始位置，1代表当前位置，2代表文件末尾）偏移offset个字节 |
| f.tell()                       | 返回当前在文件中的位置                              |
| f.truncate([size=file.tell()]) | 截取文件到size个字节，默认是截取到文件指针当前位置              |

文件操作  
+ 读取  
  read() 返回值为包含整个文件内容的一个字符串  
  readline() 返回值为文件下一行内容的字符串。  
  readlines() 返回值为整个文件内容的列表，每项是以换 行符为结尾的一行字符串。  
+ 写入  
  从计算机内存向文件写入数据  
  write()：把含有本文数据或二进制数据块的字符串写入文件中。  
  writelines()：针对列表操作，接受一个字符串列表作为参数，将它们写入文件。  

文件遍历：最常见的文件处理方法  
比如：拷贝文件， 根据数据文件定义行走路径， 将文件由一种编码转换为另外一种编码  

+ 定位
+ 其他：追加、计算等

关闭文件  
* 切断文件与程序的联系  
* 写入磁盘，并释放文件缓冲区   
```
In: a = np.random.randn(1000, 1000)
In: %timeit np.dot(a, a)
Out: 10 loops, best of 3: 85.7 ms per loop
```
## office文件操作库
* xlwt 生成excel表单  
* xlrd 读入并处理excel表单  
* python-docx 创建并更新word文件  
* python-pptx 创建并更新powerpoint文件 

### Excel编程  

利用xlrd模块读取并简单操作excel文档   
* 打开excel文档  
  workbook = xlrd.open_workbook('testread.xls')
* 获取所有sheet  
  sheet_name = workbook.sheet_names()(返回类型为sheet名字组成list)
* 获取指定sheet  
  根据sheet的sheet_by_index属性索引获取  
  根据sheet的sheet_by_name属性名字获取  
* 获取指定sheet的名字，行数，列数  
  调用指定sheet的name, nrows, ncols属性
* 获取sheet的内容  
  将sheet按照二维数组，根据行列的方式访问指定内容  
  举例：第0行第1列数据  
  sheet.row(0)[1].value  
  sheet.cell(0,1).value

举例：excel文件处理
```
import xlrd
path = input("请输入excel文件路径：")
workbook = xlrd.open_workbook(path)
sheet = workbook.sheet_by_index(0)
for row in range(sheet.nrows):
    print()
    for col in range(sheet.ncols):
        print("%7s"%sheet.row(row)[col].value,'\t',end='')
```
利用xlwt模块可实现excel文档的自动生成  
* 创建工作簿  
  file = xlwt.Workbook()(调用xlwt的Workbook实现)
* 创建sheet  
  调用add_sheet返回一个Worksheet类  
  创建sheet有可选参数cell_overwrite_ok,表示是否可以覆盖单元格，默认值为false
* Sheet的内容添加  
  调用sheet的write属性实现  
  常用write用法：write(x,y,string,stype)  
  x:表示行  
  y:表示列  
  string：表示要写入的单元格内容  
  stype:表示单元格样式  

### Word编程
python-docx库  
* 新建文档：document = Document()  
  document.save('filename.docx')
* 添加文本：text = document.add_paragraph('content of the paragraph')
* 更改项目符号：text.style = 'stylename'
* 添加标题：document.add_heading('head-name')
* 添加图片：document.add_picture('path-of-the-picture')
* 字体设置  
  设置加粗：text.run.font.bold = True  
  设置字号：text.run.font.size = pt(sizeNumber)  
  设置字体颜色：text.run.font.color =  
* 创建表格：table = document.add_table(rows=, cols=)  
* 遍历某一单元格：cell = table.cell(row_num, col_num)  
* 对单元格操作：  
  添加文本:cell.add_paragraph("content",style=None)  
  添加另一表格：cell.add_table(rows,cols)  
  返回该单元格内文本：String content = cell.text(只读)  
  返回该单元格内表格list:table[]=cell.tables(只读)
### Powerpoint编程
python_pptx  
* 用于创建和编辑PowerPoint(.pptx)文件的Python库  
* 自动生成符合模板格式的PowerPoint文件  
* 用于对幻灯片进行批量更新  

python-pptx功能  
* 新建幻灯片
* 在幻灯片的固定位置插入某一大小的图片
* 向幻灯片中添加文本框
* 向幻灯片中插入表格
* 设置字体颜色，大小，字体...
* 对某一张幻灯片的某一部分进行操作
* ...

python-pptx库的使用
* Presentation:操作ppt对象
    * Presentation()创建一个PPT文档
    * .slide_layouts[]确定幻灯片的先后顺序
    * .slides.add_slide()增加一个幻灯片
    * Slide.shape.title表示一个幻灯片的标题
    * Slide.shape.placeholders表示一个幻灯片的内容
    * .save()函数用来存储幻灯片  

举例：生成一个简单的PPT标题页面
```
from pptx import Presentation
prs = Presentation()
title_slide_layout = prs.slide_layouts[0]
slide = prs.slides.add_slide(title_slide_layout)
title = slide.shapes.title
subtitle = slide.placeholders[1]
title.text = "Hello, World!"
subtitle.text = "python-pptx was here!"
prs.save('test.pptx')
```
举例：生成一个简单的PPT内容页面
```
from pptx import Presentation

prs = Presentation()
bullet_slide_layout = prs.slide_layouts[1]

slide = prs.slides.add_slide(bullet_slide_layout)
shapes = slide.shapes

title_shape = shapes.title
body_shape = shapes.placeholders[1]

title_shape.text = 'Adding a Bullet Slide'

tf = body_shape.text_frame
tf.text = 'Find the bullet slide layout'

p = tf.add_paragraph()
p.text = 'Use _TextFrame.text for first bullet'
p.level = 1

p=tf.add_paragraph()
p.text = 'Use _TextFrame.add_paragraph() for subsequent bullets'
p.level = 2

prs.save('test.pptx')
```
* Slides:对幻灯片进行操作
* Shapes：对幻灯片的某一区域进行操作
    * shapes.add_textbox()增加文本框
    * shapes.add_picture()增加图片
    * shapes.add_shape()增加形状
    * shapes.add_table()增加表格

举例：带有图片、文本框和图形的代码
```
from pptx import Presentation
from pptx.util import Inches

img_path = 'monty-truth.png'

prs = Presentation()
blank_slide_layout = prs.slide_layouts[6]
slide = prs.slides.add_slide(blank_slide_layout)

left = top = Inches(1)
pic = slide.shapes.add_picture(img_path, left, top)

left = Inches(5)
height = Inches(5.5)
pic = slide.shapes.add_picture(img_path, left, top, height=height)
prs.save('test.pptx')
```
* Table:表格操作
* Text:文本操作


