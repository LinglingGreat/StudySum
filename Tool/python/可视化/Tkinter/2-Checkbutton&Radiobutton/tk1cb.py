from tkinter import *

root = Tk()

# 需要一个Tkinter变量，用于表示该按钮是否被选中
v = IntVar()

c = Checkbutton(root, text="测试一下", variable=v)
c.pack()

# 如果选项被选中，那么变量v被赋值为1，否则为0
# 我们可以用个Label标签动态地给大家展示：
l = Label(root, textvariable=v)
l.pack()

mainloop()
