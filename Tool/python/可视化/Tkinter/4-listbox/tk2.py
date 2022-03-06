from tkinter import *

master = Tk()

# 创建一个空列表
theLB = Listbox(master, height=11)
theLB.pack()

# 往列表里添加数据
for item in range(11):
    theLB.insert(END, item)

mainloop()
