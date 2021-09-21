from tkinter import *

master = Tk()

# 创建一个空列表
theLB = Listbox(master, setgrid=True)
theLB.pack()

# 往列表里添加数据
for item in ["鸡蛋", "鸭蛋", "鹅蛋", "李狗蛋"]:
    theLB.insert(END, item)

theButton = Button(master, text="删除", command=lambda x=theLB: x.delete(ACTIVE))
theButton.pack()

mainloop()
