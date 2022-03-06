from tkinter import *

root = Tk()

def callback():
    print("你好~")

menubar = Menu(root)

openVar = IntVar()
saveVar = IntVar()
quitVar = IntVar()

filemenu = Menu(menubar, tearoff=True)
filemenu.add_checkbutton(label="打开", command=callback, variable=openVar)
filemenu.add_checkbutton(label="保存", command=callback, variable=saveVar)
filemenu.add_separator()
filemenu.add_checkbutton(label="退出", command=root.quit, variable=quitVar)
menubar.add_cascade(label="文件", menu=filemenu)

editVar = IntVar()

editmenu = Menu(menubar, tearoff=False)
editmenu.add_radiobutton(label="剪切", command=callback, variable=editVar, value=1)
editmenu.add_radiobutton(label="拷贝", command=callback, variable=editVar, value=2)
editmenu.add_radiobutton(label="黏贴", command=callback, variable=editVar, value=3)
menubar.add_cascade(label="编辑", menu=editmenu)

root.config(menu=menubar)

mainloop()
