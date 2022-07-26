from tkinter import *

root = Tk()

e = Entry(root)
e.pack(padx=20, pady=20)

e.delete(0, END)  # 清空输入框
e.insert(0, "默认文本...")

mainloop()
