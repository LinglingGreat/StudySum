from tkinter import *

master = Tk()

def test():
    if e1.get() == "小甲鱼":
        print("正确！")
        return True
    else:
        print("错误！")
        e1.delete(0, END)
        return False

v = StringVar()

e1 = Entry(master, textvariable=v, validate="focusout", validatecommand=test)
e2 = Entry(master)
e1.pack(padx=10, pady=10)
e2.pack(padx=10, pady=10)

mainloop()
