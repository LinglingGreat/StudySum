from tkinter import *

master = Tk()

v = StringVar()

def test(content, reason, name):
    if content == "小甲鱼":
        print("正确！")
        print(content, reason, name)
        return True
    else:
        print("错误！")
        print(content, reason, name)
        return False

testCMD = master.register(test)
e1 = Entry(master, textvariable=v, validate="focusout", \
           validatecommand=(testCMD, '%P', '%v', '%W'))
e2 = Entry(master)
e1.pack(padx=10, pady=10)
e2.pack(padx=10, pady=10)

mainloop()
