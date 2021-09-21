from tkinter import *

root = Tk()

def callback():
    print("你好~")

menubar = Menu(root)
menubar.add_command(label="hello", command=callback)
menubar.add_command(label="quit", command=root.quit)

root.config(menu=menubar)

mainloop()
