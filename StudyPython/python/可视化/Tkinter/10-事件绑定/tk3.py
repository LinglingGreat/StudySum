from tkinter import *

root = Tk()

def callback(event):
    print("当前位置：", event.x, event.y)

frame = Frame(root, width=200, height=200)
frame.bind("<Motion>", callback)
frame.pack()

mainloop()
