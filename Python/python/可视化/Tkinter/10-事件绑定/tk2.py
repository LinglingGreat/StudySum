from tkinter import *

root = Tk()

def callback(event):
    print(event.keysym)

frame = Frame(root, width=200, height=200)
frame.bind("<Key>", callback)
frame.focus_set()
frame.pack()

mainloop()
