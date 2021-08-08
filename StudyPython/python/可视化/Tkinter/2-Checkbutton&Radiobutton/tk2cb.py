from tkinter import *

root = Tk()

GIRLS = ["西施", "王昭君", "貂蝉", "杨玉环"]

v = []

for girl in GIRLS:
    v.append(IntVar())
    b = Checkbutton(root, text=girl, variable=v[-1])
    b.pack(anchor=W)

mainloop()
