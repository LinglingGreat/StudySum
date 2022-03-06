from tkinter import *
import hashlib

root = Tk()

text = Text(root, width=30, height=5)
text.pack()

text.insert(INSERT, "I love FishC.com!")

def getIndex(text, index):
    return tuple(map(int, str.split(text.index(index), ".")))

start = "1.0"
while True:
    pos = text.search("o", start, stopindex=END)
    if not pos:
        break
    print("找到啦，位置是：", getIndex(text, pos))
    start = pos + "+1c"

mainloop()
