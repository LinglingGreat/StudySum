from tkinter import *

root = Tk()

text = Text(root, width=30, height=2)
text.pack()

text.insert(INSERT, "I love \n")
text.insert(END, "FishC.com!")

mainloop()
