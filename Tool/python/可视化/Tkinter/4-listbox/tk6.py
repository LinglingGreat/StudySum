from tkinter import *

root = Tk()

Scale(root, from_=0, to=42, tickinterval=5, length=200, \
      resolution=5, orient=VERTICAL).pack()
Scale(root, from_=0, to=200, tickinterval=10, length=600, \
      orient=HORIZONTAL).pack()

mainloop()
