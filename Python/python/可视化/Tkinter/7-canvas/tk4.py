from tkinter import *

root = Tk()

w = Canvas(root, width=200, height=100)
w.pack()

w.create_rectangle(40, 20, 160, 80, dash=(4, 4))
w.create_oval(70, 20, 130, 80, fill="pink")
w.create_text(100, 50, text="FishC")

mainloop()
