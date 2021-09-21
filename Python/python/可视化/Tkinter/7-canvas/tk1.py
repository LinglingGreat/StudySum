from tkinter import *

root = Tk()

w = Canvas(root, width=200, height=100)
w.pack()

w.create_line(0, 50, 200, 50, fill="yellow")
w.create_line(100, 0, 100, 100, fill="red", dash=(4, 4))
w.create_rectangle(50, 25, 150, 75, fill="blue")

mainloop()
