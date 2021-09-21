from tkinter import *

root = Tk()

listbox = Listbox(root)
listbox.pack(fill=BOTH, expand=True)

for i in range(10):
    listbox.insert(END, str(i))

mainloop()
