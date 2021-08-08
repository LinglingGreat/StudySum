from tkinter import *

root = Tk()

photo = PhotoImage(file="bg.gif")
theLabel = Label(root,
                 text="学Python\n到FishC",
                 justify=LEFT,
                 image=photo,
                 compound=CENTER,
                 font=("华康少女字体", 20),
                 fg="white"
                 )
theLabel.pack()

mainloop()
