from tkinter import *

root = Tk()

def callback():
    fileName = filedialog.askopenfilename(filetypes=[("PNG", ".png"), ("GIF", ".gif"), ("JPG", ".jpg"), ("Python", ".py")])
    print(fileName)

Button(root, text="打开文件", command=callback).pack()

mainloop()
