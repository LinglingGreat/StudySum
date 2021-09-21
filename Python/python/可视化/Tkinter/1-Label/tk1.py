from tkinter import *
# 导入tkinter模块的所有内容

root = Tk()

# 创建一个文本Label对象
textLabel = Label(root,
                  text="您所下载的影片含有未成年人限制内容，\n请满18岁后再点击观看！",
                  justify=LEFT,
                  padx=10)
textLabel.pack(side=LEFT)

# 创建一个图像Label对象
# 用PhotoImage实例化一个图片对象（支持gif格式的图片）
photo = PhotoImage(file="18.gif")
imgLabel = Label(root, image=photo)
imgLabel.pack(side=RIGHT)

mainloop()
