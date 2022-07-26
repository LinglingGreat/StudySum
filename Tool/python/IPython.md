## IPython的%魔术命令
%magic 显示所有魔术命令  
%hist IPython命令的输入历史  
%pdb 异常发生后自动进入调试器  
%reset 删除当前命名空间中的全部变量或名称  
%who 显示Ipython当前命名空间中已经定义的变量  
%time statement 给出代码的执行时间, statement表示一段代码  
%timeit statement 多次执行代码，计算综合平均执行时间  
```
In: a = np.random.randn(1000, 1000)
In: %timeit np.dot(a, a)
Out: 10 loops, best of 3: 85.7 ms per loop
```
%matplotlib inline 内嵌画图，不再需要plt.show()

## pip使用

install 安装库  
如pip install py2exe  
更新库的命令格式：pip install -U [安装库名称]  
uninstall 卸载库  
pip uninstall [安装库名称]  
list 列出已经安装库的信息  
显示已安装库：pip list  
显示有更新的库命令：pip list --outdated  
show 列出已经安装库的详细信息:pip show [安装库名称]    
search 通过PyPI搜索库：pip search [关键词]
help 帮助命令： 如pip help install  