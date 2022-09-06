## 模块(Module)

if \__name__ == ‘\_\_main__’



搜索路径

```python
import sys
print(sys.path)
sys.path.append("文件路径")
```



包（package）

​	1.创建一个文件夹，用于存放相关的模块，文件夹的名字即包的名字；

​	2.在文件夹中创建一个\__init__.py的模块文件，内容可以为空；

​	3.将相关的模块放入文件夹中

导入时：包名.模块名