## 一些安装问题

pycharm破解：https://mp.weixin.qq.com/s/4TWpEl26waFyJ10fgCzbsQ

### cn2an

先pip install PyYAML -U --ignore-installed，再pip install cn2an -U

## 进入虚拟环境并用pip install 安装包

[https://www.cnblogs.com/xiaohaodeboke/p/11837304.html](https://www.cnblogs.com/xiaohaodeboke/p/11837304.html)

1.  cmd 命令进入虚拟环境所在的文件夹（Pycharm在每创建一个新项目时就会创建一个虚拟环境文件夹）Scripts文件夹下
2.  命令行执行 activate（第一步的文件夹是虚拟环境所在的文件夹venv，scripts是虚拟环境文件夹下的，activate是激活组件）
3.  之后即进入虚拟环境
4.  如图(MyDjango是自己的项目文件夹)

退出该环境：

1.命令行执行 deactivate.bat（直接使用deactivate即可，同样执行该命令也得在Scripts文件夹下）

使用命令deactivate.bat退出虚拟环境，如图

来自 <[https://www.cnblogs.com/xiaohaodeboke/p/11837304.html](https://www.cnblogs.com/xiaohaodeboke/p/11837304.html)>

## log模块

[https://cuiqingcai.com/6080.html](https://cuiqingcai.com/6080.html)

## 多进程

```Python
Import multiprocessing
cores = multiprocessing.cpu_count()
pool = multiprocessing.Pool(processes=cores)  # 多进程
print(datetime.datetime.now())
# features = pool.map(upload_obj_fromurl, flat_list)
cnt = 0
for _ in pool.imap_unordered(upload_obj_fromurl, flat_list):
    tim = datetime.datetime.now()
    print('当前时间是%s，done %d/%d, %s\r' % (datetime.datetime.now(), cnt, len(flat_list), flat_list[cnt]))
    cnt += 1
pool.close()
pool.join()

```

## json模块

加载json文件

```python
json.load(open(filename, encoding='utf-8'))
```

写入json文件

```python
json.dump(
        data,
        open(out_file, 'w', encoding='utf-8'),
        indent=4,
        ensure_ascii=False
    )
```

## 生成requirements文件

第一种：pip freeze > requirements.txt会将环境中的依赖包全都加入

第二种会简洁一些：

`--force` 强制执行，当 生成目录下的 requirements.txt 存在时覆盖。

```Python
# 安装
pip install pipreqs
# 在当前目录生成
pipreqs . --encoding=utf8 --force
```

