
## shell 
### 什么是shell
![操作系统架构](https://www3.nd.edu/~pbui/teaching/cse.30341.fa17/static/img/house.png)

### 常用的shell命令
- 文件与目录
```bash
ls
```
列出当前目录内容
```bash
cd /home/
```
进入home目录

- 安装软件
```bash
yum install vim
```
安装vim

### shell脚本编写
- 变量    

第一个脚本的编写
1. 编辑run.sh，vim run.sh
2. 按下"i"进入编辑模式,输入:
```bash
NAME="Hello"
echo $NAME
```
3. 保存，:wq
4. 执行，bash run.sh

- 循环
```bash
for ((i = 0 ; i < 100 ; i++)); do
  echo $i
done
```

- 条件判断
```bash
if [[ X ]] && [[ Y ]]; then
  ...
fi
```

- 函数
```bash
function sayHello() {
    echo "hello $1"
}
sayHello "John"
```