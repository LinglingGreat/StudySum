## github不允许上传超过100MB文件的问题

使用Git LFS

只需设置一次LFS:

```
git lfs install
```

然后，跟踪你要push的大文件的文件或指定文件类型

```
git lfs track "*.pdf"
```

**查看正在追踪的文件模式（patterns）** 

```
git lfs track
```

设置完毕后，就是正常的add, commit, push

```
git add yourLargeFile.pdf
git commit -m "Add Large file"
git push origin master
```

查看 LFS 跟踪的文件列表

```
git lfs ls-files
```

克隆包含 Git LFS 文件的远程仓库

```
git lfs clone https://git.coding.net/coding/coding-manual.git
```

如果你已知自己提交的版本库中确实存在一些大于 100MB 的文件，不妨先搜索：

```
find ./ -size +100M
```

更多帮助

```
git lfs help
```

## 删除错误添加到暂存区的文件

仅仅删除暂存区里的文件

```
git rm --cache 文件名
```

删除暂存区和工作区的文件

```
git rm -f 文件名
```



## 删除错误提交的commit

```
//查看版本ID
git log
//仅仅只是撤销已提交的版本库，不会修改暂存区和工作区
git reset --soft 版本库ID
//仅仅只是撤销已提交的版本库和暂存区，不会修改工作区
git reset --mixed 版本库ID
//彻底将工作区、暂存区和版本库记录恢复到指定的版本库
git reset --hard 版本库ID
```

注意：写哪个版本库ID就会回到那个版本

参考

https://blog.csdn.net/tyro_java/article/details/53440666

http://www.liuxiao.org/2017/02/git-%E5%A4%84%E7%90%86-github-%E4%B8%8D%E5%85%81%E8%AE%B8%E4%B8%8A%E4%BC%A0%E8%B6%85%E8%BF%87-100mb-%E6%96%87%E4%BB%B6%E7%9A%84%E9%97%AE%E9%A2%98/

## fork别人的项目

https://zhuanlan.zhihu.com/p/51199833?utm_source=wechat_session&utm_medium=social

https://www.zhihu.com/question/21682976

1. 先点击 fork 仓库，项目现在就在你的账号下了

\2. 在你自己的机器上 git clone 这个仓库，切换分支（也可以在 master 下），做一些修改。

```bash
~  git clone https://github.com/beepony/bootstrap.git
~  cd bootstrap
~  git checkout -b test-pr
~  git add . && git commit -m "test-pr"
~  git push origin test-pr
```

\3. 完成修改之后，回到 test-pr 分支，点击旁边绿色的 Compare & pull request 按钮

4. 添加一些注释信息，确认提交



**同步上游分支**

先查看一下你目前git状态。

```text
 $ git remote -v
 origin  https://github.com/YOUR_USERNAME/YOUR_FORK.git (fetch)
 origin  https://github.com/YOUR_USERNAME/YOUR_FORK.git (push)
```

以上说明: 你的origin分支指向的是你fork的分支。

指定上游地址。

```text
$ git remote add upstream https://github.com/ORIGINAL_OWNER/ORIGINAL_REPOSITORY.git
```


upstream分支指向上游地址，这里的**upstream**名字可以任意指定，只是一般都把上游地址都叫**upstream**。

检查地址是否设置成功。

```text
 $ git remote -v
 origin    https://github.com/YOUR_USERNAME/YOUR_FORK.git (fetch)
 origin    https://github.com/YOUR_USERNAME/YOUR_FORK.git (push)
 upstream  https://github.com/ORIGINAL_OWNER/ORIGINAL_REPOSITORY.git (fetch)
 upstream  https://github.com/ORIGINAL_OWNER/ORIGINAL_REPOSITORY.git (push)
```

从upstream分支上拉取最新代码。

```text
 $ git fetch upstream
 remote: Counting objects: 75, done.
 remote: Compressing objects: 100% (53/53), done.
 remote: Total 62 (delta 27), reused 44 (delta 9)
 Unpacking objects: 100% (62/62), done.
 From https://github.com/ORIGINAL_OWNER/ORIGINAL_REPOSITORY
  * [new branch]      master     -> upstream/master
```

这里可能会出现错误

```
    error: RPC failed; curl 18 transfer closed with outstanding read data remaining
    fatal: The remote end hung up unexpectedly
    fatal: early EOF
    fatal: index-pack failed
```

需要增大缓冲区大小。只需要输入下面命令即可

```
git config --global http.postBuffer 524288000
```



切换到自己的master上，然后把upstream分支上更新内容合并到master上。

```text
 $ git checkout master
 Switched to branch 'master'
 $ git merge upstream/master
 Updating a422352..5fdff0f
 Fast-forward
  README                    |    9 -------
  README.md                 |    7 ++++++
  2 files changed, 7 insertions(+), 9 deletions(-)
  delete mode 100644 README
  create mode 100644 README.md
```

这时你的本地master就和上游同步成功啦，但是这只是表示你的本地master，一般你还需要把本地master同步到GitHub下的远程分支上。

```text
  $ git  push origin master
```



##connection timed out

报错：

ssh: connect to host github.com port 22: connection timed out

fatal: Could not read from remote repositoty.

Please make sure you have the correct access rights and the repository exists.

试了网上的各种方法（包括添加config文件，本质上是修改port为443，以及重新生成SSH key，以及添加http的远程仓库地址，都没有解决）

其实本质上是网络问题，虽然网页端可以打开github.com，但是本地无法ping通github.com。

解决方法是修改hosts文件，添加github.com的dns解析地址，使得可以跳过 DNS 解析直接访问域名对应 IP 地址。

感谢吴怡的帮助！

参考资料：https://zhuanlan.zhihu.com/p/107334179

