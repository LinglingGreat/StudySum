


# 配置相关

[配置同时使用 Gitlab、Github、Gitee(码云) 共存的开发环境](https://www.jianshu.com/p/68578d52470c "配置同时使用 Gitlab、Github、Gitee(码云) 共存的开发环境")

[配置多个Git账号（windows 10）](https://blog.csdn.net/q13554515812/article/details/83506172 "配置多个Git账号（windows 10）")
  
```bash

ssh-keygen -t ed25519 -C "your_email@example.com"

cat ~/.ssh/id_ed25519.pub

拷贝到SSH公钥

```

  
## 仓库建立

```bash

# 全局配置

git config --global user.name "user"

git config --global user.email "git邮箱"

#如果你公司的项目是放在自建的gitlab上面, 如果你不进行配置用户名和邮箱的话, 则会使用全局的, 正确的做法是针对公司的项目, 在项目根目录下进行单独配置

git config user.name "gitlab's Name"

git config user.email "gitlab@xx.com"

#查看当前配置, 在当前项目下面查看的配置是全局配置+当前项目的配置, 使用的时候会优先使用当前项目的配置

git config --list

  

git init

# 如果报错fatal: remote origin already exists.

# git remote set-url origin git@github.com:ppreyer/first_app.git

git remote add origin git@github.com:ppreyer/first_app.git

git add .

git commit -m 'commit message'

git push -u origin master

#如果报错src refspec master does not match any.

#git push -u origin main

```

  

## 设置 Git 短命令

方式一：`git config --global alias.ps push`

方式二：

打开全局配置文件`vim ~/.gitconfig`
  
写入内容
```bash

[alias]

co = checkout

ps = push

pl = pull

mer = merge --no-ff

cp = cherry-pick

```

## git-lfs
安装
```
wget https://github.com/git-lfs/git-lfs/releases/download/v3.2.0/git-lfs-linux-amd64-v3.2.0.tar.gz 
tar -xzf git-lfs-linux-amd64-v3.2.0.tar.gz PATH=$PATH:/export/fs04/a12/rhuang/git-lfs-3.2.0/  写入配置文件bashrc
git lfs install 
git lfs version
```
  

# Git 分支

`git fetch --all`同步所有信息，但是不会修改代码

1.查看所有分支`git branch -a` （看看是否连接上远程的git）;

2.创建分支`git branch xxx`（为你的分支起名字）

3.切换分支`git checkout xxx`（切换到你创建的分支，xxx为你要切换分支的名字）

4.添加修改代码到缓存（注意最后的"."前面有个空格）`git add .`

5.添加提交代码的备注`git commit -m "xxx"`（xxx为本次提交代码的备注）
	注意：`git commit -a -m "message"` 只提交修改和删除的文件，不包括未跟踪的文件

6.提交代码到指定的分支`git push origin xxx` （xxx为要提交代码的分支名称）(可以不用提交，如果是临时分支的话)

7.如果git push这个步骤出现了错误，是因为是git服务器中的你提交的文件不在本地代码目录中，可以通过如下命令进行代码合并，然后在使用第6步
`git pull --rebase origin xxx`（xxx为要提交代码的分支名称）

8.将分支合并到当前分支`git merge 分支名`

1.删除远程你所创建的分支
`git push origin --delete xxx`（xxx为你想删除的远程分支名称）
2.删除本地分支
`git branch -D xxx`（xxx为你想删除的本地分支名称）
**如果提示你无法删除本地分支，那是因为你目前还在当前分支，切换一下分支就好了**

**分支A的文件覆盖分支B**

`git checkout A xxx xxx`（可以写多个文件，也可以写文件夹）
假如你需要用A分支的**XXX文件**覆盖B分支的**XXX文件**。需要先切换到B分支，然后执行上面的命令。
也就是说要先切换到需要修改的分支然后，执行git checkout A XXX文件路径 来覆盖对应文件。覆盖完成以后执行提交命令提交即可。

一、查看远程分支  
使用如下git命令查看所有远程分支：·`git branch -r

查看远程和本地所有分支：`git branch -a

查看本地分支：`git branch `

在输出结果中，前面带* 的是当前分支。

二、拉取远程分支并创建本地分支  
方法一  
使用如下命令：```git checkout -b 本地分支名x origin/远程分支名x```

使用该方式会在本地新建分支x，并自动切换到该本地分支x。

采用此种方法建立的本地分支会和远程分支建立映射关系。

方式二  
使用如下命令：```git fetch origin 远程分支名x:本地分支名x```

使用该方式会在本地新建分支x，但是不会自动切换到该本地分支x，需要手动checkout。

采用此种方法建立的本地分支不会和远程分支建立映射关系。

三、本地分支和远程分支建立映射关系的作用  
建立本地分支与远程分支的映射关系（或者为跟踪关系track）。  
这样使用git pull或者git push时就不必每次都要指定从远程的哪个分支拉取合并和推送到远程的哪个分支了。```git branch -vv ```

上面的本地分支和远程分支都有映射关系，如果没有，就需要手动建立：

```
git branch -u origin/分支名， 
或者 
git branch --set-upstream-to origin/分支名 
```

origin 为git地址的标志，可以建立当前分支与远程分支的映射关系。

撤销本地分支与远程分支的映射关系`git branch --unset-upstream 
`

之后可以再次用git branch -vv 查看本地分支和远程分支映射关系


[Git 分支 - 分支的新建与合并](https://git-scm.com/book/zh/v2/Git-%E5%88%86%E6%94%AF-%E5%88%86%E6%94%AF%E7%9A%84%E6%96%B0%E5%BB%BA%E4%B8%8E%E5%90%88%E5%B9%B6 "Git 分支 - 分支的新建与合并")

# git操作

## 同步远程已删除分支

有时候一些分支在远程已经删除了，但是使用git branch -a（用来查看所有的分支，包括本地和远程的）仍然可以看见已经被删除的分支

`git remote show origin`查看关于origin的一些信息，可以查看remote地址，远程分支，还有本地分支与之相对应关系(包括分支是否tracking)等信息

通过`git remote prune origin` 移除那些远程仓库不存在的分支

如果远程主机删除了某个分支，默认情况下，git pull 在拉取远程分支的时候，不会删除对应的本地分支。以防其他人操作了远程主机，导致git pull不知不觉删除了本地分支。但是，我们可以改变这个行为，加上参数 -p 就会在本地删除远程已经删除的分支.`git pull -p`

## 同步远程仓库的所有分支

`git clone`克隆远程仓库默认是只克隆`master`分支，当想把远程仓库上的所有的分支都克隆下来的话

```shell
git clone git地址
cd 目录
git branch -r | grep -v '\->' | while read remote; do git branch --track "${remote#origin/}" "$remote"; done
git fetch --all
git pull --all
```

## 不同的远程仓库可以跟踪不同的仓库地址
在此之前需要给github和gitlab都设置key，命名不一样。然后在.ssh文件夹里创建config文件，内容如下

```
Host github.com
    HostName github.com
    User xxx
    IdentityFile ~/.ssh/id_rsa_github
```

在 Git 中，你可以将多个不同的远程仓库关联到同一个本地仓库上，以便于从多个远程仓库拉取代码或将代码推送到多个远程仓库。

你可以使用 `git remote add` 命令将多个远程仓库添加到本地仓库中。例如，假设你有一个 GitHub 上的远程仓库和一个 GitLab 上的远程仓库，你可以使用以下命令将这两个远程仓库添加到你的本地仓库中：

`git remote add github https://github.com/username/repo.git 
`git remote add gitlab https://gitlab.com/username/repo.git`

上述命令将 GitHub 远程仓库命名为 `github`，GitLab 远程仓库命名为 `gitlab`。

如果你想查看目前本地仓库中所有的远程仓库名称及其对应的 URL，可以使用以下命令：

`git remote -v`

此外，你在从远程仓库拉取代码或将代码推送到远程仓库的时候，需要指定远程仓库的名称。例如，如果你要从 GitHub 远程仓库拉取分支 `main` 上的代码到本地仓库中，并且在将新修改推送到 GitHub 远程仓库时使用 `github` 远程仓库，可以使用以下命令：

`git pull github main git push github`

类似地，如果你要从 GitLab 远程仓库拉取分支 `develop` 上的代码到本地仓库中，并且在将新修改推送到 GitLab 远程仓库时使用 `gitlab` 远程仓库，可以使用以下命令：

`git pull gitlab develop git push gitlab`

总之，你可以为每个远程仓库指定不同的地址，以实现多个远程仓库的跟踪。



## 同步最新代码

**1.git pull：获取最新代码到本地，并自动合并到当前分支**

```bash

//查询当前远程的版本

$ git remote -v

//直接拉取并合并最新代码

$ git pull origin master [示例1：拉取远端origin/master分支并合并到当前分支]

$ git pull origin dev [示例2：拉取远端origin/dev分支并合并到当前分支]

```

[Git将master最新代码拉取到当前开发分支](https://www.cnblogs.com/keenajiao/p/16444063.html)

假设你正在开发一个新功能，还没开发完成。但是团队成员A最近开发了B功能，这个功能最近上线后合并到master了，此时你要拉取master最新代码到你的分支中。

1. 切换到master主分支上 `git checkout master`

2. 将master更新的代码拉取到本地 `git pull`

3. 再切换到自己的分支假设为： add_order上 `git checkout add_order`

4. 合并master到自己的分支add_order上  `git merge master`

5. 提交合并后的代码

git add .  
git commit -m "merge master"

 6. 提交到远程仓库  `git push origin add_order`


**2.git fetch + merge: 获取最新代码到本地，然后手动合并分支**

2.1.额外建立本地分支

```bash

//查看当前远程的版本

$ git remote -v

//获取最新代码到本地临时分支(本地当前分支为[branch]，获取的远端的分支为[origin/branch])

$ git fetch origin master:master1 [示例1：在本地建立master1分支，并下载远端的origin/master分支到master1分支中]

$ git fetch origin dev:dev1[示例1：在本地建立dev1分支，并下载远端的origin/dev分支到dev1分支中]

//查看版本差异

$ git diff master1 [示例1：查看本地master1分支与当前分支的版本差异]

$ git diff dev1 [示例2：查看本地dev1分支与当前分支的版本差异]

//合并最新分支到本地分支

$ git merge master1 [示例1：合并本地分支master1到当前分支]

$ git merge dev1 [示例2：合并本地分支dev1到当前分支]

//删除本地临时分支

$ git branch -D master1 [示例1：删除本地分支master1]

$ git branch -D dev1 [示例1：删除本地分支dev1]

```

2.2.不额外建立本地分支

```bash

//查询当前远程的版本

$ git remote -v

//获取最新代码到本地(本地当前分支为[branch]，获取的远端的分支为[origin/branch])

$ git fetch origin master [示例1：获取远端的origin/master分支]

$ git fetch origin dev [示例2：获取远端的origin/dev分支]

//查看版本差异

$ git log -p master..origin/master [示例1：查看本地master与远端origin/master的版本差异]

$ git log -p dev..origin/dev [示例2：查看本地dev与远端origin/dev的版本差异]

//合并最新代码到本地分支

$ git merge origin/master [示例1：合并远端分支origin/master到当前分支]

$ git merge origin/dev [示例2：合并远端分支origin/dev到当前分支]

```

**3.git 放弃本地修改，强制拉取更新（可以先git stash暂存）**
```git

git fetch --all

git reset --hard origin/master

```

可以直接使用命令    **git reset HEAD**这个是整体回到上次一次操作

**绿字变红字(撤销add)** 如果是某个文件回滚到上一次操作：  \*\*git reset HEAD \*\*


**文件名红字变无 (撤销没add修改)git checkout -- 文件**


强制push到远程，忽视远程的修改：`git push -f origin master`


## git撤销commit和add操作
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

撤销刚刚的add：

`git reset --hard` 能让 commit 记录强制回溯到某一个节点。
* `–mixed` 意思是：不删除工作空间改动代码，撤销commit，并且撤销git add . 操作;

这个为默认参数,git reset --mixed HEAD^ 和 `git reset HEAD^` 效果是一样的。

* `–soft` 不删除工作空间改动代码，撤销commit，不撤销git add .常用：`git reset --soft HEAD^`（HEAD^的意思是上一个版本，也可以写成HEAD~1 ；如果你进行了2次commit，想都撤回，可以使用HEAD~2）

* `–hard` 删除工作空间改动代码，撤销commit，撤销git add .注意完成这个操作后，就恢复到了上一次的commit状态。`git reset HEAD`

* 对于已经 push 的 commit，也可以使用该命令，不过再次 push 时，由于远程分支和本地分支有差异，需要强制推送 `git push -f` 来覆盖被 reset 的 commit。


参考

https://blog.csdn.net/tyro_java/article/details/53440666

http://www.liuxiao.org/2017/02/git-%E5%A4%84%E7%90%86-github-%E4%B8%8D%E5%85%81%E8%AE%B8%E4%B8%8A%E4%BC%A0%E8%B6%85%E8%BF%87-100mb-%E6%96%87%E4%BB%B6%E7%9A%84%E9%97%AE%E9%A2%98/

## 删除错误添加到暂存区的文件

仅仅删除暂存区里的文件

```
git rm --cache 文件名
```

删除暂存区和工作区的文件

```
git rm -f 文件名
```


## git diff

git diff 命令比较文件的不同，即比较文件在暂存区和工作区的差异。

git diff 命令显示已写入暂存区和已经被修改但尚未写入暂存区文件的区别。

git diff 有两个主要的应用场景。

* 尚未缓存的改动：**git diff**

* 查看已缓存的改动： **git diff --cached**

* 查看已缓存的与未缓存的所有改动：**git diff HEAD**

* 显示摘要而非整个 diff：**git diff --stat**

显示暂存区和工作区的差异:`$ git diff [file]`

显示暂存区和上一次提交(commit)的差异:`git diff --cached [file] 或 $ git diff --staged [file]`

1.比较两次commit提交之后的差异：`git diff hash1 hash2 --stat`

能够查看出两次提交之后，文件发生的变化。

2.具体查看两次commit提交之后某文件的差异：`git diff hash1 hash2 -- 文件名`

3.比较两个分支的所有有差异的文件的详细差异：`git diff branch1 branch2`

4.比较两个分支的指定文件的详细差异`git diff branch1 branch2 文件名(带路径)`

5.比较两个分支的所有有差异的文件列表`git diff branch1 branch2 --stat`

## revert恢复相关提交引入的更改


给定一个或多个现有提交，恢复相关提交引入的更改，并记录一些这些更改的新提交。这就要求你的工作树是干净的（没有来自头部的修改）。

`git revert commithash` revert掉自己提交的commit。

因为 revert 会生成一条新的提交记录，这时会让你编辑提交信息，编辑完后 :wq 保存退出就好了。

## reflog误操作后找回记录

如果说 `reset --soft` 是后悔药，那 reflog 就是强力后悔药。它记录了所有的 commit 操作记录，便于错误操作后找回记录。

应用场景：某天你眼花，发现自己在其他人分支提交了代码还推到远程分支，这时因为分支只有你的最新提交，就想着使用 `reset --hard`，结果紧张不小心记错了 commitHash，reset 过头，把同事的 commit 搞没了。

这时用 `git reflog` 查看历史记录，把错误提交的那次 commitHash 记下。

再次 reset 回去，就会发现 b 回来了。

## stash保存本地修改，返回工作目录

当您想记录工作目录和索引的当前状态，但又想返回一个干净的工作目录时，请使用git stash。该命令将保存本地修改，并恢复工作目录以匹配头部提交。

应用场景：某一天你正在 feature 分支开发新需求，突然产品经理跑过来说线上有bug，必须马上修复。只需要`git stash`，代码就被存起来了。当你修复完线上问题，切回 feature 分支，想恢复代码也只需要：`git stash apply`

```bash

# 保存当前未commit的代码

git stash

# 保存当前未commit的代码并添加备注

git stash save "备注的内容"

# 列出stash的所有记录

git stash list

# 删除stash的所有记录

git stash clear

# 应用最近一次的stash

git stash apply

# 应用最近一次的stash，随后删除该记录

git stash pop

# 删除最近的一次stash

git stash drop

复制代码

```


当有多条 stash，可以指定操作stash，首先使用stash list 列出所有记录，

```bash

$ git stash list

stash@{0}: WIP on ...

stash@{1}: WIP on ...

stash@{2}: On ...

```

应用第二条记录：`git stash apply stash@{1}`

pop，drop 同理。

## cherry-pick复制commit

将已经提交的 commit，复制出新的 commit 应用到分支里

应用场景1：有时候版本的一些优化需求开发到一半，可能其中某一个开发完的需求要临时上，或者某些原因导致待开发的需求卡住了已开发完成的需求上线。这时候就需要把 commit 抽出来，单独处理。

应用场景2：有时候开发分支中的代码记录被污染了，导致开发分支合到线上分支有问题，这时就需要拉一条干净的开发分支，再从旧的开发分支中，把 commit 复制到新分支。

**复制单个**

```bash

# 在分支查看commit记录，复制要操作的commitHash

git log

# 切到master分支，使用cherry-pick 把 b 应用到当前分支。

git cherry-pick commithash

# 查看记录

git log

```

**复制多个**
* 一次转移多个提交：`git cherry-pick commit1 commit2`
* 多个连续的commit，也可区间复制：`git cherry-pick commit1^..commit2`。上面的命令将 commit1 到 commit2 这个区间的 commit 都应用到当前分支（包含commit1、commit2），commit1 是最早的提交。

代码冲突

* 在 `cherry-pick` 多个commit时，可能会遇到代码冲突，这时 `cherry-pick` 会停下来，让用户决定如何继续操作。

* 这时需要解决代码冲突，重新提交到暂存区。
* 然后使用 `cherry-pick --continue` 让 `cherry-pick` 继续进行下去。
以上是完整的流程，但有时候可能需要在代码冲突后，放弃或者退出流程：

放弃：`gits cherry-pick --abort`，回到操作前的样子，就像什么都没发生过。

退出：`git cherry-pick --quit`，不回到操作前的样子。即保留已经 `cherry-pick` 成功的 commit，并退出 `cherry-pick` 流程。

# gitignore用法

`文件夹名称/` 是忽略所有该文件夹名称的文件夹

[https://www.cnblogs.com/kevingrace/p/5690241.html](https://www.cnblogs.com/kevingrace/p/5690241.html "https://www.cnblogs.com/kevingrace/p/5690241.html")

git取消跟踪文件目录：[<https://blog.csdn.net/sun2009> \_/article/details/70198580](https://blog.csdn.net/sun2009_/article/details/70198580 "https://blog.csdn.net/sun2009_/article/details/70198580")

首先 `git rm -r -n --cached 文件/目录` 列出你需要取消跟踪的文件，可以查看列表，检查下是否有误操作导致一些不应该被取消的文件取消了，是为了再次确认的。 &#x20;

`git rm -r --cached 文件/目录` 才是真正的取消缓存不想要跟踪的文件

# fork别人的项目

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

  

# 常见问题

### git 无法添加文件夹下的文件

目录下含有.git文件夹，删除该文件

git rm --cached folder
git add folder

### 大文件

[.git文件过大！删除大文件](https://blog.csdn.net/gaojy19881225/article/details/81121643 ".git文件过大！删除大文件")

[git filter-branch 从历史仓库中删除文件夹](https://blog.csdn.net/a0936/article/details/114922501 "git filter-branch 从历史仓库中删除文件夹")

永久删除history中的大文件，达到让.git文件瘦身的目的。

会同时删除本地的文件，谨慎！！！
```bash

git rev-list --objects --all | grep "$(git verify-pack -v .git/objects/pack/*.idx | sort -k 3 -n | tail -15 | awk '{print$1}')"

# 第一行的字母其实相当于文件的id,用以下命令可以找出id 对应的文件名
git rev-list --objects --all | grep id

# 文件夹BlazorApp/obj

git filter-branch --tree-filter 'rm -rf BlazorApp/obj' --tag-name-filter cat -- --all

# 文件

git filter-branch --index-filter 'git rm --cached --ignore-unmatch <your-file-name>'

  

# 回收内存

rm -rf .git/refs/original/

git reflog expire --expire=now --all

git fsck --full --unreachable

git repack -A -d

git gc --aggressive --prune=now

  

# 提交

git push --force --all

  

```


### git项目合并与拆分

拆分：[https://www.jianshu.com/p/a107f2eaa1d6](https://www.jianshu.com/p/a107f2eaa1d6 "https://www.jianshu.com/p/a107f2eaa1d6")

![](img/Pasted%20image%2020220802142501.png)


[gitlab两个项目代码合并](https://blog.csdn.net/qq_34642406/article/details/112830174 "gitlab两个项目代码合并")


### 报错：! \[remote rejected] master -＞ master (pre-receive hook declined)

* 原因：添加了较大的文件，撤销commit和add就可以了不允许上传超过100MB文件的问题

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



## connection timed out

报错：

ssh: connect to host github.com port 22: connection timed out

fatal: Could not read from remote repositoty.

Please make sure you have the correct access rights and the repository exists.

试了网上的各种方法（包括添加config文件，本质上是修改port为443，以及重新生成SSH key，以及添加http的远程仓库地址，都没有解决）

其实本质上是网络问题，虽然网页端可以打开github.com，但是本地无法ping通github.com。

解决方法是修改hosts文件，添加github.com的dns解析地址，使得可以跳过 DNS 解析直接访问域名对应 IP 地址。

参考资料：https://zhuanlan.zhihu.com/p/107334179

