# -*- coding:utf-8 -*-
'''
Created on Oct 14, 2010

@author: Peter Harrington
'''
import matplotlib.pyplot as plt

"""使用文本注解绘制树节点"""
# 定义文本框和箭头格式
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

"""获取叶节点的数目"""
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        # 如果子节点是字典类型，则该节点也是一个判断节点，需要递归调用getNumLeafs()
        if type(secondDict[key]).__name__=='dict':    # test to see if the nodes are dictonaires, if not they are leaf nodes
            numLeafs += getNumLeafs(secondDict[key])
        else:   numLeafs +=1
    return numLeafs


"""获取树的层数"""
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':#test to see if the nodes are dictonaires, if not they are leaf nodes
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:   thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth

"""
绘制带箭头的注释
该函数需要一个绘图区，该区域由全局变量createPlot.ax1定义
Python语言中所有的变量默认都是全局有效的
"""
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',
             xytext=centerPt, textcoords='axes fraction',
             va="center", ha="center", bbox=nodeType, arrowprops=arrow_args )


"""
在父子节点间填充文本信息:
计算父节点和子节点的中间位置，并在此处添加简单的文本标签信息
"""
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)


"""绘制树"""
def plotTree(myTree, parentPt, nodeTxt):#if the first key tells you what feat was split on
    # 计算树的宽和高
    numLeafs = getNumLeafs(myTree)  #this determines the x width of this tree
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]     #the text label for this node should be this
    # 全局变量totalW和totalD分别存储树的宽度和深度，使用这两个变量计算树节点的摆放位置
    # 树的宽度用于计算放置判断节点的位置，主要的计算原则是将它放在所有叶子节点的中间，而不仅仅是它子节点的中间
    # 全局变量plotTree.xOff和plotTree.yOff用于追踪已经绘制的节点位置，以及放置下一个节点的恰当位置
    # 绘制图形的x轴有效范围是0.0到1.0, y轴有效范围也是
    # 按照叶子节点的数目将x轴划分为若干部分，按照图形比例绘制树形图无需关心实际输出图形的大小，一旦图形大小发生变化，
    # 函数会自动按照图形大小重新绘制。如果以像素为单位绘制图形，则缩放图形就不是一件简单的事
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    # 绘制子节点具有的特征值，或者沿此分支向下的数据实例必须具有的特征值
    # 计算父节点和子节点的中间位置，并在此处添加简单的文本标签信息
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    # 按比例减少全局变量plotTree.yOff,并标注此处将要绘制子节点，这些节点既可以是叶子节点也可以是判断节点
    # 自顶向下绘图，所以需要依次递减y坐标值
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':#test to see if the nodes are dictonaires, if not they are leaf nodes   
            plotTree(secondDict[key],cntrPt,str(key))        #recursion
        else:   #it's a leaf node print the leaf node
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD
#if you do get a dictonary you know it's a tree, and the first element will be another dict


"""创建绘图区，计算树形图的全局尺寸，并调用递归函数plotTree()"""
def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)    #no ticks
    #createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses 
    plotTree.totalW = float(getNumLeafs(inTree))   # 全局变量，存储树的宽度
    plotTree.totalD = float(getTreeDepth(inTree))   # 全局变量，存储树的深度
    # 全局变量，追踪已经绘制的节点位置，以及放置下一个节点的恰当位置
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;
    plotTree(inTree, (0.5,1.0), '')
    plt.show()


"""第一个版本的createPlot, 首先创建一个新图形并清空绘图区，然后在绘图区上绘制两个代表不同类型的树节点"""
def createPlotv1():
   fig = plt.figure(1, facecolor='white')
   fig.clf()
   createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses
   plotNode('a decision node', (0.5, 0.1), (0.1, 0.5), decisionNode)
   plotNode('a leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
   plt.show()


"""输出预先存储的树信息"""
def retrieveTree(i):
    listOfTrees =[{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                  {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                  ]
    return listOfTrees[i]

# createPlotv1()
# createPlot(thisTree)


# myTree = retrieveTree(0)
# print(getNumLeafs(myTree))
# print(getTreeDepth(myTree))

# myTree = retrieveTree(0)
# print(myTree)
# createPlot(myTree)
# myTree['no surfacing'][3] = 'maybe'
# print(myTree)
# createPlot(myTree)
