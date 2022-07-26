# -*- coding:utf-8 -*-
'''
Created on Oct 12, 2010
Decision Tree Source Code for Machine Learning in Action Ch. 3
@author: Peter Harrington
'''
from math import log
import operator
import treePlotter

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels

"""计算给定数据集的香农熵"""
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)    # 计算数据集中实例的总数
    # 创建字典，字典的每个键值都记录了当前类别出现的次数
    labelCounts = {}
    for featVec in dataSet: #the the number of unique elements and their occurance
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    # 使用所有类标签的发生频率计算类别出现的概率，用这个概率计算香农熵
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2) #log base 2
    return shannonEnt

def mytest1():
    myDat, labels = createDataSet()
    print(myDat)
    print('Shannon entropy', calcShannonEnt(myDat))
    # 熵越高，混合的数据也越多
    myDat[0][-1] = 'maybe'
    print(myDat)
    calcShannonEnt(myDat)


"""按照给定特征划分数据集
 输入参数分别是：待划分的数据集、划分数据集的特征、需要返回的特征的值
 返回所有符合要求的数据列表，即返回所有第axis个特征值为value的数据"""
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        # 将符合特征的数据抽取出来
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]     #chop out axis used for splitting
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def mytest2():
    myDat, labels = createDataSet()
    print(myDat)
    print(splitDataSet(myDat, 0, 1))
    print(splitDataSet(myDat, 0, 0))

"""找到最好的数据集划分方式"""
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1      #the last column is used for the labels
    baseEntropy = calcShannonEnt(dataSet)   # 整个数据集的原始香农熵
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):        #iterate over all the features
        featList = [example[i] for example in dataSet]#create a list of all the examples of this feature
        uniqueVals = set(featList)       #get a set of unique values
        newEntropy = 0.0
        # 遍历当前特征中的唯一属性值，对每个唯一属性值划分一次数据集，然后计算数据集的新熵值，并对所有唯一特征值得到的熵求和
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)     
        infoGain = baseEntropy - newEntropy     #calculate the info gain; ie reduction in entropy
        if (infoGain > bestInfoGain):       #compare this to the best gain so far
            bestInfoGain = infoGain         #if better than current best, set to best
            bestFeature = i
    return bestFeature                      #returns an integer

def mytest3():
    myDat, labels = createDataSet()
    print(chooseBestFeatureToSplit(myDat))

"""至此都是从数据集构造决策树算法所需要的子功能模块
其工作原理为：得到原始数据集，然后基于最好的属性值划分数据集，由于特征值可能多于两个，因此可能存在大于两个分支的数据集划分
第一次划分之后，数据将被向下传递到树分支的下一个节点，在这个节点上，我们可以再次划分数据
因此我们可以采用递归的原则处理数据集
递归结束的条件是：程序遍历完所有划分数据集的属性，或者每个分支下的所有实例都具有相同的分类。如果所有实例具有相同的分类
则得到一个叶子节点或者终止块。任何到达叶子节点的数据必然属于叶子节点的分类。"""

"""使用分类名称的列表，创建键值为classList中唯一值的数据字典，字典对象存储了classList中每个类标签出现的频率，最后利用
operator操作键值排序字典，并返回出现次数最多的分类名称"""
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

"""创建树，labels包含了数据集中所有特征的标签,给出了数据明确的含义"""
def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    # 类别完全相同则停止继续划分
    if classList.count(classList[0]) == len(classList): 
        return classList[0]    # stop splitting when all of the classes are equal
    # 遍历完所有特征时返回出现次数最多的类别
    if len(dataSet[0]) == 1:     # stop splitting when there are no more features in dataSet
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)    # 选择最好的划分特征(对应的index)
    bestFeatLabel = labels[bestFeat]    # 最好的划分特征对应的名称
    myTree = {bestFeatLabel:{}}    # 使用字典存储树
    del(labels[bestFeat])      # 将该特征从标签列表中去除
    # 得到该特征包含的所有属性值
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    # 遍历当前选择特征包含的所有属性值，在每个数据集划分上递归调用函数创建树
    for value in uniqueVals:
        subLabels = labels[:]       #copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
    return myTree                            

def mytest4():
    myDat, labels = createDataSet()
    myTree = createTree(myDat, labels)
    print("myTree: ", myTree)


"""使用决策树的分类函数"""
def classify(inputTree,featLabels,testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    # 将标签字符串转换为索引,使用index方法查找当前列表中第一个匹配firstStr变量的元素
    featIndex = featLabels.index(firstStr)
    # 递归遍历整棵树，比较testVec变量中的值与树节点的值，如果到达叶子节点，则返回当前节点的分类标签
    # for key in secondDict.keys():
    #     if testVec[featIndex] == key:
    #         if type(secondDict[key]).__name__ == 'dict':
    #             classLabel = classify(secondDict[key], featLabels, testVec)
    #         else: classLabel = secondDict[key]
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict): 
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel


"""存储分类器到硬盘中"""
def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()

"""读取硬盘中的分类器"""
def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)

# mytest1()
# mytest2()
# mytest3()
# mytest4()

# myDat, labels = createDataSet()
# myTree = treePlotter.retrieveTree(0)
# print(classify(myTree, labels, [1,0]))
# print(classify(myTree, labels, [1,1]))

# storeTree(myTree, 'classifierStorage.txt')
# print grabTree('classifierStorage.txt')
    
# 示例：使用决策树预测隐形眼镜类型
# (1)收集数据：提供的文本文件
# (2)准备数据：解析tab键分隔的数据行
# (3)分析数据：快速检查数据，确保正确地解析数据内容，使用createPlot()函数绘制最终的树形图
# (4)训练算法：使用createTree()函数
# (5)测试算法：编写测试函数验证决策树可以正确分类给定的数据实例
# (6)使用算法：存储树的数据结构，以便下次使用时无需重新构造树

# fr = open('lenses.txt')
# lenses = [inst.strip().split('\t') for inst in fr.readlines()]
# lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
# lensesTree = createTree(lenses, lensesLabels)
# print(lensesTree)
# print(treePlotter.createPlot(lensesTree))
# 本章使用的算法称为ID3，可以划分标称型数据集，无法直接处理数值型数据
