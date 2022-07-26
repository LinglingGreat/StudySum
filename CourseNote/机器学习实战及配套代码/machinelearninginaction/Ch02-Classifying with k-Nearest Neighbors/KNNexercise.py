from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt
from os import listdir


""" 创建数据集和标签"""
def createDataSet():
    group = array([[1.0, 1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group,labels


""" k-近邻算法"""
def classify0(inX, dataSet, labels, k):
    """
    对未知类别属性的数据集中的每个点依次执行以下操作：
    1.计算已知类别数据集中的点与当前点之间的距离；
    2.按照距离递增次序排序；
    3.选取与当前点距离最小的k个点；
    4.确定前k个点所在类别的出现频率；
    5.返回前k个点出现频率最高的类别作为当前点的预测分类
    """
    dataSetSize = dataSet.shape[0]    # 获取数据dataSet的行数
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    # 计算输入向量和每个训练样本向量的差
    # inX是用于分类的输入向量
    # tile函数的第二个参数用来控制inX重复次数；参数列表的元组第一个是控制行数的，第二个是控制inX的重复次数
    # tile(a,x): x是控制a重复几次的，结果是一个一维数组
    # tile(a,(x,y))：结果是一个二维矩阵，其中行数为x，列数是一维数组a的长度和y的乘积
    # tile(a,(x,y,z)): 结果是一个三维矩阵，其中矩阵的行数为x，矩阵的列数为y，而z表示矩阵每个单元格里a重复的次数。(三维矩阵可以看成一个二维矩阵，每个矩阵的单元格里存者一个一维矩阵a)
    # print(diffMat)
    # print(inX-dataSet)
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)    # axis=1对矩阵的每一行向量进行相加,axis=0对矩阵的列相加
    # print(sqDistances)
    distances = sqDistances**0.5
    # print(distances)
    sortedDistIndicies = distances.argsort()    # 从小到大排序,返回排序后对应的index
    # print(sortedDistIndicies)
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        # print(sortedDistIndicies[i]) 输出数字
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    # 将classCount字典分解为元组列表，按照第二个元素的次序对元组进行逆序排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)   # sorted后返回一个list
    # print(sortedClassCount) 元组组成的列表
    return sortedClassCount[0][0]

# group, labels = createDataSet()
# print(classify0([0,0], group, labels, 3))


"""将文本记录转换为NumPy，输入文件名字符串，输出训练样本矩阵和类标签向量"""
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    # 创建返回的NumPy矩阵，样本个数作为行数，特征个数作为列数
    returnMat = zeros((numberOfLines,3))
    # print(returnMat)
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()  # 截取掉所有的回车字符
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector


# datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
# print(datingDataMat[0:5])
# print(datingLabels[0:10])
"""图形化展示数据内容"""
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0*array(datingLabels), 15.0*array(datingLabels))
# 这里的后两个参数分别表示散列点的大小和颜色,可选
# scatter（x,y,s=1,c="g",marker="s",linewidths=0）
# s:散列点的大小,c:散列点的颜色，marker：形状，linewidths：边框宽度
# plt.show()


"""归一化特征值[0,1]区间"""
def autoNorm(dataSet):
    minVals = dataSet.min(0)   # 列的最小值
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))
    # 注意这里是具体特征值相除，NumPy中矩阵除法需要使用函数linalg.solve(matA,matB)
    return normDataSet, ranges, minVals

# normMat, ranges, minVals = autoNorm(datingDataMat)
# print(normMat)
# print(ranges)
# print(minVals)

"""分类器针对约会网站的测试代码"""
def datingClassTest():
    hoRatio = 0.10  # 测试集百分比
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount =0.0
    for i in range(numTestVecs):
        # 参数分别是欲分类向量，数据集，数据集标签
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))

# datingClassTest()


"""约会网站预测函数:输入数据，给出预测值"""
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    # 先对数据做归一化处理再进行预测
    classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    # 类别标签是1,2,3
    print("You will probably like this person: ", resultList[classifierResult - 1])

# classifyPerson()


"""手写识别系统"""
"""将图像格式转换为分类器使用的向量格式
将一个32*32的二进制图像矩阵转换为1*1024的向量"""
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    # 循环读出文件的前32行，并将每行的头32个字符值存储在数组中
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect


# testVector = img2vector('testDigits/0_13.txt')
# print(testVector[0, 0:31])


"""手写数字识别系统的测试代码"""
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]  # 获取文件名，去掉后缀
        classNumStr = int(fileStr.split('_')[0])  # 获取类别标签，即文件名的首数字
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr):
            errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount/float(mTest)))

# handwritingClassTest()

