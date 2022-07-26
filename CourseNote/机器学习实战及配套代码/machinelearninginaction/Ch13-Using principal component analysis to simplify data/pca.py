'''
Created on Jun 1, 2011

@author: Peter Harrington
'''
from numpy import *


def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [list(map(float,line)) for line in stringArr]
    return mat(datArr)


def pca(dataMat, topNfeat=9999999):
    """
    去除平均值
    计算协方差矩阵
    计算协方差矩阵的特征值和特征向量
    将特征值从大到小排序
    保留最上面的N个特征向量
    将数据转换到上述N个特征向量构建的新空间中
    :param dataMat: 数据集
    :param topNfeat: 应用的N个特征
    :return:
    """
    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals   # remove mean
    covMat = cov(meanRemoved, rowvar=0)
    eigVals,eigVects = linalg.eig(mat(covMat))
    eigValInd = argsort(eigVals)            # sort, sort goes smallest to largest
    eigValInd = eigValInd[:-(topNfeat+1):-1]  # cut off unwanted dimensions
    redEigVects = eigVects[:,eigValInd]       # reorganize eig vects largest to smallest
    lowDDataMat = meanRemoved * redEigVects   # transform data into new dimensions
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat


def replaceNanWithMean(): 
    datMat = loadDataSet('secom.data', ' ')
    numFeat = shape(datMat)[1]
    for i in range(numFeat):
        meanVal = mean(datMat[nonzero(~isnan(datMat[:,i].A))[0],i])  # values that are not NaN (a number)
        datMat[nonzero(isnan(datMat[:,i].A))[0],i] = meanVal  # set NaN values to mean
    return datMat


# dataMat = loadDataSet('testSet.txt')
# lowDMat, reconMat = pca(dataMat, 1)
# lowDMat, reconMat = pca(dataMat, 2)
# print(shape(lowDMat))
# import matplotlib
# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(dataMat[:, 0].flatten().A[0], dataMat[:, 1].flatten().A[0], marker='^', s=90)
# ax.scatter(reconMat[:, 0].flatten().A[0], reconMat[:, 1].flatten().A[0], marker='o', s=50, c='red')
# plt.show()

dataMat = replaceNanWithMean()
meanVals = mean(dataMat, axis=0)
meanRemoved = dataMat - meanVals  # remove mean
covMat = cov(meanRemoved, rowvar=0)
eigVals, eigVects = linalg.eig(mat(covMat))
print(eigVals)
