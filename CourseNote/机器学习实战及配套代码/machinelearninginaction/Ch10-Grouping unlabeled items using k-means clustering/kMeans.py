'''
Created on Feb 16, 2011
k Means Clustering for Ch10 of Machine Learning in Action
@author: Peter Harrington
'''
from numpy import *

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine)) #map all elements to float()
        dataMat.append(fltLine)
    return dataMat

def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2))) #la.norm(vecA-vecB)

def randCent(dataSet, k):
    n = shape(dataSet)[1]   # 特征个数
    centroids = mat(zeros((k, n)))#create centroid mat
    for j in range(n):#create random cluster centers, within bounds of each dimension
        minJ = min(dataSet[:,j]) 
        rangeJ = float(max(dataSet[:,j]) - minJ)
        centroids[:,j] = mat(minJ + rangeJ * random.rand(k,1))   # 生成这个维度的k个点
    return centroids

# datMat = mat(loadDataSet('testSet.txt'))
# print(randCent(datMat, 2))
# print(distEclud(datMat[0], datMat[1]))

    
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]    # 样例个数
    clusterAssment = mat(zeros((m,2)))  # create mat to assign data points
                                       # to a centroid, also holds SE of each point
    centroids = createCent(dataSet, k)    # 产生k个初始质心
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):  # for each data point assign it to the closest centroid
            minDist = inf; minIndex = -1
            # 对每个数据点，计算它到每个质心的距离，取最小的那个
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            if clusterAssment[i,0] != minIndex: clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2    # 该样例所属簇的索引及误差
        print(centroids)
        for cent in range(k):  # recalculate centroids
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]#get all the point in this cluster
            centroids[cent,:] = mean(ptsInClust, axis=0) #assign centroid to mean 
    return centroids, clusterAssment

# datMat = mat(loadDataSet('testSet.txt'))
# myCentroids, clustAssing = kMeans(datMat, 4)


def biKmeans(dataSet, k, distMeas=distEclud):
    """
    二分K-均值聚类算法
    :param dataSet:
    :param k:
    :param distMeas:
    :return:
    """
    m = shape(dataSet)[0]   # 样例个数
    clusterAssment = mat(zeros((m,2)))
    centroid0 = mean(dataSet, axis=0).tolist()[0]  # 沿矩阵的列方向进行均值计算
    centList =[centroid0]  # create a list with one centroid
    for j in range(m):  # calc initial Error
        clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j,:])**2
    while (len(centList) < k):
        lowestSSE = inf
        for i in range(len(centList)):
            # get the data points currently in cluster i
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)  # 分成两类
            # 划分的数据集的误差与剩余未被划分的数据集的误差之和作为总误差
            sseSplit = sum(splitClustAss[:,1])
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])
            print("sseSplit, and notSplit: ",sseSplit,sseNotSplit)
            # 选择使得总误差最小的那个簇进行划分操作
            # compare the SSE to the currrent minimum
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        # 分配所有点的簇结果
        # KMeans()函数得到编号为0和1的簇，修改编号，改为划分簇及新加簇的编号
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList)  # change 1 to 3,4, or whatever
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
        print('the bestCentToSplit is: ',bestCentToSplit)
        print('the len of bestClustAss is: ', len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]  # replace a centroid with two best centroids
        centList.append(bestNewCents[1,:].tolist()[0])
        # reassign new clusters, and SSE
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss
    return mat(centList), clusterAssment

# datMat3 = mat(loadDataSet('testSet2.txt'))
# centList, myNewAssments = biKmeans(datMat3, 3)
# print(centList)

import urllib
import urllib.request
import json
def geoGrab(stAddress, city):
    apiStem = 'http://where.yahooapis.com/geocode?'  #create a dict and constants for the goecoder
    params = {}
    params['flags'] = 'J'#JSON return type
    params['appid'] = 'aaa0VN6k'
    params['location'] = '%s %s' % (stAddress, city)
    url_params = urllib.parse.urlencode(params)
    yahooApi = apiStem + url_params      #print url_params
    print(yahooApi)
    c=urllib.request.urlopen(yahooApi)
    return json.loads(c.read())

from time import sleep
def massPlaceFind(fileName):
    fw = open('places.txt', 'w')
    for line in open(fileName).readlines():
        line = line.strip()
        lineArr = line.split('\t')
        retDict = geoGrab(lineArr[1], lineArr[2])
        if retDict['ResultSet']['Error'] == 0:
            lat = float(retDict['ResultSet']['Results'][0]['latitude'])
            lng = float(retDict['ResultSet']['Results'][0]['longitude'])
            print("%s\t%f\t%f" % (lineArr[0], lat, lng))
            fw.write('%s\t%f\t%f\n' % (line, lat, lng))
        else: print("error fetching")
        sleep(1)
    fw.close()


# geoResults = geoGrab('1 VA Center', 'Augusta, ME')
# print(geoResults)
# massPlaceFind('portlandClubs.txt')

    
def distSLC(vecA, vecB):  # Spherical Law of Cosines球面距离计算，地球表面两点之间的距离
    a = sin(vecA[0,1]*pi/180) * sin(vecB[0,1]*pi/180)   # 球面余弦定理
    b = cos(vecA[0,1]*pi/180) * cos(vecB[0,1]*pi/180) * \
                      cos(pi * (vecB[0,0]-vecA[0,0]) /180)
    return arccos(a + b)*6371.0  # pi is imported with numpy

import matplotlib
import matplotlib.pyplot as plt
def clusterClubs(numClust=5):
    datList = []
    for line in open('places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])
    datMat = mat(datList)
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)
    fig = plt.figure()   # 创建一个图
    rect=[0.1,0.1,0.8,0.8]
    scatterMarkers=['s', 'o', '^', '8', 'p', \
                    'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0=fig.add_axes(rect, label='ax0', **axprops)    # 创建一个矩形
    imgP = plt.imread('Portland.png')    # 基于一副图像创建矩阵
    ax0.imshow(imgP)   # 绘制该矩阵
    ax1=fig.add_axes(rect, label='ax1', frameon=False)   # 绘制新的图
    for i in range(numClust):
        print(clustAssing)
        print(clustAssing.A)
        ptsInCurrCluster = datMat[nonzero(clustAssing[:,0].A==i)[0],:]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0], ptsInCurrCluster[:,1].flatten().A[0], marker=markerStyle, s=90)
    ax1.scatter(myCentroids[:,0].flatten().A[0], myCentroids[:,1].flatten().A[0], marker='+', s=300)   # 簇中心
    plt.show()


clusterClubs(5)
