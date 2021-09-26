'''
Created on Jan 8, 2011

@author: Peter
'''
from numpy import *

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) - 1 #get number of fields 
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

def standRegres(xArr,yArr):
    xMat = mat(xArr); yMat = mat(yArr).T
    xTx = xMat.T*xMat
    if linalg.det(xTx) == 0.0:   # 计算行列式
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T*yMat)   # 最小二乘法(OLS)解出的最优w, 目标函数是最小化sum(yi-xi^T*w)^2
    # ws = linalg.solve(xTx, xMat.T*yMatT)   # 求解未知矩阵的函数
    return ws

# xArr, yArr = loadDataSet('ex0.txt')
# print(xArr[0:2])
# ws = standRegres(xArr, yArr)
# print(ws)
# xMat = mat(xArr)
# yMat = mat(yArr)
# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(xMat[:,1].flatten().A[0], yMat.T[:,0].flatten().A[0])
# # 如果直线上的数据点次序混乱，绘图时将会出现问题，所以要先升序排列
# xCopy = xMat.copy()
# xCopy.sort(0)
# yHat = xCopy * ws
# ax.plot(xCopy[:,1], yHat)
# plt.show()
# 评估模型好坏，可以计算预测值和真实值的相关系数
# yHat = xMat*ws
# print(corrcoef(yHat.T, yMat))


"""局部加权线性回归(Locally Weighted Linear Regression, LWLR)函数"""
def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat = mat(xArr); yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))   # 创建对角权重矩阵w
    for j in range(m):                      #next 2 lines create weights matrix
        diffMat = testPoint - xMat[j,:]     # 计算样本点与待预测点的距离
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))   # 高斯核
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

def lwlrTest(testArr,xArr,yArr,k=1.0):  #loops over all the data points and applies lwlr to each one
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat

# xArr, yArr = loadDataSet('ex0.txt')
# print(yArr[0])
# print(lwlr(xArr[0],xArr,yArr,1.0))
# print(lwlr(xArr[0],xArr,yArr,0.001))
# yHat = lwlrTest(xArr, xArr,yArr,0.003)
# # 绘图
# xMat = mat(xArr)
# srtInd = xMat[:,1].argsort(0)
# xSort = xMat[srtInd][:,0,:]
# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(xSort[:,1],yHat[srtInd])
# ax.scatter(xMat[:,1].flatten().A[0], mat(yArr).T.flatten().A[0], s=2, c='red')
# plt.show()


def lwlrTestPlot(xArr,yArr,k=1.0):  #same thing as lwlrTest except it sorts X first
    yHat = zeros(shape(yArr))       #easier for plotting
    xCopy = mat(xArr)
    xCopy.sort(0)
    for i in range(shape(xArr)[0]):
        yHat[i] = lwlr(xCopy[i],xArr,yArr,k)
    return yHat,xCopy

def rssError(yArr,yHatArr): #yArr and yHatArr both need to be arrays
    return ((yArr-yHatArr)**2).sum()

# 预测鲍鱼的年龄
# abX, abY = loadDataSet('abalone.txt')
# yHat01 = lwlrTest(abX[0:99], abX[0:99], abY[0:99],0.1)
# yHat1 = lwlrTest(abX[0:99], abX[0:99], abY[0:99],1)
# yHat10 = lwlrTest(abX[0:99], abX[0:99], abY[0:99],10)
# print(rssError(abY[0:99],yHat01.T))
# print(rssError(abY[0:99],yHat1.T))
# print(rssError(abY[0:99],yHat10.T))
# yHat01 = lwlrTest(abX[100:199],abX[0:99],abY[0:99],0.1)
# print(rssError(abY[100:199],yHat01.T))
# yHat1 = lwlrTest(abX[100:199],abX[0:99],abY[0:99],1)
# print(rssError(abY[100:199],yHat1.T))
# yHat10 = lwlrTest(abX[100:199],abX[0:99],abY[0:99],10)
# print(rssError(abY[100:199],yHat10.T))
# ws = standRegres(abX[0:99],abY[0:99])
# yHat = mat(abX[100:199])*ws
# print(rssError(abX[100:199],yHat.A))

"""岭回归"""
def ridgeRegres(xMat,yMat,lam=0.2):
    xTx = xMat.T*xMat
    denom = xTx + eye(shape(xMat)[1])*lam
    if linalg.det(denom) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = denom.I * (xMat.T*yMat)   # w = (xT*x+lam*I)^(-1)*xT*y
    return ws
    
def ridgeTest(xArr,yArr):
    xMat = mat(xArr); yMat=mat(yArr).T
    yMean = mean(yMat,0)
    yMat = yMat - yMean     #to eliminate X0 take mean off of Y
    #regularize X's
    xMeans = mean(xMat,0)   #calc mean then subtract it off
    xVar = var(xMat,0)      #calc variance of Xi then divide by it
    xMat = (xMat - xMeans)/xVar
    numTestPts = 30
    wMat = zeros((numTestPts,shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat,yMat,exp(i-10))
        wMat[i,:]=ws.T
    return wMat

# abX,abY = loadDataSet('abalone.txt')
# ridgeWeights = ridgeTest(abX,abY)
# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(ridgeWeights)
# plt.show()

def regularize(xMat):#regularize by columns
    inMat = xMat.copy()
    inMeans = mean(inMat,0)   #calc mean then subtract it off
    inVar = var(inMat,0)      #calc variance of Xi then divide by it
    inMat = (inMat - inMeans)/inVar
    return inMat

"""前向逐步回归
数据标准化，使其分布满足0均值和单位方差
在每轮迭代过程中：
    设置当前最小误差lowestError为正无穷
    对每个特征：
    增大或缩小：
        改变一个系数得到一个新的W
        计算新W下的误差
        如果误差Error小于当前最小误差lowestError:设置Wbest等于当前的W
    将W设置为新的Wbest"""
def stageWise(xArr,yArr,eps=0.01,numIt=100):
    xMat = mat(xArr); yMat=mat(yArr).T
    yMean = mean(yMat,0)
    yMat = yMat - yMean     #can also regularize ys but will get smaller coef
    xMat = regularize(xMat)
    m,n=shape(xMat)
    returnMat = zeros((numIt,n)) #testing code remove
    ws = zeros((n,1)); wsTest = ws.copy(); wsMax = ws.copy()
    for i in range(numIt):
        print(ws.T)
        lowestError = inf; 
        for j in range(n):
            for sign in [-1,1]:
                wsTest = ws.copy()
                wsTest[j] += eps*sign
                yTest = xMat*wsTest
                rssE = rssError(yMat.A,yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i,:]=ws.T
    return returnMat

# xArr,yArr = loadDataSet('abalone.txt')
# print(stageWise(xArr,yArr,0.01,200))
# print(stageWise(xArr,yArr,0.001,5000))
# xMat = mat(xArr)
# yMat = mat(yArr).T
# xMat = regularize(xMat)
# yM = mean(yMat,0)
# yMat = yMat - yM
# weights = standRegres(xMat,yMat.T)
# print(weights.T)

#def scrapePage(inFile,outFile,yr,numPce,origPrc):
#    from BeautifulSoup import BeautifulSoup
#    fr = open(inFile); fw=open(outFile,'a') #a is append mode writing
#    soup = BeautifulSoup(fr.read())
#    i=1
#    currentRow = soup.findAll('table', r="%d" % i)
#    while(len(currentRow)!=0):
#        title = currentRow[0].findAll('a')[1].text
#        lwrTitle = title.lower()
#        if (lwrTitle.find('new') > -1) or (lwrTitle.find('nisb') > -1):
#            newFlag = 1.0
#        else:
#            newFlag = 0.0
#        soldUnicde = currentRow[0].findAll('td')[3].findAll('span')
#        if len(soldUnicde)==0:
#            print "item #%d did not sell" % i
#        else:
#            soldPrice = currentRow[0].findAll('td')[4]
#            priceStr = soldPrice.text
#            priceStr = priceStr.replace('$','') #strips out $
#            priceStr = priceStr.replace(',','') #strips out ,
#            if len(soldPrice)>1:
#                priceStr = priceStr.replace('Free shipping', '') #strips out Free Shipping
#            print "%s\t%d\t%s" % (priceStr,newFlag,title)
#            fw.write("%d\t%d\t%d\t%f\t%s\n" % (yr,numPce,newFlag,origPrc,priceStr))
#        i += 1
#        currentRow = soup.findAll('table', r="%d" % i)
#    fw.close()

"""预测乐高玩具套装的价格"""
from time import sleep
import json
# import urllib2  # in python2
import urllib.request
# Google 购物API抓取价格，http://code.google.com/apis/shopping/search/v1/getting_started.html
def searchForSet(retX, retY, setNum, yr, numPce, origPrc):
    sleep(10)   # 防止短时间内有过多的API调用
    myAPIstr = 'AIzaSyD2cR2KFyx12hXu6PFU-wrWot3NXvko8vY'
    searchURL = 'https://www.googleapis.com/shopping/search/v1/public/products?key=%s&country=US&q=lego+%d&alt=json' % (myAPIstr, setNum)
    pg = urllib.request.urlopen(searchURL)
    retDict = json.loads(pg.read())
    for i in range(len(retDict['items'])):
        try:
            currItem = retDict['items'][i]
            if currItem['product']['condition'] == 'new':
                newFlag = 1
            else: newFlag = 0
            listOfInv = currItem['product']['inventories']
            for item in listOfInv:
                sellingPrice = item['price']
                if sellingPrice > origPrc * 0.5:  # 过滤掉不完整的套装
                    print("%d\t%d\t%d\t%f\t%f" % (yr,numPce,newFlag,origPrc, sellingPrice))
                    retX.append([yr, numPce, newFlag, origPrc])
                    retY.append(sellingPrice)
        except: print('problem with item %d' % i)
    
def setDataCollect(retX, retY):
    searchForSet(retX, retY, 8288, 2006, 800, 49.99)
    searchForSet(retX, retY, 10030, 2002, 3096, 269.99)
    searchForSet(retX, retY, 10179, 2007, 5195, 499.99)
    searchForSet(retX, retY, 10181, 2007, 3428, 199.99)
    searchForSet(retX, retY, 10189, 2008, 5922, 299.99)
    searchForSet(retX, retY, 10196, 2009, 3263, 249.99)


"""交叉验证测试岭回归"""
def crossValidation(xArr,yArr,numVal=10):
    m = len(yArr)                           
    indexList = range(m)
    errorMat = zeros((numVal,30))#create error mat 30columns numVal rows
    for i in range(numVal):
        # 创建训练集和测试集容器
        trainX=[]; trainY=[]
        testX = []; testY = []
        random.shuffle(indexList)
        for j in range(m):#create training set based on first 90% of values in indexList
            if j < m*0.9: 
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        wMat = ridgeTest(trainX,trainY)    #get 30 weight vectors from ridge
        for k in range(30):#loop over all of the ridge estimates
            matTestX = mat(testX); matTrainX=mat(trainX)
            meanTrain = mean(matTrainX,0)
            varTrain = var(matTrainX,0)
            matTestX = (matTestX-meanTrain)/varTrain #regularize test with training params
            yEst = matTestX * mat(wMat[k,:]).T + mean(trainY)#test ridge results and store
            errorMat[i,k]=rssError(yEst.T.A,array(testY))
            #print errorMat[i,k]
    meanErrors = mean(errorMat,0)#calc avg performance of the different ridge weight vectors
    minMean = float(min(meanErrors))
    bestWeights = wMat[nonzero(meanErrors==minMean)]
    #can unregularize to get model
    #when we regularized we wrote Xreg = (x-meanX)/var(x)
    #we can now write in terms of x not Xreg:  x*w/var(x) - meanX/var(x) +meanY
    xMat = mat(xArr); yMat=mat(yArr).T
    meanX = mean(xMat,0); varX = var(xMat,0)
    unReg = bestWeights/varX  # 数据还原，因为standRegres()没有进行数据标准化，而岭回归使用了数据标准化
    print("the best model from Ridge Regression is:\n",unReg)
    print("with constant term: ",-1*sum(multiply(meanX,unReg)) + mean(yMat))

lgX = []; lgY = []
setDataCollect(lgX, lgY)
print(shape(lgX))
lgX1=mat(ones((58,5)))
lgX1[:, 1:5]=mat(lgX)
print(lgX[0])
print(lgX1[0])
ws=standRegres(lgX1,lgY)
print(ws)
print(lgX1[0]*ws)
print(lgX1[-1]*ws)
print(lgX[43]*ws)
crossValidation(lgX,lgY,10)
ridgeTest(lgX,lgY)
