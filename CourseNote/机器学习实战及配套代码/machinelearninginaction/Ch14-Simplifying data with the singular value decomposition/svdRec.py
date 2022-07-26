'''
Created on Mar 8, 2011

@author: Peter
'''
from numpy import *
from numpy import linalg as la

# U, Sigma, VT = la.svd([[1, 1], [7, 7]])
# print(U)
# print(Sigma)
# print(VT)


def loadExData():
    return[[0, 0, 0, 2, 2],
           [0, 0, 0, 3, 3],
           [0, 0, 0, 1, 1],
           [1, 1, 1, 0, 0],
           [2, 2, 2, 0, 0],
           [5, 5, 5, 0, 0],
           [1, 1, 1, 0, 0]]


def loadExData2():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]


# Data = loadExData()
# U, Sigma, VT = linalg.svd(Data)
# print(Sigma)
# Sig3 = mat([[Sigma[0], 0, 0], [0, Sigma[1], 0], [0, 0, Sigma[2]]])
# print(U[:, :3]*Sig3*VT[:3, :])


def ecludSim(inA,inB):   # 欧氏距离对应的相似度
    return 1.0/(1.0 + la.norm(inA - inB))


def pearsSim(inA,inB):   # 皮尔逊相关系数
    if len(inA) < 3 : return 1.0
    return 0.5+0.5*corrcoef(inA, inB, rowvar = 0)[0][1]


def cosSim(inA,inB):   # 余弦相似度
    num = float(inA.T*inB)
    denom = la.norm(inA)*la.norm(inB)
    return 0.5+0.5*(num/denom)


# myMat = mat(loadExData())
# print(ecludSim(myMat[:, 0], myMat[:, 4]))
# print(ecludSim(myMat[:, 0], myMat[:, 0]))
# print(cosSim(myMat[:, 0], myMat[:, 4]))
# print(cosSim(myMat[:, 0], myMat[:, 0]))
# print(pearsSim(myMat[:, 0], myMat[:, 4]))
# print(pearsSim(myMat[:, 0], myMat[:, 0]))

def standEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]
    simTotal = 0.0; ratSimTotal = 0.0
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating == 0: continue
        # 寻找两个用户都评级的物品
        overLap = nonzero(logical_and(dataMat[:,item].A>0, \
                                      dataMat[:,j].A>0))[0]
        if len(overLap) == 0: similarity = 0
        else: similarity = simMeas(dataMat[overLap,item], \
                                   dataMat[overLap,j])
        print('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0: return 0
    else: return ratSimTotal/simTotal


def svdEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]
    simTotal = 0.0; ratSimTotal = 0.0
    U,Sigma,VT = la.svd(dataMat)
    Sig4 = mat(eye(4)*Sigma[:4])  # arrange Sig4 into a diagonal matrix
    xformedItems = dataMat.T * U[:,:4] * Sig4.I  # create transformed items
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating == 0 or j==item: continue
        similarity = simMeas(xformedItems[item,:].T,\
                             xformedItems[j,:].T)
        print('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0: return 0
    else: return ratSimTotal/simTotal


def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    unratedItems = nonzero(dataMat[user,:].A==0)[1]  # find unrated items
    if len(unratedItems) == 0: return 'you rated everything'
    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        itemScores.append((item, estimatedScore))
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]


# myMat = mat(loadExData())
# myMat[0, 1] = myMat[0, 0] = myMat[1, 0] = myMat[2, 0] =4
# myMat[3, 3] = 2
# myMat = mat(loadExData2())
# print(myMat)
# print(recommend(myMat, 2))
# print(recommend(myMat, 2, simMeas=ecludSim))
# print(recommend(myMat, 2, simMeas=pearsSim))
# print(recommend(myMat, 1, estMethod=svdEst))
# print(recommend(myMat, 1, estMethod=svdEst, simMeas=pearsSim))

# U, Sigma, VT = la.svd(mat(loadExData2()))
# print(Sigma)
# Sig2 = Sigma ** 2
# print(sum(Sig2))
# print(sum(Sig2)*0.9)
# print(sum(Sig2[:2]))
# print(sum(Sig2[:3]))


def printMat(inMat, thresh=0.8):
    for i in range(32):
        for k in range(32):
            if float(inMat[i,k]) > thresh:
                print(1,)
            else: print(0,)
        print('')


def imgCompress(numSV=3, thresh=0.8):    # 图像压缩
    myl = []
    for line in open('0_5.txt').readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    myMat = mat(myl)
    print("****original matrix******")
    printMat(myMat, thresh)
    U,Sigma,VT = la.svd(myMat)
    SigRecon = mat(zeros((numSV, numSV)))
    for k in range(numSV):  # construct diagonal matrix from vector
        SigRecon[k,k] = Sigma[k]
    reconMat = U[:,:numSV]*SigRecon*VT[:numSV,:]
    print("****reconstructed matrix using %d singular values******" % numSV)
    printMat(reconMat, thresh)


# imgCompress(2)
