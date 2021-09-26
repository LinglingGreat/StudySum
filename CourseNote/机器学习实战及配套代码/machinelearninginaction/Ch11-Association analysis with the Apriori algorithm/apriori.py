'''
Created on Mar 24, 2011
Ch 11 code
@author: Peter
'''
from numpy import *

def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

def createC1(dataSet):
    """
    创建大小为1的所有候选项集的集合
    :param dataSet:
    :return:
    """
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
                
    C1.sort()
    # frozenset是被冰冻的集合，就是说它们是不可改变的，这些集合之后要作为字典键值使用，必须不可变
    return list(map(frozenset, C1))  # use frozen set so we can use it as a key in a dict

def scanD(D, Ck, minSupport):
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if can not in ssCnt: ssCnt[can]=1
                else: ssCnt[can] += 1
    numItems = float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key]/numItems
        if support >= minSupport:
            retList.insert(0, key)
        supportData[key] = support
    return retList, supportData   # 返回频繁项集以及其支持度dict

# dataSet = loadDataSet()
# print(dataSet)
# C1 = createC1(dataSet)
# print(C1)
# D = list(map(set, dataSet))
# L1, suppData0 = scanD(D, C1, 0.5)
# print(L1)

def aprioriGen(Lk, k): #creates Ck
    """

    :param Lk: 频繁项集列表
    :param k: 项集元素个数k
    :return:
    """
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk):
            # 前k-2个项相同时，将两个集合合并
            L1 = list(Lk[i])[:k-2]; L2 = list(Lk[j])[:k-2]
            L1.sort(); L2.sort()
            if L1==L2: #if first k-2 elements are equal
                retList.append(Lk[i] | Lk[j]) #set union
    return retList

def apriori(dataSet, minSupport = 0.5):
    C1 = createC1(dataSet)
    D = list(map(set, dataSet))
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while (len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2], k)   # 得到候选项集列表
        Lk, supK = scanD(D, Ck, minSupport)#scan DB to get Lk
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData

# dataSet = loadDataSet()
# print(dataSet)
# L, suppData = apriori(dataSet)
# print(L)
# print(aprioriGen(L[0], 2))
# L, suppData = apriori(dataSet, minSupport=0.7)
# print(L)


def generateRules(L, supportData, minConf=0.7):  #supportData is a dict coming from scanD
    """
    关联规则生成函数
    :param L: 频繁项集列表
    :param supportData: 包含哪些频繁项集支持数据的字典
    :param minConf: 最小可信度阈值
    :return:
    """
    bigRuleList = []
    for i in range(1, len(L)):  # only get the sets with two or more items
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if i > 1:
                # 进一步进行合并
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                # 计算可信度值
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList         

def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    prunedH = []   # create new list to return
    for conseq in H:
        conf = supportData[freqSet]/supportData[freqSet-conseq] #calc confidence
        if conf >= minConf: 
            print(freqSet-conseq,'-->',conseq,'conf:',conf)
            brl.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH

def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    m = len(H[0])
    if (len(freqSet) > (m + 1)):   # try further merging
        Hmp1 = aprioriGen(H, m+1)  # create Hm+1 new candidates
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        if (len(Hmp1) > 1):    # need at least two sets to merge
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)

dataSet = loadDataSet()
print(dataSet)
L, suppData = apriori(dataSet, minSupport=0.5)
rules = generateRules(L, suppData, minConf=0.7)
print(rules)
rules = generateRules(L, suppData, minConf=0.5)
print(rules)

def pntRules(ruleList, itemMeaning):
    for ruleTup in ruleList:
        for item in ruleTup[0]:
            print(itemMeaning[item])
        print("           -------->")
        for item in ruleTup[1]:
            print(itemMeaning[item])
        print("confidence: %f" % ruleTup[2])
        print()       # print a blank line
        
            
from time import sleep
# from votesmart import votesmart
# votesmart.apikey = 'a7fa40adec6f4a77178799fae4441030'
# #votesmart.apikey = 'get your api key first'

# bills = votesmart.votes.getBillsByStateRecent()   # 获得最近的100条议案
# for bill in bills:
#     print(bill.title, bill.billId)
# bill = votesmart.votes.getBill(11820)  # 根据ID获得议案的更多内容
# print(bill.actions)
# for action in bill.actions:
#     if action.stage == 'Passage':
#         print(action.actionId)
# voteList = votesmart.votes.getBillActionVotes(31670)  # 获得某条议案的投票信息
# print(voteList[21])
# def getActionIds():
#     actionIdList = []; billTitleList = []
#     fr = open('recent20bills.txt')
#     for line in fr.readlines():
#         billNum = int(line.split('\t')[0])
#         try:
#             billDetail = votesmart.votes.getBill(billNum) #api call
#             for action in billDetail.actions:
#                 # 过滤出包含投票的行为
#                 if action.level == 'House' and \
#                 (action.stage == 'Passage' or action.stage == 'Amendment Vote'):
#                     actionId = int(action.actionId)
#                     print('bill: %d has actionId: %d' % (billNum, actionId))
#                     actionIdList.append(actionId)
#                     billTitleList.append(line.strip().split('\t')[1])
#         except:
#             print("problem getting bill %d" % billNum)
#         sleep(1)                                      #delay to be polite
#     return actionIdList, billTitleList
#
# def getTransList(actionIdList, billTitleList): #this will return a list of lists containing ints
#     itemMeaning = ['Republican', 'Democratic']#list of what each item stands for
#     for billTitle in billTitleList:#fill up itemMeaning list
#         itemMeaning.append('%s -- Nay' % billTitle)
#         itemMeaning.append('%s -- Yea' % billTitle)
#     transDict = {}#list of items in each transaction (politician)
#     voteCount = 2
#     for actionId in actionIdList:
#         sleep(3)
#         print('getting votes for actionId: %d' % actionId)
#         try:
#             voteList = votesmart.votes.getBillActionVotes(actionId)
#             for vote in voteList:
#                 if not transDict.has_key(vote.candidateName):
#                     transDict[vote.candidateName] = []
#                     if vote.officeParties == 'Democratic':
#                         transDict[vote.candidateName].append(1)
#                     elif vote.officeParties == 'Republican':
#                         transDict[vote.candidateName].append(0)
#                 if vote.action == 'Nay':
#                     transDict[vote.candidateName].append(voteCount)
#                 elif vote.action == 'Yea':
#                     transDict[vote.candidateName].append(voteCount + 1)
#         except:
#             print("problem getting actionId: %d" % actionId)
#         voteCount += 2
#     return transDict, itemMeaning

# transDict, itemMeaning = getTransList(actionIdList[:2], billTitles[:2])
# for key in transDict.keys():
#     print(transDict[key])
# print(transDict.keys()[6])
# for item in transDict[" Doyle, Michael 'Mike'"]:
#     print(itemMeaning[item])
# transDict, itemMeaning = getTransList(actionIdList, billTitles)
#
# dataSet = [transDict[key] for key in transDict.keys()]
# L, suppData = apriori(dataSet, minSupport=0.5)
# print(L)
# L, suppData = apriori(dataSet, minSupport=0.3)
# print(L[6])
# rules = generateRules(L, suppData)
# rules = generateRules(L, suppData, minConf=0.95)
# rules = generateRules(L, suppData, minConf=0.99)
# print(itemMeaning[26])

# 发现毒蘑菇的相似特征
mushDatSet = [line.split() for line in open('mushroom.dat').readlines()]
L, suppData = apriori(mushDatSet, minSupport=0.3)
for item in L[1]:
    if item.intersection('2'):
        print(item)

