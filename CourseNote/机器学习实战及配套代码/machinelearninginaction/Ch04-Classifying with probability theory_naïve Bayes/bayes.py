'''
Created on Oct 19, 2010

@author: Peter
'''
from numpy import *
import feedparser

""" 准备数据：从文本中构建词向量    词表到向量的转换函数 """
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive侮辱性文字, 0 not
    return postingList,classVec

# 创建一个包含在所有文档中出现的不重复词的列表
def createVocabList(dataSet):
    vocabSet = set([])  #create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document) # union of the two sets两个集合的并集，|也是一个按位或(OR)操作符
    return list(vocabSet)

# 输入参数为词汇表及某个文档，输出的是文档向量，向量的每个元素为1或0，表示词汇表中的单词在输入文档中是否出现。
# 词集模型，每个词只能出现一次
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else: print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

# listOPosts, listClasses = loadDataSet()
# myVocabList = createVocabList(listOPosts)
# print(myVocabList)
# print(setOfWords2Vec(myVocabList, listOPosts[0]))
# print(setOfWords2Vec(myVocabList, listOPosts[3]))


""" 
朴素贝叶斯分类器训练函数

计算每个类别中的文档数目,即p(ci)
对每篇训练文档：
    对每个类别：
        如果词条出现在文档中——增加该词条的计数值
        增加所有词条的计数值
对每个类别：
    对每个词条：
        将该词条的数目除以总词条数目得到条件概率,即p(w|ci)
返回每个类别的条件概率
"""
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)  # 计算文档属于侮辱性文档的概率
    # 初始化概率
    # 避免出现其中一个概率值为0，那么最后的乘积也为0的情况
    p0Num = ones(numWords); p1Num = ones(numWords)      #change to ones() 
    p0Denom = 2.0; p1Denom = 2.0                        #change to 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]  # 增加类别1各个词条的计数值
            p1Denom += sum(trainMatrix[i])   # 所有词条的计数值
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 对每个元素除以该类别中的总词数，数组除以浮点数，为防止下溢出，使用log
    # 采用自然对数可以避免下溢出或者浮点数舍入导致的错误。同时，采用自然对数进行处理不会有任何损失。
    # 处理前后的两条曲线在相同区域内同时增加或者减少，并且在相同点上取到极值
    p1Vect = log(p1Num/p1Denom)          #change to log()
    p0Vect = log(p0Num/p0Denom)          #change to log()
    return p0Vect,p1Vect,pAbusive

# listOPosts, listClasses = loadDataSet()
# myVocabList = createVocabList(listOPosts)
# trainMat = []
# for postinDoc in listOPosts:
#     trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
# p0V, p1V, pAb = trainNB0(trainMat, listClasses)
# print(p0V, p1V, pAb, sep='\n')

"""朴素贝叶斯分类函数
输入：要分类的向量以及使用函数trainNB0()计算得到的三个概率p(w|c0),p(w|c1),p(c1)"""
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)    # element-wise mult对应元素相乘
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else: 
        return 0

# 词袋模型，每个单词可以出现多次
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

# 便利函数，封装所有操作
def TestingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))

# TestingNB()


"""示例：使用朴素贝叶斯过滤垃圾邮件"""
""" 准备数据：切分文本"""
def textParse(bigString):    #input is big string, #output is word list
    import re
    # 使用正则表达式来切分，其中分隔符是除单词、数字外的任意字符串
    listOfTokens = re.split(r'\W*', bigString)
    # 单词转换成小写，去掉里面的空字符串和长度小于3的字符串
    return [tok.lower() for tok in listOfTokens if len(tok) > 2] 

"""测试算法：使用朴素贝叶斯进行交叉验证"""
def spamTest():
    docList=[]; classList = []; fullText =[]
    # 导入文件夹spam和ham下的文本文件，并将它们解析为词列表。
    for i in range(1,26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)#create vocabulary
    trainingSet = list(range(50)); testSet=[]           #create test set
    # 构建一个测试集与一个训练集，两个集合中的邮件都是随机选出的，留存交叉验证
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        print(randIndex, trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:#train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:        #classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
            print("classification error",docList[docIndex])
    print('the error rate is: ',float(errorCount)/len(testSet))
    #return vocabList,fullText

# spamTest()


"""示例：使用朴素贝叶斯分类器从个人广告中获取区域倾向
通过观察单词和条件概率值来发现与特定城市相关的内容"""
"""RSS源分类器及高频词去除函数"""
def calcMostFreq(vocabList,fullText):
    import operator
    freqDict = {}
    # 计算出现频率
    for token in vocabList:
        freqDict[token]=fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]       

def localWords(feed1,feed0):
    docList=[]; classList = []; fullText =[]
    minLen = min(len(feed1['entries']),len(feed0['entries']))
    # 每次访问一条RSS源
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1) #NY is class 1
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)#create vocabulary
    # 去掉出现次数最高的那些词
    # 如果去掉这三行代码，错误率为54%，保留的错误率为70%
    # 事实上，这些留言中出现次数最多的前30个词涵盖了所有用词的30%
    top30Words = calcMostFreq(vocabList,fullText)   #remove top 30 words
    for pairW in top30Words:
        if pairW[0] in vocabList: vocabList.remove(pairW[0])
    trainingSet = list(range(2*minLen)); testSet=[]           #create test set
    for i in range(20):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])  
    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:#train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:        #classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is: ',float(errorCount)/len(testSet))
    return vocabList,p0V,p1V

# 该网页已经找不到数据了，404，怎么解决？
# ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
# sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
# vocabList, pSF, pNY = localWords(ny, sf)


"""使用两个RSS源作为输入，然后训练并测试朴素贝叶斯分类器，返回使用的概率值"""
def getTopWords(ny,sf):
    import operator
    vocabList,p0V,p1V=localWords(ny,sf)
    topNY=[]; topSF=[]
    for i in range(len(p0V)):
        if p0V[i] > -6.0 : topSF.append((vocabList[i],p0V[i]))
        if p1V[i] > -6.0 : topNY.append((vocabList[i],p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
    for item in sortedSF:
        print(item[0])
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")
    for item in sortedNY:
        print(item[0])
