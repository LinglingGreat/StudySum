import pandas as pd
from sklearn.model_selection import train_test_split
# 导入数据，划分训练测试集
Train_data = pd.read_csv('breast-cancer-unsupervised-ad.csv')
Train_data.columns = ['f' + str(i) for i in range(30)] + ['label']
Train_data['label'][Train_data['label']=='o']=1
Train_data['label'][Train_data['label']=='n']=0
Train_data['label'] = Train_data['label'].astype(float)
print(Train_data.head())
X_train, X_test, y_train, y_test = train_test_split(Train_data.loc[:, Train_data.columns != 'label'], Train_data['label'], test_size=0.3,stratify=Train_data['label'])

from pyod.models.pca import PCA
from pyod.models.hbos import HBOS
from pyod.utils.data import evaluate_print

# 训练PCA模型
clf_name = 'PCA'
clf = PCA()
clf.fit(X_train) # 注意训练模型的时候，不需要输入y参数

# 得到训练标签和训练分数
y_train_pred = clf.labels_   # 0正常，1异常
y_train_scores = clf.decision_scores_  # 数值越大越异常

# 用训练好的模型预测测试数据的标签和分数
y_test_pred = clf.predict(X_test) 
y_test_scores = clf.decision_function(X_test)  

# 评估并打印结果
print("\nOn Training Data:")
evaluate_print(clf_name, y_train, y_train_scores)
print("\nOn Test Data:")
evaluate_print(clf_name, y_test, y_test_scores)