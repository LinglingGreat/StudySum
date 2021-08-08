# -*- coding: utf-8 -*-
# @Author: LiLing
# @Date:   2019-01-02 22:18:04
# @Last Modified by:   LL
# @Last Modified time: 2019-01-03 15:51:10
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

data = pd.read_csv('./data_all.csv', engine='python')

y = data['status']
X = data.drop(['status'],axis = 1)
print('The shape of X: ', X.shape)
print('Numbers of label 1: ', len(y[y==1]), ' Numbers of label 0: ', len(y[y==0]))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2018)
print('For train, Numbers of label 1: ', len(y_train[y_train==1]), ' Numbers of label 0: ', len(y_train[y_train==0]))
print('For test, Numbers of label 1: ', len(y_test[y_test==1]), ' Numbers of label 0: ', len(y_test[y_test==0]))

df_result = pd.DataFrame(columns=('Model', 'Accuracy', 'AUC'))
row = 0

# 逻辑回归
clf_lr = LogisticRegression(solver='liblinear').fit(X_train, y_train)
y_test_pred = clf_lr.predict(X_test)
acc_lr = metrics.accuracy_score(y_test, y_test_pred)

y_test_proba = clf_lr.predict_proba(X_test)
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_test_proba[:,1])
# p_lr = metrics.precision_score(y_test, y_test_pred, average='binary')
# r_lr = metrics.recall_score(y_test, y_test_pred)
f1_lr = metrics.f1_score(y_test, y_test_pred, average='micro')
print(type(y_test), type(y_test_pred))
auc_lr = metrics.auc(fpr, tpr)

df_result.loc[row] = ['LR', acc_lr, auc_lr]
row += 1

print('Logistic Regression...')
print('Accuracy is ', acc_lr)
print('AUC is ', auc_lr)
print('f1 ', f1_lr)
print()


# SVM
clf_svm = SVC(gamma='auto', probability=True).fit(X_train, y_train)
y_test_pred = clf_svm.predict(X_test)
acc_svm = metrics.accuracy_score(y_test, y_test_pred)

y_test_proba = clf_svm.predict_proba(X_test)
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_test_proba[:,1])
# p_svm = metrics.precision_score(y_test, y_test_pred)
# r_svm = metrics.recall_score(y_test, y_test_pred)
auc_svm = metrics.auc(fpr, tpr)

df_result.loc[row] = ['SVM', acc_svm, auc_svm]
row += 1


print('SVM...')
print('Accuracy is ', acc_svm)
print('AUC is ', auc_svm)
print()


# 决策树
clf_dt = DecisionTreeClassifier().fit(X_train, y_train)
y_test_pred = clf_dt.predict(X_test)
acc_dt = metrics.accuracy_score(y_test, y_test_pred)

y_test_proba = clf_dt.predict_proba(X_test)
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_test_proba[:,1])
# p_dt = metrics.precision_score(y_test, y_test_pred)
# r_dt = metrics.recall_score(y_test, y_test_pred)
auc_dt = metrics.auc(fpr, tpr)

df_result.loc[row] = ['DT', acc_dt, auc_dt]
row += 1


print('Decision Tree...')
print('Accuracy is ', acc_dt)
print('AUC is ', auc_dt)
print()

print(df_result)