# -*- coding: utf-8 -*-
# @Time    : 2019/1/25 18:23
# @Author  : LiLing
import pandas as pd
import numpy as np
data = pd.read_csv('./data.csv', engine='python')
print(data.shape)

data.drop(['Unnamed: 0', 'custid', 'trade_no', 'bank_card_no', 'source', 'id_name', 'student_feature'], axis=1, inplace=True)
print(data.shape)

regcols = ['reg_preference_for_trad_' + str(i) for i in range(5)]
print(regcols)
tmpdf = pd.get_dummies(data['reg_preference_for_trad'].replace('nan', np.nan))
tmpdf.columns = regcols
data[regcols] = tmpdf
data.drop(['reg_preference_for_trad'],axis=1, inplace=True)

# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# tmpcol = le.fit_transform(data['reg_preference_for_trad'].astype(str))
# print(np.unique(tmpcol))

data.fillna(method='ffill', inplace=True)
data.fillna(method='bfill', inplace=True)

data[['latest_query_time', 'loans_latest_time']]=data[['latest_query_time', 'loans_latest_time']].\
    applymap(lambda x:float(str(x).split('-')[0]+str(x).split('-')[1]+str(x).split('-')[2]))

# from sklearn.feature_selection import VarianceThreshold
# selector = VarianceThreshold()
# print(X.shape)
# X = selector.fit_transform(X)
# print(X.shape)


def get_metric(clf, X, y_true):
    y_pred = clf.predict(X)
    y_proba = clf.predict_proba(X)

    acc = metrics.accuracy_score(y_true, y_pred)
    p = metrics.precision_score(y_true, y_pred)
    r = metrics.recall_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred)
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_proba[:, 1])
    auc = metrics.auc(fpr, tpr)
    return acc, p, r, f1, fpr, tpr, auc


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

y = data['status']
X = data.drop(['status'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2018)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

clf = LogisticRegressionCV(class_weight='balanced', max_iter=1000)

clf.fit(X_train, y_train)
acc, p, r, f1, fpr_train, tpr_train, auc_train = get_metric(clf, X_train, y_train)
print("train accuracy:{:.2%}, precision:{:.2%}, recall:{:.2%}, F1:{:.2}".format(acc, p, r, f1))
acc, p, r, f1, fpr_test, tpr_test, auc_test = get_metric(clf, X_test, y_test)
print("test accuracy:{:.2%}, precision:{:.2%}, recall:{:.2%}, F1:{:.2}".format(acc, p, r, f1))

plt.figure()
lw = 2
plt.plot(fpr_train, tpr_train, color='darkorange', lw=lw, label='train (AUC:%0.2f)' % auc_train)
plt.plot(fpr_test, tpr_test, color='cornflowerblue', lw=lw, label='test (AUC:%0.2f)' % auc_test)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic of LR')
plt.legend(loc="lower right")
plt.show()
