# -*- coding: utf-8 -*-
# @Time    : 2019/1/28 20:11
# @Author  : LiLing
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from mlxtend.classifier import StackingClassifier
from sklearn.naive_bayes import GaussianNB
import lightgbm as lgb
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, LabelEncoder
# 显示所有列
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


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


def plot_roc(fprs, tprs, aucs, title):
    plt.figure()
    lw = 2
    for i, name in enumerate(models):
        plt.plot(fprs[i], tprs[i], lw=lw,
                 label='{0} (AUC:{1:0.2f})'.format(name, aucs[i]))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic of '+title)
    plt.legend(loc="lower right")
    plt.savefig(title + '.jpg')
    plt.show()


data = pd.read_csv('./data.csv', engine='python')
print(data.shape)

# 删除无关特征列
data.drop(['Unnamed: 0', 'custid', 'trade_no', 'bank_card_no', 'source', 'id_name', 'student_feature'], axis=1, inplace=True)
print(data.shape)

# 处理缺失值
data.fillna(method='ffill', inplace=True)
data.fillna(method='bfill', inplace=True)

# 数据类型转换
le = LabelEncoder()
data['reg_preference_for_trad'] = le.fit_transform(data['reg_preference_for_trad'].astype(str))

data[['latest_query_time', 'loans_latest_time']] = data[['latest_query_time', 'loans_latest_time']].\
    applymap(lambda x: float(str(x).split('-')[0]+str(x).split('-')[1]+str(x).split('-')[2]))


# IV特征选择
def CalcIV(Xvar, Yvar):
    N_0 = np.sum(Yvar == 0)    # 非响应客户
    N_1 = np.sum(Yvar == 1)    # 响应客户
    N_0_group = np.zeros(np.unique(Xvar).shape)   # 分组

    N_1_group = np.zeros(np.unique(Xvar).shape)
    for i in range(len(np.unique(Xvar))):
        # 计算非响应客户和响应客户的各个组内的相关值
        N_0_group[i] = Yvar[(Xvar == np.unique(Xvar)[i]) & (Yvar == 0)].count()
        N_1_group[i] = Yvar[(Xvar == np.unique(Xvar)[i]) & (Yvar == 1)].count()
    # iv值
    iv = np.sum((N_0_group / N_0 - N_1_group / N_1) * np.log((N_0_group / N_0) / (N_1_group / N_1)))
    if iv >= 1.0:  # 处理极端值
        iv = 1
    return iv


def caliv_batch(df, Yvar):
    ivlist = []
    for col in df.columns:
        iv = CalcIV(df[col], Yvar)
        ivlist.append(iv)
    names = list(df.columns)
    iv_df = pd.DataFrame({'Var': names, 'Iv': ivlist}, columns=['Var', 'Iv'])
    return iv_df, ivlist


y = data['status']
X = data.drop(['status'], axis=1)

# im_iv, ivl = caliv_batch(X, y)
# threshold = 0.02
# data_index = []
# for i in range(len(ivl)):
#     if im_iv['Iv'][i] < threshold:
#         data_index.append(im_iv['Var'][i])
# print(X.shape)
# print(data_index)
# X.drop(data_index, axis=1, inplace=True)
# print(X.shape)

# 随机森林特征选择
feat_lables = X.columns
# forest = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=1)
# forest.fit(X, y)
# importance = forest.feature_importances_
# imp_result = np.argsort(importance)[::-1]
# for i in range(X.shape[1]):
#     print("%2d. %-*s %f" % (i+1, 30, feat_lables[i], importance[imp_result[i]]))
# threshold = 0.01
# data_index = list(X.columns[importance < threshold])
# print(X.shape)
# X.drop(data_index, axis=1, inplace=True)
# print(X.shape)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2018)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

lr_model = LogisticRegressionCV(class_weight='balanced', cv=5, max_iter=1000)
svm_model = SVC(class_weight='balanced', gamma='auto', probability=True)
dt_model = DecisionTreeClassifier(class_weight='balanced')
rf_model = RandomForestClassifier(class_weight='balanced', n_estimators=100)
gbdt_model = GradientBoostingClassifier(n_estimators=100)
xg_model = XGBClassifier(n_estimators=100)
lgb_model = lgb.LGBMClassifier(n_estimators=100)
gnb_model = GaussianNB()

sclf = StackingClassifier(classifiers=[lgb_model, gbdt_model, rf_model], use_probas=True,
                          average_probas=False,
                          meta_classifier=lr_model)

models = {'LR': lr_model,
          'SVM': svm_model,
          'DT': dt_model,
          'RF': rf_model,
          'GBDT': gbdt_model,
          'XGBoost': xg_model,
          'LightGBM': lgb_model,
          'NB': gnb_model,
          'StackingClassifier': sclf}

df_result = pd.DataFrame(columns=('Model', 'dataset', 'Accuracy', 'Precision', 'Recall', 'F1 score', 'AUC'))
row = 0
fprs_train = []
tprs_train = []
aucs_train = []
fprs_test = []
tprs_test = []
aucs_test = []
for name, clf in models.items():
    clf.fit(X_train, y_train)
    print(name)
    acc, p, r, f1, fpr_train, tpr_train, auc_train = get_metric(clf, X_train, y_train)
    fprs_train.append(fpr_train)
    tprs_train.append(tpr_train)
    aucs_train.append(auc_train)
    df_result.loc[row] = [name, 'train', acc, p, r, f1, auc_train]
    row += 1

    acc, p, r, f1, fpr_test, tpr_test, auc_test = get_metric(clf, X_test, y_test)
    fprs_test.append(fpr_test)
    tprs_test.append(tpr_test)
    aucs_test.append(auc_test)
    df_result.loc[row] = [name, 'test', acc, p, r, f1, auc_test]
    row += 1

    # plt.figure()
    # lw = 2
    # plt.plot(fpr_train, tpr_train, color='darkorange', lw=lw, label='train (AUC:%0.2f)' % auc_train)
    # plt.plot(fpr_test, tpr_test, color='cornflowerblue', lw=lw, label='test (AUC:%0.2f)' % auc_test)
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic of '+name)
    # plt.legend(loc="lower right")
    # plt.savefig(name + '.jpg')
    # plt.show()


print(df_result)
df_result.to_csv("df_result.csv")

plot_roc(fprs_train, tprs_train, aucs_train, 'train')
plot_roc(fprs_test, tprs_test, aucs_test, 'test')
