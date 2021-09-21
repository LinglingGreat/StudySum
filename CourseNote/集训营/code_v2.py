# -*- coding: utf-8 -*-
# @Author: LiLing
# @Date:   2019-01-02 22:18:04
# @Last Modified by:   LL
# @Last Modified time: 2019-01-03 22:57:52
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import lightgbm as lgb
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn import metrics

data = pd.read_csv('./data_all.csv', engine='python')

y = data['status']
X = data.drop(['status'], axis=1)
print('The shape of X: ', X.shape)
print('proportion of label 1: ', len(y[y == 1])/len(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2018)
print('For train, proportion of label 1: ', len(y_train[y_train == 1])/len(y_train))
print('For test, proportion of label 1: ', len(y_test[y_test == 1])/len(y_test))

rf_model = RandomForestClassifier(n_estimators='warn', criterion='gini', max_depth=None, min_samples_split=2,
                                  min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto',
                                  max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,
                                  bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0,
                                  warm_start=False, class_weight=None)

gbdt_model = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0,
                                        criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1,
                                        min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0,
                                        min_impurity_split=None, init=None, random_state=None, max_features=None,
                                        verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto',
                                        validation_fraction=0.1, n_iter_no_change=None, tol=0.0001)

xg_model = XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=100, silent=True, objective='binary:logistic',
                         booster='gbtree', n_jobs=1, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0,
                         subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1,
                         scale_pos_weight=1, base_score=0.5, random_state=0, seed=None, missing=None)

lgb_model = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=31, max_depth=-1, learning_rate=0.1, n_estimators=100,
                               subsample_for_bin=200000, objective=None, class_weight=None, min_split_gain=0.0,
                               min_child_weight=0.001, min_child_samples=20, subsample=1.0, subsample_freq=0,
                               colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=0.0, random_state=None, n_jobs=-1,
                               silent=True, importance_type='split')


models = {'RF': rf_model,
          'GBDT': gbdt_model,
          'XGBoost': xg_model,
          'LightGBM': lgb_model}

df_result = pd.DataFrame(columns=('Model', 'Accuracy', 'Precision', 'Recall', 'F1 score', 'AUC'))
row = 0
fprs = []
tprs = []
aucs = []
for name, clf in models.items():
    clf.fit(X_train, y_train)
    y_test_pred = clf.predict(X_test)

    acc = metrics.accuracy_score(y_test, y_test_pred)
    p = metrics.precision_score(y_test, y_test_pred)
    r = metrics.recall_score(y_test, y_test_pred)
    f1 = metrics.f1_score(y_test, y_test_pred)

    y_test_proba = clf.predict_proba(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_test_proba[:, 1])
    fprs.append(fpr)
    tprs.append(tpr)
    auc = metrics.auc(fpr, tpr)
    aucs.append(auc)

    df_result.loc[row] = [name, acc, p, r, f1, auc]
    print(df_result.loc[row])
    row += 1

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic of '+name)
    plt.legend(loc="lower right")
    plt.show()

plt.figure()
lw = 2
colors = ['deeppink', 'aqua', 'darkorange', 'cornflowerblue']
for i, name in enumerate(models):
    plt.plot(fprs[i], tprs[i], color=colors[i], lw=lw, label='ROC curve of {0} (area = {1:0.2f})'.format(name, aucs[i]))
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic of Models')
plt.legend(loc="lower right")
plt.show()

print(df_result)
