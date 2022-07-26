# -*- coding: utf-8 -*-
# @Author: LiLing
# @Date:   2018-10-23 22:31:38
# @Last Modified by:   Liling
# @Last Modified time: 2018-10-23 22:51:50

# 定义一个函数对 n 折训练集和测试集进行预测，该函数返回每个模型对训练集和测试集的预测结果
def Stacking(model,train,y,test,n_fold):
  folds=StratifiedKFold(n_splits=n_fold,random_state=1)
  test_pred=np.empty((test.shape[0],1),float)
  train_pred=np.empty((0,1),float)
  for train_indices,val_indices in folds.split(train,y.values):
     x_train,x_val=train.iloc[train_indices],train.iloc[val_indices]
     y_train,y_val=y.iloc[train_indices],y.iloc[val_indices]

     model.fit(X=x_train,y=y_train)
     train_pred=np.append(train_pred,model.predict(x_val))
     test_pred=np.append(test_pred,model.predict(test))
   return test_pred.reshape(-1,1),train_pred

model1 = tree.DecisionTreeClassifier(random_state=1)
test_pred1 ,train_pred1=Stacking(model=model1,n_fold=10, train=x_train,test=x_test,y=y_train)
train_pred1=pd.DataFrame(train_pred1)
test_pred1=pd.DataFrame(test_pred1)

model2 = KNeighborsClassifier()
test_pred2 ,train_pred2=Stacking(model=model2,n_fold=10,train=x_train,test=x_test,y=y_train)
train_pred2=pd.DataFrame(train_pred2)
test_pred2=pd.DataFrame(test_pred2)

df = pd.concat([train_pred1, train_pred2], axis=1)
df_test = pd.concat([test_pred1, test_pred2], axis=1)

model = LogisticRegression(random_state=1)
model.fit(df,y_train)
model.score(df_test, y_test)

# blending
model1 = tree.DecisionTreeClassifier()
model1.fit(x_train, y_train)
val_pred1=model1.predict(x_val)
test_pred1=model1.predict(x_test)
val_pred1=pd.DataFrame(val_pred1)
test_pred1=pd.DataFrame(test_pred1)

model2 = KNeighborsClassifier()
model2.fit(x_train,y_train)
val_pred2=model2.predict(x_val)
test_pred2=model2.predict(x_test)
val_pred2=pd.DataFrame(val_pred2)
test_pred2=pd.DataFrame(test_pred2)

df_val=pd.concat([x_val, val_pred1,val_pred2],axis=1)
df_test=pd.concat([x_test, test_pred1,test_pred2],axis=1)

model = LogisticRegression()
model.fit(df_val,y_val)
model.score(df_test,y_test)

# Bagging&Boosting
# https://datahack.analyticsvidhya.com/contest/practice-problem-loan-prediction-iii/
import pandas as pd
import numpy as np

df=pd.read_csv("/home/user/Desktop/train.csv")

#filling missing values
df['Gender'].fillna('Male', inplace=True)
# https://www.analyticsvidhya.com/blog/2015/04/comprehensive-guide-data-exploration-sas-using-python-numpy-scipy-matplotlib-pandas/
#split dataset into train and test

from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.3, random_state=0)

x_train=train.drop('Loan_Status',axis=1)
y_train=train['Loan_Status']

x_test=test.drop('Loan_Status',axis=1)
y_test=test['Loan_Status']

#create dummies
x_train=pd.get_dummies(x_train)
x_test=pd.get_dummies(x_test)

# Bagging元估计
from sklearn.ensemble import BaggingClassifier
from sklearn import tree
model = BaggingClassifier(tree.DecisionTreeClassifier(random_state=1))
model.fit(x_train, y_train)
model.score(x_test,y_test)

# from sklearn.ensemble import BaggingRegressor
# model = BaggingRegressor(tree.DecisionTreeRegressor(random_state=1))
# model.fit(x_train, y_train)
# model.score(x_test,y_test)

# 随机森林
from sklearn.ensemble import RandomForestClassifier
model= RandomForestClassifier(random_state=1)
model.fit(x_train, y_train)
model.score(x_test,y_test)

# from sklearn.ensemble import RandomForestRegressor
# model= RandomForestRegressor()
# model.fit(x_train, y_train)
# model.score(x_test,y_test)

# 查看特征重要性
for i, j in sorted(zip(x_train.columns, model.feature_importances_)):
   print(i, j)

# AdaBoost
from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier(random_state=1)
model.fit(x_train, y_train)
model.score(x_test,y_test)

# from sklearn.ensemble import AdaBoostRegressor
# model = AdaBoostRegressor()
# model.fit(x_train, y_train)
# model.score(x_test,y_test)

# Gradient Boosting (GBM)
from sklearn.ensemble import GradientBoostingClassifier
model= GradientBoostingClassifier(learning_rate=0.01,random_state=1)
model.fit(x_train, y_train)
model.score(x_test,y_test)

# from sklearn.ensemble import GradientBoostingRegressor
# model= GradientBoostingRegressor()
# model.fit(x_train, y_train)
# model.score(x_test,y_test)

# XGBoost
import xgboost as xgb
model=xgb.XGBClassifier(random_state=1,learning_rate=0.01)
model.fit(x_train, y_train)
model.score(x_test,y_test)

# import xgboost as xgb
# model=xgb.XGBRegressor()
# model.fit(x_train, y_train)
# model.score(x_test,y_test)

# LightGBM
import lightgbm as lgb
train_data=lgb.Dataset(x_train,label=y_train)
#define parameters
params = {'learning_rate':0.001}
model= lgb.train(params, train_data, 100) 
y_pred=model.predict(x_test)
for i in range(0,185):
  if y_pred[i]>=0.5: 
  y_pred[i]=1
else: 
  y_pred[i]=0

# import lightgbm as lgb
# train_data=lgb.Dataset(x_train,label=y_train)
# params = {'learning_rate':0.001}
# model= lgb.train(params, train_data, 100)
# from sklearn.metrics import mean_squared_error
# rmse=mean_squared_error(y_pred,y_test)**0.5

# CatBoost
from catboost import CatBoostClassifier
model=CatBoostClassifier()
categorical_features_indices = np.where(df.dtypes != np.float)[0]
model.fit(x_train,y_train,cat_features=([ 0,  1, 2, 3, 4, 10]),eval_set=(x_test, y_test))
model.score(x_test,y_test)

# from catboost import CatBoostRegressor
# model=CatBoostRegressor()
# categorical_features_indices = np.where(df.dtypes != np.float)[0]
# model.fit(x_train,y_train,cat_features=([ 0,  1, 2, 3, 4, 10]),eval_set=(x_test, y_test))
# model.score(x_test,y_test)
