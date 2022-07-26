#coding=utf-8
"""
Author:tongqing
data:2020/6/10 14:17
desc
"""
from data_process import  load_rating,load_train_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,accuracy_score
import pandas as pd
import numpy as np



if __name__ == '__main__':
    x_feature,y=load_train_data()
    enc=OneHotEncoder()
    x_feature=enc.fit_transform(x_feature).todense()

    #split train and test
    X_train, X_test, y_train, y_test = train_test_split(x_feature, y, test_size = 0.33, random_state = 42)
    lr=LogisticRegression()
    lr.fit(X_train,y_train)
    predict=lr.predict(X_test)
    predict_score=lr.predict_proba(X_test)
    predict_true=pd.DataFrame({"true":np.array(y_test),"predict": predict})
    print(lr.coef_)
    print(predict_true)
    print("auc :",roc_auc_score(y_test,predict_score[:,1]))
    print("acc :",accuracy_score(y_test,predict))