#coding=utf-8
"""
Author:tongqing
data:2020/6/10 14:17
desc
"""

from data_process import  load_rating,load_train_data
from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
if __name__ == '__main__':
    #1.load data
    x_feature,y=load_train_data()

    # 2 no need to onehot encoder
    X_train, X_test, y_train, y_test = train_test_split(x_feature, y, test_size=0.33, random_state=42)
    num_tree=50
    gbdt = GradientBoostingClassifier(n_estimators=num_tree, random_state=1, max_depth=8,
                                      min_samples_split=30)

    gbdt.fit(X_train,y_train)

    train_new_feature =gbdt.apply(X_train)
    test_new_feature=gbdt.apply(X_test)

    # 3. gbdt generate feature
    train_new_feature = train_new_feature.reshape(-1, num_tree)
    test_new_feature=test_new_feature.reshape(-1,num_tree)
    print(train_new_feature[1])
    enc = OneHotEncoder()

    enc.fit(np.concatenate([train_new_feature,test_new_feature],axis=0))

    #new feature
    train_new_feature2 = np.array(enc.transform(train_new_feature).toarray())
    test_new_feature2=np.array(enc.transform(test_new_feature).toarray())
    #lr demo
    lr=LogisticRegression()
    lr.fit(train_new_feature2,y_train)
    lr_predict=lr.predict(test_new_feature2)
    lr_predict_score = lr.predict_proba( test_new_feature2)
    gbdt_predict=gbdt.predict(X_test)
    gbdt_predict_score=gbdt.predict_proba(X_test)
    df_re=pd.DataFrame({"true":y_test,"lr_pre":lr_predict,"gbdt_pre":gbdt_predict,})
    print(df_re.head(20))
    print("LRmode acc：",accuracy_score(y_test, lr_predict))
    print("GBDTMode acc：",accuracy_score(y_test, gbdt_predict))
    print("LRmode auc：", roc_auc_score(y_test, lr_predict_score[:,1]))
    print("GBDTMode auc：", roc_auc_score(y_test, gbdt_predict_score[:,1]))

