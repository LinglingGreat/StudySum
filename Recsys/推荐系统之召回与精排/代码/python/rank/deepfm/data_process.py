#coding=utf-8
"""
Author:tongqing
data:2020/6/10 11:55
desc
"""
import pandas as pd
import  numpy as np

from sklearn.preprocessing import LabelEncoder




rating_colnames=["UserID","MovieID","Rating","Timestamp"]

user_colnames=["UserID","Gender","Age","Occupation","Zip-code"]

movies_colnames=["MovieID","Title","Genres"]

def load_rating(path=r"data/ratings.dat",make_class=True):
    dt = pd.read_csv(path, sep=":", names=rating_colnames)
    if make_class:
        dt["click"]=dt["Rating"].map(lambda x: 1 if x>3 else 0)
        dt.drop("Rating",inplace=True,axis=1)
    return dt

def load_movies(path=r"data/movies.dat"):
    dt = pd.read_csv(path, sep="::", names=movies_colnames)
    return dt


def load_users(path=r"data/users.dat"):
    dt = pd.read_csv(path, sep="::", names=user_colnames)
    return dt


def load_train_data():
    user_log = load_rating()
    user_profile = load_users()

    dt = pd.merge(user_log, user_profile, 'left', on="UserID")
    feature_name = ["UserID", "MovieID", "Gender", "Age", "Occupation", "Zip-code"]
    x_feature = dt[feature_name]

    enc_label = LabelEncoder()
    x_feature['Gender'] = enc_label.fit_transform(x_feature['Gender']).copy()
    x_feature["Zip-code"] = enc_label.fit_transform(x_feature["Zip-code"]).copy()
    y_label = dt["click"]

    return np.array(x_feature),np.array(y_label)

load_train_data()