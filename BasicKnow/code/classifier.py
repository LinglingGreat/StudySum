# coding=utf-8

import time
from sklearn import metrics
import pickle as pickle
import pandas as pd
import numpy as np


# Multinomial Naive Bayes Classifier
def naive_bayes_classifier(train_x, train_y):
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB(alpha=0.01)
    model.fit(train_x, train_y)
    return model


# KNN Classifier
def knn_classifier(train_x, train_y):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier()
    model.fit(train_x, train_y)
    return model


# Logistic Regression Classifier
def logistic_regression_classifier(train_x, train_y):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty='l2')
    model.fit(train_x, train_y)
    return model


# Random Forest Classifier
def random_forest_classifier(train_x, train_y):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=8)
    model.fit(train_x, train_y)
    return model


# Decision Tree Classifier
def decision_tree_classifier(train_x, train_y):
    from sklearn import tree
    model = tree.DecisionTreeClassifier()
    model.fit(train_x, train_y)
    return model


# GBDT(Gradient Boosting Decision Tree) Classifier
def gradient_boosting_classifier(train_x, train_y):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=200)
    model.fit(train_x, train_y)
    return model


# AdaBoost(Gradient Boosting Decision Tree) Classifier
def adaBoost_classifier(train_x, train_y):
    from sklearn.ensemble import AdaBoostClassifier
    model = AdaBoostClassifier(n_estimators=200)
    model.fit(train_x, train_y)
    return model


# SVM Classifier
def svm_classifier(train_x, train_y):
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    model.fit(train_x, train_y)
    return model


def only_name(data, sign):
    name = []
    for col in data.columns:
        if sign in col:
            name.append(col)
    return name


def read_data(data_file, func=None, only_price=False, only_volume=False):
    """

    :param data_file:
    :param func:
    :param only_price:
    :param only_volume:
    :return:
    """
    train = pd.read_excel(data_file, sheet_name=0, header=None).dropna().reset_index(drop=True)
    test = pd.read_excel(data_file, sheet_name=1, header=None).dropna().reset_index(drop=True)
    colum = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'label']
    train.columns = colum
    test.columns = colum
    if func is not None:
        train = func(train)
        test = func(test)
    train_y = train.label
    train_x = train.drop('label', axis=1)
    test_y = test.label
    test_x = test.drop('label', axis=1)
    if only_volume:
        vol_name = only_name(train, 'v')
        train_x = train_x[vol_name]
        test_x = test_x[vol_name]
    if only_price:
        price_name = only_name(train, 'p')
        train_x = train_x[price_name]
        test_x = test_x[price_name]
    return train_x, train_y, test_x, test_y


# 每一列减去第一列
def differ(df):
    df1 = df.ix[:, :-1]
    label = df.ix[:, -1]
    df1 = df1.diff(axis=1)
    df1.drop(['p1', 'v1'], axis=1, inplace=True)
    df = pd.concat([df1, label], axis=1)
    return df


# 每一列除以第一列
def normalize_divfirst(df):
    columns1 = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7']
    for col in columns1:
        df[col] /= df['p1']
    columns2 = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6']
    for col in columns2:
        df[col] /= df['v1']
    return df


# 纵向归一化，（原始值-最小值）/（最大值-最小值）
def normalize_ver(df):
    df1 = df.ix[:, :-1]
    label = df.ix[:, -1]
    df1 = (df1 - df1.min()) / (df1.max() - df1.min())
    df = pd.concat([df1, label], axis=1)
    return df


# 横向归一化，（原始值-最小值）/（最大值-最小值）
def normalize_hor(df):
    price = df.ix[:, 0:7]
    price = (price.sub(price.min(axis=1), axis=0)).div((price.max(axis=1) - price.min(axis=1)), axis=0)
    volume = df.ix[:, 7:13]
    label = df.ix[:, -1]
    volume = (volume.sub(volume.min(axis=1), axis=0)).div((volume.max(axis=1) - volume.min(axis=1)), axis=0)
    df = pd.concat([price, volume, label], axis=1)
    return df


# 每一列除以前一列
def normalize_divprev(df):
    price = df.ix[:, 0:7]
    dprice = price.shift(-1, axis=1).dropna(axis=1)
    fprice = (dprice / price).dropna(axis=1)
    volume = df.ix[:, 7:13]
    dvolume = volume.shift(-1, axis=1).dropna(axis=1)
    label = df.ix[:, -1]
    fvolume = (dvolume / volume).dropna(axis=1)
    df = pd.concat([fprice, fvolume, label], axis=1)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    return df


# 纵向z-score,减去均值除以标准差
def zscore(df):
    price = df.ix[:, 0:7]
    price = (price.sub(price.mean(axis=1), axis=0)).div(price.std(axis=1), axis=0)
    volume = df.ix[:, 7:13]
    label = df.ix[:, -1]
    volume = (volume.sub(volume.mean(axis=1), axis=0)).div(volume.std(axis=1), axis=0)
    df = pd.concat([price, volume, label], axis=1)
    return df


def model(train_x, train_y, test_x, test_y, model_save, model_save_file, writer, sheetname):
    test_classifiers = ['AB', 'NB', 'KNN', 'LR', 'RF', 'DT', 'SVM', 'GBDT']
    classifiers = {'AB': adaBoost_classifier,
                   'NB': naive_bayes_classifier,
                   'KNN': knn_classifier,
                   'LR': logistic_regression_classifier,
                   'RF': random_forest_classifier,
                   'DT': decision_tree_classifier,
                   'SVM': svm_classifier,
                   'GBDT': gradient_boosting_classifier
                   }

    writerdata = pd.DataFrame(test_y)
    for classifier in test_classifiers:
        print('******************* %s ********************' % classifier)
        start_time = time.time()
        try:
            model = classifiers[classifier](train_x, train_y)
        except:
            print("Error!")
            continue
        print('training took %fs!' % (time.time() - start_time))
        predict = model.predict(test_x)
        if model_save_file is not None:
            model_save[classifier] = model
        precision = metrics.precision_score(test_y, predict)
        recall = metrics.recall_score(test_y, predict)
        print('precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall))
        accuracy_test = metrics.accuracy_score(test_y, predict)
        print('accuracy: %.2f%%' % (100 * accuracy_test))

        train_predict = model.predict(train_x)
        precision = metrics.precision_score(train_y, train_predict)
        recall = metrics.recall_score(train_y, train_predict)
        print('train data precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall))
        accuracy_train = metrics.accuracy_score(train_y, train_predict)
        print('train data accuracy: %.2f%%' % (100 * accuracy_train))
        predictdata = pd.DataFrame(predict)
        predictdata.columns = [str(classifier) + "\n" + str(round(accuracy_train*100, 2)) + "%\n" + str(round(accuracy_test*100, 2))+"%"]
        writerdata = pd.concat([writerdata, predictdata], axis=1)
    if model_save_file is not None:
        pickle.dump(model_save, open(model_save_file, 'wb'))
    writerdata.to_excel(writer, sheetname)
    writer.save()


def runmodel(data_file, normal_method, normal_name, writer, only_price=False, only_volume=False, model_save={}, model_save_file=None):
    print('******************* For  %s  Data ********************' % normal_name)
    train_x, train_y, test_x, test_y = read_data(data_file, normal_method, only_price, only_volume)
    model(train_x, train_y, test_x, test_y, model_save, model_save_file, writer, normal_name)


if __name__ == '__main__':
    data_file = "股票形态特征数据.xlsx"
    writer = pd.ExcelWriter('result.xlsx')
    print('reading training and testing data...')
    runmodel(data_file, normal_method=normalize_hor, normal_name='normalize_hor', writer=writer, only_price=True,
             only_volume=False)

    # runmodel(data_file, normal_method=differ, normal_name='differ', writer=writer, only_price=True,
    #          only_volume=False)
    #
    # runmodel(data_file, normal_method=normalize_divfirst, normal_name='normalize_divfirst', writer=writer, only_price=True,
    #          only_volume=False)
    #
    # runmodel(data_file, normal_method=normalize_ver, normal_name='normalize_ver', writer=writer, only_price=True,
    #          only_volume=False)
    #
    # runmodel(data_file, normal_method=normalize_divprev, normal_name='normalize_divprev', writer=writer, only_price=True,
    #          only_volume=False)
    #
    # runmodel(data_file, normal_method=zscore, normal_name='zscore', writer=writer, only_price=True,
    #          only_volume=False)

