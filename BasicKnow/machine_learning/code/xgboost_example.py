# -*- coding: utf-8 -*-
# @Author: LiLing
# @Date:   2018-10-05 21:09:31
# @Last Modified by:   Liling
# @Last Modified time: 2018-10-05 21:24:17
import xgboost

# First XGBoost model for Pima Indians dataset
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if __name__=='__main__':
	# load data
	dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",")
	# split data into X and y
	X = dataset[:,0:8]
	Y = dataset[:,8]
	# split data into train and test sets
	seed = 7
	test_size = 0.33
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
	# fit model no training data
	model = XGBClassifier()
	model.fit(X_train, y_train)
	# make predictions for test data
	y_pred = model.predict(X_test)
	predictions = [round(value) for value in y_pred]
	# evaluate predictions
	accuracy = accuracy_score(y_test, predictions)
	print("Accuracy: %.2f%%" % (accuracy * 100.0))

	# load data
	dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",")
	# split data into X and y
	X = dataset[:,0:8]
	Y = dataset[:,8]
	# split data into train and test sets
	seed = 7
	test_size = 0.33
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
	# fit model no training data
	model = XGBClassifier()
	eval_set = [(X_test, y_test)]
	model.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="logloss", eval_set=eval_set, verbose=True)
	# make predictions for test data
	y_pred = model.predict(X_test)
	predictions = [round(value) for value in y_pred]
	# evaluate predictions
	accuracy = accuracy_score(y_test, predictions)
	print("Accuracy: %.2f%%" % (accuracy * 100.0))

	from xgboost import plot_importance
	from matplotlib import pyplot
	# load data
	dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",")
	# split data into X and y
	X = dataset[:,0:8]
	y = dataset[:,8]
	# fit model no training data
	model = XGBClassifier()
	model.fit(X, y)
	# plot feature importance
	plot_importance(model)
	pyplot.show()

	# Tune learning_rate
	from sklearn.model_selection import GridSearchCV
	from sklearn.model_selection import StratifiedKFold
	# load data
	dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",")
	# split data into X and y
	X = dataset[:,0:8]
	Y = dataset[:,8]
	# grid search
	model = XGBClassifier()
	learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
	param_grid = dict(learning_rate=learning_rate)
	kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
	grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
	grid_result = grid_search.fit(X, Y)
	# summarize results
	print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
	means = grid_result.cv_results_['mean_test_score']
	params = grid_result.cv_results_['params']
	for mean, param in zip(means, params):
	    print("%f  with: %r" % (mean, param))

# 1.learning rate
# 2.tree 
# max_depth
# min_child_weight
# subsample, colsample_bytree
# gamma 
# 3.正则化参数
# lambda 
# alpha 
# xgb1 = XGBClassifier(
#  learning_rate =0.1,
#  n_estimators=1000,
#  max_depth=5,
#  min_child_weight=1,
#  gamma=0,
#  subsample=0.8,
#  colsample_bytree=0.8,
#  objective= 'binary:logistic',
#  nthread=4,
#  scale_pos_weight=1,
#  seed=27)