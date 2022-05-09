
import numpy as np
import pandas as pd


x1=np.random.random(1000000)
x2=np.random.random(1000000)

y=0.6*x1++2*x2+np.random.random(1000000)*0.1

dt=pd.concat([pd.Series(x1,name="x1"),pd.Series(x2,name="x2"),pd.Series(y,name="y")],axis=1)
dt.to_csv("train.csv",index=False)
# #
dt=pd.read_csv(r"C:\Users\tongqing\PycharmProjects\untitled\tesnsorflow_test\train_disturb\data\train.csv")
print(dt[["x1","x2"]])