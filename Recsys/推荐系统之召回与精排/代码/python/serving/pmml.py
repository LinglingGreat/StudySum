import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
import pandas as pd
from sklearn import tree
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn2pmml import sklearn2pmml


if __name__ == '__main__':
    data=load_iris()
    x=data.data
    y=data.target
    print(x,y)
    # pipeline=PMMLPipeline([("classifier", tree.DecisionTreeClassifier(random_state=3))])
    # pipeline.fit(x,y)
    # pre=pipeline.predict(x)
    #
    # sklearn2pmml(pipeline,"./tree_version_test.pmml",with_repr=True)