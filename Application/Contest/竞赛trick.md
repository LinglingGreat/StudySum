**多分类下准召f1优化**

如果多类别不均衡的话，这时候直接使用神经网络优化交叉熵损失得到的结果，f1显然不是全局最优的。

需要用阈值搜索。

传统的多分类我们预测结果使用argmax(logits)。这时候，可以形式化的表达为求$argmax(w*logits)$使得f1均值最大。其中w就是要求得的再放缩权重。 我们可以使用非线性优化的方法求解这个问题，scipy的库里有很多实现。

**有序关系的离散标签优化**

我们经常遇到这样的问题，比如情感打分预测1-5，我们用mse指标来评价，通常，我们用回归拟合1-5的时候，如何切分阈值对我们的结果有很大的影响，在这里我们也是进行阈值的搜索，不一样的是。我们的阈值要在1-5之间。使用的方法也是非线性优化。下面一段代码给一个简单的例子，同理前面多分类下的准召优化我们也可以这样写

```python
from functools import partial
import numpy as np
import scipy as sp

class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4

        ll = quadratic_weighted_kappa(y, X_p)
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        return X_p

    def coefficients(self):
        return self.coef_['x']
```



## 参考资料
[Kaggle进阶：显著提分trick之指标优化](https://mp.weixin.qq.com/s/jH9grYg-xiuQxMTDq99olg)


