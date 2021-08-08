#!/usr/bin/env python

import numpy as np
import random


# First implement a gradient checker by filling in the following functions
def gradcheck_naive(f, x):
    """ Gradient check for a function f.

    Arguments:
    f -- a function that takes a single argument and outputs the
         cost and its gradients
    x -- the point (numpy array) to check the gradient at
    """

    rndstate = random.getstate()  # 可理解为设定状态，记录下数组被打乱的操作
    random.setstate(rndstate)   # 接收get_state()返回的值，并进行同样的操作
    fx, grad = f(x) # Evaluate function value at original point
    h = 1e-4        # Do not change this!

    # Iterate over all indexes in x
    # flags=['multi_index']表示对a进行多重索引，op_flags=['readwrite']表示不仅可以对a进行read（读取），还可以write（写入）
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index   # 迭代索引，如果x是二维的，从左到右，从上到下迭代

        # Try modifying x[ix] with h defined above to compute
        # numerical gradients. Make sure you call random.setstate(rndstate)
        # before calling f(x) each time. This will make it possible
        # to test cost functions with built in randomness later.
        # 因为我们在做函数f运算的时候可能存在随机操作，但是我们在梯度校验，所以需要两次求f的随机设置是一样的，所以需要设置种子。
        # 举个例子，在第三部分要实现的word2vec模型中，我们需要进行负采样来得到错误样本，这一步就是随机操作，如果不设置种子，梯度校验就会出错。

        ### YOUR CODE HERE:
        x[ix] += h
        random.setstate(rndstate)
        fx_plus, _ = f(x)

        x[ix] -= 2*h
        random.setstate(rndstate)
        fx_minus, _ = f(x)

        x[ix] += h #还原
        numgrad = (fx_plus - fx_minus) / (2 * h)
        ### END YOUR CODE

        # Compare gradients
        reldiff = abs(numgrad - grad[ix]) / max(1, abs(numgrad), abs(grad[ix]))
        if reldiff > 1e-5:
            print ("Gradient check failed.")
            print ("First gradient error found at index %s" % str(ix))
            print ("Your gradient: %f \t Numerical gradient: %f" % (
                grad[ix], numgrad))
            return

        it.iternext() # Step to next dimension

    print ("Gradient check passed!")


def sanity_check():
    """
    Some basic sanity checks.
    """
    quad = lambda x: (np.sum(x ** 2), x * 2)

    print ("Running sanity checks...")
    gradcheck_naive(quad, np.array(123.456))      # scalar test
    gradcheck_naive(quad, np.random.randn(3,))    # 1-D test
    gradcheck_naive(quad, np.random.randn(4,5))   # 2-D test
    print ()


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_gradcheck.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print ("Running your sanity checks...")
    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE


if __name__ == "__main__":
    sanity_check()
    # your_sanity_checks()
