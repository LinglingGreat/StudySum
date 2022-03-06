# pdb

命令行输入：run -d 文件名     进行debug模式(ipython中)

| Command         | Meaning                                                      |
| --------------- | ------------------------------------------------------------ |
| l(ist)          | Displays 11 lines around the current line or continue the previous listing. |
| s(tep)          | Execute the current line, stop at the first possible occasion |
| n(ext)          | Continue execution until the next line in the current function is reached or it returns. |
| c(ontinue)      | Continue execution until it encounters a break point         |
| b(reak)         | Set a breakpoint (depending on the argument provided)        |
| r(eturn)        | Continue execution until the current function returns        |
| !               | (bang) The next command is interpreted as Python cmd instead of pdb command |
| h(elp)          | Help if you forget the specific command                      |
| p(rint)  变量名 | 打印变量的值                                                 |

举例

```python
# foo.py
import pdb
def sum_all(n):
    """ Add up 0, 1, 2, ..., n
    """
    s = 0
    for i in range(n):
        pdb.set_trace()
        s += i
        return s
if __name__ == "__main__":
    n = 100
    print("sum to %d: %d" % (n, sum_all(n + 1)))
```

Alternatively, we can invoke pdb in  the terminal

```python
python -m pdb foo.py
```

(-m mod : the given module is located on the Python module path and executed as a script.)

https://github.com/spiside/pdb-tutorial

https://www.ibm.com/developerworks/cn/linux/l-cn-pythondebugger/

# Jupyter

Alternatively, we can perform  debugging in IPython console.

**jupyter qtconsole**

# Profiling

Profiling helps us to find bottlenecks such that we could obtain the biggest performance improvement.
1.Basic profiling: built-in functions & commands
2.cProfile: enhanced profiler
3.Line_profiler: line by line measurements

## Basic profiling

* timeit module
* Unix system command /usr/bin/time -p python foo.py


* time module

## cProfiling

cProfile is a built-in profiling tool
Example
**python -m cProfile -s cumulative foo.py**

## Line_profiler

In foo.py

```python
@profile
def quicksort(myList, start, end):
# 	body
```

* Use **kernprof** to create an instance of LineProfiler and insert it into the **\_\_builtins\_\_ ** namespace with the name profile**. kernprof -l foo.py**


* python -m line_profiler foo.py.lprof

https://github.com/rkern/line_profiler



# Python performance

Some weaknesses:

1.Python applies garbage collector to allocate resources; it is difficult to perform low-level optimization.
2.Python applies dynamic typing (in contrast to static typing); much of compiler's work is delayed to the runtime.
3.Python implements global interpreter lock which prevents multiple threads from executing Python bytecodes in parallel.

accelerate Python !

* Compiling ahead of time: Cython, Shed Skin, Pythran, etc.


* Compling just in time: PyPy, Numba, etc.


* Parallel and distributed computing

## Cython

Cython builds a bridge between Python and C(C++):

1. Cython compiler turns Cython code (in .pyx file) into optimized and platform-independent C or C++ code.
2. C or C++ code is turned into a shared library running on a standard C or C++ compiler.
3. The flags passed to the C or C++ compiler ensure this shared library is a full-fledged Python module.

Example

```python
def fib(n):
    a, b = 0.0, 1.0
    for i in range(n):
        a, b = a + b, a
    return a
```

```cython
# .pyx file
def fib_cy(n):
    a=0.0
    b=1.0
    for i in xrange(n):
        a, b = a + b, a
    return a

def fib_cy2(int n):
    cdef double a=0.0
    cdef double b=1.0
    cdef int i
    for i in xrange(n):
        a, b = a+b, a
    return a
```

Setup using python distuitls. Compare the performance (.pyx file)

```cython
n = 500
start_t = time.time()
f = fib(n)
t = time.time() - start_t # Executation time of Python code

start_t = time.time()
f1 = fib_cy(n)
t1 = time.time() - start_t # Executation time of Cython code1

start_t = time.time()
f2 = fib_cy2(n)
t2 = time.time() - start_t # Executation time of Cython code2
```

结果，运行时间Python > Cython 1>Cython 2

## Numba

Numba gives you the power to speed up your applications with high performance functions written directly
in Python. With a few annotations, array-oriented and math-heavy Python code can be just-in-time compiled to native machine instructions, similar in performance to C, C++ and Fortran,
without having to switch languages or Python interpreters.

QuickSort

```python
from numba import jit

@jit
def partition(myList, start, end):
# 	body

@jit
def quicksort(myList, start, end):
# 	body
```

Efficiency

Generate random data: n = 5000
arr = np.random.permutation(n)
Use timeit to estimate the running time

http://numba.pydata.org.



