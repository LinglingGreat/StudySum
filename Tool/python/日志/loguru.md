## 基本用法

```Python
from loguru import logger
format_ = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)
logger.add('logs/{time:YYYY-MM-DD}.log', format=format_, level="INFO", rotation="00:00", enqueue=True)

logger.debug('调试消息')
logger.info('普通消息')
logger.warning('警告消息')
logger.error('错误消息')
logger.critical('严重错误消息')
logger.success('成功调用')

```

### rotation

```Python
logger.add("file_1.log", rotation="500 MB")  # 自动循环过大的文件
logger.add("file_2.log", rotation="12:00")  # 每天中午创建新文件
logger.add("file_3.log", rotation="1 week")  # 一旦文件太旧进行循环
```

### compression

**随着分割文件的数量越来越多之后，我们也可以进行压缩对日志进行留存，这里就要使用到 compression 参数，该参数只要你传入通用的压缩文件扩展名即可，如 zip、tar、gz 等。**

### retention

**不想对日志进行留存，或者只想保留一段时间内的日志并对超期的日志进行删除，那么直接使用 retention 参数就好了。**

对 retention 传入整数时，该参数表示的是所有文件的索引，而非要保留的文件数。=1会看到只有两个时间最近的日志文件会被保留下来，其他都被直接清理掉了。

### filter

**filter 参数能够对日志文件进行过滤，利用这个特性我们可以按照日志级别分别存入不同的文件**

```Python
from loguru import logger

logger.add("logs/jobs-info-{time:YYYY-MM-DD}.log", filter=lambda record: "INFO" in record['level'].name)
logger.add("logs/jobs-error-{time:YYYY-MM-DD}.log", filter=lambda record: "ERROR" in record['level'].name)
```

## 序列化

如果在实际中你不太喜欢以文件的形式保留日志，那么你也可以通过 serialize 参数将其转化成序列化的 json 格式，最后将导入类似于 [MongoDB](https://cloud.tencent.com/product/mongodb?from=10680)、ElasticSearch 这类数 NoSQL 数据库中用作后续的日志分析。

```Python
from loguru import logger
import os

logger.add(os.path.expanduser("~/Desktop/testlog.log"), serialize=True)
logger.info("hello, world!")
```

最后保存的日志都是序列化后的单条记录：

```Python
{
    "text": "2020-10-07 18:23:36.902 | INFO     | __main__:<module>:6 - hello, world\n",
    "record": {
        "elapsed": {
            "repr": "0:00:00.005412",
            "seconds": 0.005412
        },
        "exception": null,
        "extra": {},
        "file": {
            "name": "log_test.py",
            "path": "/Users/Bobot/PycharmProjects/docs-python/src/loguru/log_test.py"
        },
        "function": "<module>",
        "level": {
            "icon": "\u2139\ufe0f",
            "name": "INFO",
            "no": 20
        },
        "line": 6,
        "message": "hello, world",
        "module": "log_test",
        "name": "__main__",
        "process": {
            "id": 12662,
            "name": "MainProcess"
        },
        "thread": {
            "id": 4578131392,
            "name": "MainThread"
        },
        "time": {
            "repr": "2020-10-07 18:23:36.902358+08:00",
            "timestamp": 1602066216.902358
        }
    }
}
```

### bind

通过 `bind()` 添加额外属性来结构化日志

```Python
from loguru import logger

logger.add("file.log", format="{extra[ip]} {extra[user]} {message}")
context_logger = logger.bind(ip="192.168.0.1", user="someone")
context_logger.info("Contextualize your logger easily")
context_logger.bind(user="someone_else").info("Inline binding of extra attribute")
context_logger.info("Use kwargs to add context during formatting: {user}", user="anybody")
```

file.log

```Python
192.168.0.1 someone Contextualize your logger easily
192.168.0.1 someone_else Inline binding of extra attribute
192.168.0.1 anybody Use kwargs to add context during formatting: anybod
```

结合 `bind(special=True)` 和 `filter` 对日志进行更细粒度的控制

```Python
from loguru import logger

logger.add("special.log", filter=lambda record: "special" in record["extra"])
logger.debug("This message is not logged to the file")
logger.bind(special=True).info("This message, though, is logged to the file!")
```

special.log

```Python
2020-07-22 17:06:40.998 | INFO     | __main__:<module>:5 - This message, though, is logged to the file!
```

### 异步、线程安全、多进程安全

默认为线程安全，但不是异步或多进程安全的，添加参数 `enqueue=True` 即可：

logger.add("somefile.log", enqueue=True)

协程可用 `complete()` 等待

## 解析器

通常需要从日志中提取特定信息， `parse()` 可用处理日志和正则表达式。

```Python
# -*- coding: utf-8 -*-
from loguru import logger
from dateutil import parser

logger.add('file.log', format='{time} - {level.no} - {message}', encoding='utf-8')
logger.debug('调试消息')

pattern = r'(?P<time>.*) - (?P<level>[0-9]+) - (?P<message>.*)'  # 带命名组的正则表达式
caster_dict = dict(time=parser.parse, level=int)  # 匹配)

for i in logger.parse('file.log', pattern, cast=caster_dict):
    print(i)
    # {'time': datetime.datetime(2020, 7, 22, 17, 33, 12, 554282, tzinfo=tzoffset(None, 28800)), 'level': 10, 'message': '璋冭瘯娑堟伅'}
```

`logger.parse()` 没有参数 `encoding`，测试解析中文会乱码

## 异常追溯

loguru 集成了一个名为 better_exceptions 的库，不仅能够将异常和错误记录，并且还能对异常进行追溯，这里是来自一个官网的例子

只需要添加参数 `backtrace=True` 和 `diagnose=True` 就会显示整个堆栈跟踪，包括变量的值

```Python
import os
import sys

from loguru import logger

logger.add(os.path.expanduser("~/Desktop/exception_log.log"), backtrace=True, diagnose=True)

def func(a, b):
    return a / b

def nested(c):
    try:
        func(5, c)
    except ZeroDivisionError:
        logger.exception("What?!")

if __name__ == "__main__":
    nested(0)
```

使用`catch()`装饰器 或 上下文管理器

```Python
from loguru import logger


@logger.catch
def func(x, y, z):
    return 1 / (x + y + z)


if __name__ == '__main__':
    func(0, 1, -1)
```

或者上下文管理器

```Python
from loguru import logger


def func(x, y, z):
    return 1 / (x + y + z)


with logger.catch():
    func(0, 1, -1)
```

## 与 Logging 兼容

```Python
import logging.handlers
import os
import sys

from loguru import logger

LOG_FILE = os.path.expanduser("~/Desktop/testlog.log")
file_handler = logging.handlers.RotatingFileHandler(LOG_FILE, encoding="utf-8")
logger.add(file_handler)
logger.debug("hello, world")
```

在之前基于 logging 写好的模块中集成 loguru，只要重新编写一个继承自 logging.Handler 类并实现了 emit() 方法的 Handler 即可。

```Python
import logging.handlers
import os
import sys

from loguru import logger

class InterceptHandler(logging.Handler):
    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

logging.basicConfig(handlers=[InterceptHandler()], level=0)

def func(a, b):
    return a / b

def nested(c):
    try:
        func(5, c)
    except ZeroDivisionError:
        logging.exception("What?!")

if __name__ == "__main__":
    nested(0)
```

后结果同之前的异常追溯一致。而我们只需要在配置后直接调用 logging 的相关方法即可，减少了迁移和重写的成本。

## 参考资料

[loguru 简单方便的 Python 日志记录管理模块](https://cloud.tencent.com/developer/article/1835774)

