**用Metaclass元类**

创建

```
from django.db import models

class Employee(models.Model):
    name = models.CharField(maxlength = 50)    
    age  = models.IntegerField()
    #其他代码略#
```

在Employee中没有看到Metaclass, 就去父类Model中去寻找，找到了metaclass ，叫做ModelBase。ModelBase是为了实现ORM，就是对象和关系数据库的映射。

```
class Model(metaclass=ModelBase)：
    #其他代码略
```

执行数据库操作：

```
employee = Employee(name="andy",age=20)  
employee.save()
```

