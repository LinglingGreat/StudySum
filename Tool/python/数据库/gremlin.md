```Python
from gremlin_python.driver import client
from gremlin_python.process.traversal import T

def search_from_gdb(item:RequestItem):
    dsl = item.dsl
    client_ = client.Client(Config.gdb_url, 'g', username=Config.gdb_usr, password=Config.gdb_pwd)
    callback = client_.submit(dsl).all().result()
    client_.close()
    return callback
   
from gremlin_python.process.anonymous_traversal import traversal
from gremlin_python.driver.driver_remote_connection import DriverRemoteConnection
# Plain text authentication
url = xxx
g = traversal().with_remote(DriverRemoteConnection(url, 'g', username=xxx, password=xxx))


dsl = "g.V('3').drop()"
dsl = "g.V().hasLabel('gdb_sample_person')"
dsl = "g.V().hasLabel('gdb_sample_person').has('age', gt(29))"
msg_list = str(["句子1", "句子2", "句子3"])
print(type(msg_list), repr(msg_list))
dsl = f'''g.addV("launch_origin").property(id, "1").property("content","{msg_list}").property("event_type","1").property("org_id","123").property("launch_topic", "ask_name")'''
dsl = f'''g.addE('condition').from(V('1')).to(V('2')).property('name', 'true')'''

org_id = "123"
greet_type = "1"
dsl = f"g.V().hasLabel('launch_origin').has('org', '{org_id}').has('event_type', '{greet_type}').valueMap(true)"
# dsl = f"g.V('1').outE('condition').inV()"
# dsl = "g.V('1').outE().id()"
dsl = "g.V('1').outE().valueMap(true)"
dsl = "g.V().hasLabel('launch_origin').has('org', '123').has('event_type', '1').has('topic', without('')).valueMap(true)"
```

## 参考资料

[Gremlin 常用语法总结](https://ittang.com/2018/11/15/gremlin_traversal_language/)

[Gremlin中文文档](http://tinkerpop-gremlin.cn)

