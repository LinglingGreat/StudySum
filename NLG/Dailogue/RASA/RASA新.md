


# 训练

`rasa train`

不存cache，加了cache会减少训练成本，但是不方便迁移到别的机器运行

`RASA_MAX_CACHE_SIZE=0 rasa train`

# 启动rasa api

使用CPU/GPU

```bash
export CUDA_VISIBLE_DEVICES=''
# export ACTION_SERVER_SANIC_WORKERS=5
# export SANIC_WORKERS=5
#export TF_FORCE_GPU_ALLOW_GROWTH=true
```

rasa run --enable-api

rasa run actions

启动之后的访问地址："[http://0.0.0.0:5005/webhooks/rest/webhook](http://0.0.0.0:5005/webhooks/rest/webhook "http://0.0.0.0:5005/webhooks/rest/webhook")"

其中rest指的是`RestInput`（from rasa.core.channels.rest import RestInput），还有一个`CallbackInput`（from rasa.core.channels.calback import CallbackOutput），他们提供了一个可以发送消息和接受消息的URL。

参考：[https://rasa.com/docs/rasa/connectors/your-own-website/](https://rasa.com/docs/rasa/connectors/your-own-website/ "https://rasa.com/docs/rasa/connectors/your-own-website/")

这两个都可以自定义。参考：[https://rasa.com/docs/rasa/connectors/custom-connectors/](https://rasa.com/docs/rasa/connectors/custom-connectors/ "https://rasa.com/docs/rasa/connectors/custom-connectors/")

如果需要传入自定义的字段，可以自定义Channel，实现get\_metadata方法

```python
class MyIO(RestInput):
    """A custom http input channel.

    This implementation is the basis for a custom implementation of a chat
    frontend. You can customize this to send messages to Rasa and
    retrieve responses from the assistant."""

    @classmethod
    def name(cls) -> Text:
        return "myio"

    def get_metadata(self, request: Request) -> Optional[Dict[Text, Any]]:
        """Extracts additional information from the incoming request.

         Implementing this function is not required. However, it can be used to extract
         metadata from the request. The return value is passed on to the
         ``UserMessage`` object and stored in the conversation tracker.

        Args:
            request: incoming request with the message of the user

        Returns:
            Metadata which was extracted from the request.
        """
        
        metadata = request.json.get("metadata", {})
        return metadata
```

然后在`credentials.yml`中加上这个类的位置：

```yaml
addons.custom_channel.MyIO:
```



# RASA耗时分析

**问题定位**

先在rasa run命令中加上`--debug`，打印出更多日志，发现日志`Calling action endpoint to run action`和`Policy prediction ended with events`之间，`Predicted next action 'action_listen' with confidence 1.00.`和`Policy prediction ended with events`之间耗时，但是中间没有详细日志。

然后去源码里找到日志所在位置，发现问题出在rasa.core.processor.py的\_run\_action函数，看代码可能耗时的地方就在`temporary_tracker.update_with_events`，以及`action.run`函数中的`self.action_endpoint.request`（rasa.core.actions.action.py.RemoteAction）。前者是更新tracker中的events（最耗时）；后者是请求action的HTTP服务（偶尔耗时），不好定位。

更新：tracker耗时的话可能是因为tracker太多导致的。

为进一步确认问题，采取(1)在可能耗时的地方加上日志，(2)查看历史报错信息对应的sender\_id。发现绝大部分报错都是同一个用户，他的tracker events有7000多个，和其它用户差距很大。

**问题解决**

需要保证tracker较少。tracker涉及redis存储，搜索`rasa tracker too much`问题+查询官方文档找到解决方案：redis设置加上参数`record_exp: 180`，意思是`here record_exp is a time in seconds to clear all events and slots in tracker of the conversation with last message time more then 180 seconds`

为快速解决问题，将时间改成60s。

**问题确认**

1.  用户聊了一会不再报错
2.  查看日志确认是`update_with_events`和`self.action_endpoint.request`最耗时

