## 基础服务代码

```Python
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union, List
from fastapi.encoders import jsonable_encoder
import requests
import uvicorn

class RequestItem(BaseModel):
    org_id: str
    msg_id: str
    sender_id: str
    msg: List[str]
    
class RasaRequestItem(BaseModel):
    sender: str
    message: str
    metadata: RequestItem


class ResponseItem(BaseModel):
    status: str
    org_id: str
    msg_id: str
    sender_id: str
    response: List[str]


app = FastAPI()

@app.get("/")
def check_health():
    return {"message": "Hello World"}


@app.post("/rasa", response_model=ResponseItem)
def rasa_main(item:RequestItem):
    logger.info(f"request is {item}")
    user_id, org_id, msg_id = item.sender_id, item.org_id, item.msg_id
    msg = item.msg
    message = msg[-1][4:] if len(msg) > 0 else ""
    data = RasaRequestItem(sender=user_id, message=message, metadata=item)
    # 其它逻辑...

    response=ResponseItem(
        status=str(status),
        org_id=org_id,
        msg_id=msg_id,
        sender_id=user_id,
        response=response
        )
    response=jsonable_encoder(response)
    logger.info(f"response is {response}")
    return response


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6888)

```

## 启动

`uvicorn main:app --reload`，其中main是py文件名

