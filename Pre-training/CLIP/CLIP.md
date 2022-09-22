

## CLIP

https://github.com/openai/CLIP

https://huggingface.co/openai/clip-vit-large-patch14

- The base model uses a ViT-L/14 Transformer architecture as an image encoder and uses a masked self-attention Transformer as a text encoder. These encoders are trained to maximize the similarity of (image, text) pairs via a contrastive loss.

- The original implementation had two variants: one using a ResNet image encoder and the other using a Vision Transformer. This repository has the variant with the Vision Transformer.

https://huggingface.co/openai/clip-vit-base-patch16

- The base model uses a ViT-B/16 Transformer architecture as an image encoder and uses a masked self-attention Transformer as a text encoder. These encoders are trained to maximize the similarity of (image, text) pairs via a contrastive loss.

- The original implementation had two variants: one using a ResNet image encoder and the other using a Vision Transformer. This repository has the variant with the Vision Transformer.

https://huggingface.co/openai/clip-vit-base-patch32

- The model uses a ViT-B/32 Transformer architecture as an image encoder and uses a masked self-attention Transformer as a text encoder. These encoders are trained to maximize the similarity of (image, text) pairs via a contrastive loss.

- The original implementation had two variants: one using a ResNet image encoder and the other using a Vision Transformer. This repository has the variant with the Vision Transformer.

用法
```
from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities

```

CLIPTextModel用法

```
>>> from transformers import CLIPTokenizer, CLIPTextModel

  

>>> model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")

>>> tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

  

>>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

  

>>> outputs = model(**inputs)

>>> last_hidden_state = outputs.last_hidden_state

>>> pooled_output = outputs.pooler_output # pooled (EOS token) states
```

last_hidden_state=CLIPTextTransformer(),分解：
- encoder_outputs = CLIPEncoder()
- nn.LayerNorm(encoder_outputs.last_hidden_state)

## Multilingual-CLIP

OpenAI CLIP **text encoders** for any language

https://github.com/FreddeFrallan/Multilingual-CLIP

https://huggingface.co/M-CLIP/M-BERT-Base-ViT-B

包括了多个预训练模型：

[LABSE Vit-L/14](https://huggingface.co/M-CLIP/LABSE-Vit-L-14)

[XLM-R Large Vit-B/32](https://huggingface.co/M-CLIP/XLM-Roberta-Large-Vit-B-32)

[XLM-R Large Vit-L/14](https://huggingface.co/M-CLIP/XLM-Roberta-Large-Vit-L-14)

[XLM-R Large Vit-B/16+](https://huggingface.co/M-CLIP/XLM-Roberta-Large-Vit-B-16Plus)

详细的可以看github项目

https://huggingface.co/M-CLIP/M-BERT-Base-ViT-B

用法

```
from multilingual_clip import pt_multilingual_clip

import transformers

  

texts = [

'Three blind horses listening to Mozart.',

'Älgen är skogens konung!',

'Wie leben Eisbären in der Antarktis?',

'Вы знали, что все белые медведи левши?'

]

model_name = 'M-CLIP/XLM-Roberta-Large-Vit-L-14'

# Load Model & Tokenizer

model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(model_name)

tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

model = model.to(torch_device)

embeddings = model.forward(texts, tokenizer) # [4,768]

print("Text features shape:", embeddings.shape)

import torch
import clip
import requests
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
image = preprocess(image).unsqueeze(0).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)

print("Image features shape:", image_features.shape)

```

https://huggingface.co/sentence-transformers/clip-ViT-B-32-multilingual-v1

- Transformer model: DistilBertModel

## Taiyi

https://huggingface.co/IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese

- This model is a Chinese CLIP model trained on [Noah-Wukong Dataset(100M)](https://wukong-dataset.github.io/wukong-dataset/) and [Zero(23M)](https://zero.so.com/). We use ViT-L-14 from [openAI](https://github.com/openai/CLIP) as image encoder and Chinese pre-trained language model [chinese-roberta-wwm-large](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large) as text encoder. We freeze the image encoder and only finetune the text encoder. The model was first trained 10 epochs on wukong and then train another 12 epochs on wukong and zero.

https://huggingface.co/IDEA-CCNL/Taiyi-CLIP-Roberta-102M-Chinese

- This model is a Chinese CLIP model trained on [Noah-Wukong Dataset(100M)](https://wukong-dataset.github.io/wukong-dataset/) and [Zero(23M)](https://zero.so.com/). We use ViT-B-32 from [openAI](https://github.com/openai/CLIP) as image encoder and Chinese pre-trained language model [chinese-roberta-wwm](https://huggingface.co/hfl/chinese-roberta-wwm-ext) as text encoder. We freeze the image encoder and only finetune the text encoder. The model was trained for 24 epochs and it takes about 10 days with 16 A100 GPUs.

用法

```
from PIL import Image
import requests
import clip
import torch
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer
from transformers import CLIPProcessor, CLIPModel
import numpy as np

query_texts = ["一只猫", "一只狗",'两只猫', '两只老虎','一只老虎']  # 这里是输入文本的，可以随意替换。
# 加载Taiyi 中文 text encoder
text_tokenizer = BertTokenizer.from_pretrained("IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese")
text_encoder = BertForSequenceClassification.from_pretrained("IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese").eval()
text = text_tokenizer(query_texts, return_tensors='pt', padding=True)['input_ids']

url = "http://images.cocodataset.org/val2017/000000039769.jpg"  # 这里可以换成任意图片的url
# 加载CLIP的image encoder
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")  
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
image = processor(images=Image.open(requests.get(url, stream=True).raw), return_tensors="pt")

with torch.no_grad():
    image_features = clip_model.get_image_features(**image)
    text_features = text_encoder(text).logits
    # 归一化
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)
    # 计算余弦相似度 logit_scale是尺度系数
    logit_scale = clip_model.logit_scale.exp()
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    print(np.around(probs, 3))

```