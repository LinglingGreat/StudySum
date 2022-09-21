

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



https://huggingface.co/sentence-transformers/clip-ViT-B-32-multilingual-v1

- Transformer model: DistilBertModel

## Taiyi

https://huggingface.co/IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese

- This model is a Chinese CLIP model trained on [Noah-Wukong Dataset(100M)](https://wukong-dataset.github.io/wukong-dataset/) and [Zero(23M)](https://zero.so.com/). We use ViT-L-14 from [openAI](https://github.com/openai/CLIP) as image encoder and Chinese pre-trained language model [chinese-roberta-wwm-large](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large) as text encoder. We freeze the image encoder and only finetune the text encoder. The model was first trained 10 epochs on wukong and then train another 12 epochs on wukong and zero.

https://huggingface.co/IDEA-CCNL/Taiyi-CLIP-Roberta-102M-Chinese

- This model is a Chinese CLIP model trained on [Noah-Wukong Dataset(100M)](https://wukong-dataset.github.io/wukong-dataset/) and [Zero(23M)](https://zero.so.com/). We use ViT-B-32 from [openAI](https://github.com/openai/CLIP) as image encoder and Chinese pre-trained language model [chinese-roberta-wwm](https://huggingface.co/hfl/chinese-roberta-wwm-ext) as text encoder. We freeze the image encoder and only finetune the text encoder. The model was trained for 24 epochs and it takes about 10 days with 16 A100 GPUs.

