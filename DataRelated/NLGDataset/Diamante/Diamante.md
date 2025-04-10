---
title: Diamante
created: 2022-09-05
tags: 百度

---


论文：Towards Boosting the Open-Domain Chatbot with Human Feedback


基于反馈机制

数据集链接：https://www.luge.ai/#/luge/dataDetail?id=52

Diamante是百度持续进化的开放域闲聊AI辅助人工标注的数据集，含有机器生成的候选回复，以及人工选择/修改的最终回复集，用于全方面提高对话模型的闲聊能力。

示例

```
{
  "id": "dialogue-00000",
  "conversation": [
    {
      "role": "speaker1",
      "utterance": "最近天气好好，想出去拍照片",
      "response_candidates": [
        "去哪，带上我",
        "想去哪拍？",
        "去哪里拍，我也想拍",
        "拍人还是拍景呢",
        "我也想，可是不知道有啥好地方",
        "你是摄影专业的么",
        "我也有这个打算，不过我的相机是渣渣"
      ]
    },
    {
      "role": "speaker2",
      "utterance": "我也有这个打算，不过我的相机是渣渣",
      "response_candidates": [
        "我也渣，我都是用手机拍的，相机还没买",
        "我相机也是个渣渣",
        "你用相机拍，我用手机帮你拍，哈哈",
        "没事啊，我也是菜鸟",
        "手机也可以拍啊，",
        "可以一起拍啊，我是手机，没相机，你是什么相机啊？",
        "我也就入门机"
      ]
    },
    {
      "role": "speaker1",
      "utterance": "哈哈哈我也不是专业的，我就是瞎拍，多拍拍就好了",
      "response_candidates": [
        "那你打算什么时候去拍啊",
        "你是什么相机啊",
        "我都好久没拍了",
        "哈哈，我是纯业余的",
        "哈哈，我都没怎么好好拍过",
        "你打算什么时候，在哪拍？",
        "好，我也是业余的，可以加个微信吗"
      ]
    },
    {
      "role": "speaker2",
      "utterance": "那你打算什么时候去拍啊",
      "response_candidates": [
        "这周没时间了，下周吧，你想去哪拍？",
        "过两天，这两天没时间",
        "这两天有时间就去",
        "今天啊，天气这么好，明天也行啊",
        "周末吧，平时上班的",
        "我这几天都在上班，周末有空",
        "我想拍的时候就去了啊"
      ]
    },
    {
      "role": "speaker1",
      "utterance": "就这两天吧，刚好是清明节",
      "response_candidates": [
        "好啊，去哪里拍啊",
        "你有没有想去的地方？",
        "那你是在哪里拍啊",
        "我是打算明天下午去拍的",
        "嗯，我也差不多，清明后就要开始忙了",
        "哦哦，我清明不在长沙",
        "去拍的地方选好了没，我还不知道去拍什么"
      ]
    },
    {
      "role": "speaker2",
      "utterance": "嗯，我也差不多，清明后就要开始忙了",
      "response_candidates": [
        "那你有没有想好要去哪",
        "你是要拍人还是拍风景呀？",
        "你是在哪边？",
        "那要一起么？",
        "那一起吧！我加你微信",
        "那我们可以一起呀！你是在哪个区？",
        "你要去吗，那你是男的女的"
      ]
    },
    {
      "role": "speaker1",
      "utterance": "我是学生所以还好哈哈哈，平时间都比较多的",
      "response_candidates": [
        "嗯，那你是在哪里读书啊",
        "真好，我已经毕业快一年了",
        "羡慕，我都已经毕业了",
        "好吧，我已经毕业了，我打算后天去拍",
        "那你比我好多了，我工作了的",
        "好吧，真幸福，我是打算这几天抽一天去拍的",
        "那挺好的，有时间可以一起出去，我在南山这边"
      ]
    },
    {
      "role": "speaker2",
      "utterance": "羡慕，我都已经毕业了",
      "response_candidates": [
        "我也快了，马上大四",
        "我还想赶紧毕业呢",
        "我也想快点毕业，然后赚钱",
        "没事的，可以自己赚钱自己花！我就想这样！嘿嘿嘿",
        "我现在都大二了也快啦",
        "我还要读五年呢，也很苦",
        "那是该努力赚钱咯"
      ]
    },
    {
      "role": "speaker1",
      "utterance": "我也想快点毕业，然后赚钱",
      "response_candidates": [
        "等你毕业就不这么想了，还是在学校好，你是大几了",
        "你还有多久毕业啊",
        "毕业之后你就会怀念上学的时候，还是好好珍惜当下的生活吧",
        "你是在长沙读书吗？",
        "我还想在读大学呢",
        "你大几啊？毕业就怀念大学了",
        "毕业以后压力会大很多的"
      ]
    },
    {
      "role": "speaker2",
      "utterance": "毕业之后你就会怀念上学的时候，还是好好珍惜当下的生活吧",
      "response_candidates": [
        "我知道，可是我现在很迷茫，都不知道自己该干什么",
        "我知道啊，但是我现在就是想赚钱",
        "那你是在工作了吗",
        "是啊，但是现在也挺忙的",
        "嗯嗯，珍惜现在",
        "当下就要好好学习了",
        "是的吧，但是我感觉自己现在就很不珍惜"
      ]
    },
    {
      "role": "speaker1",
      "utterance": "哈哈哈，每个阶段都会怀念过去",
      "response_candidates": [
        "嗯，是的，你大几了",
        "嗯，人都是这样的，所以要好好的珍惜现在",
        "也是，都是这么过来的",
        "嗯，你还不睡觉啊",
        "是呀，所以现在好好生活就好了",
        "是啊，所以我打算有机会就回去学校",
        "是的，我现在就很怀念大学的时候"
      ]
    },
    {
      "role": "speaker2",
      "utterance": "嗯，人都是这样的，所以要好好的珍惜现在",
      "response_candidates": [
        "嗯嗯，会的，我现在很努力的在过好每一天",
        "所以我现在要睡觉了，好困，晚安",
        "好的呢，我会的啦",
        "嗯啊，都知道这个道理，可是总是事与愿违",
        "嗯呐，好好享受现在",
        "现在也就这样吧，一天忙得要死",
        "嗯啊，现在就很好！你也是哈"
      ]
    },
    {
      "role": "speaker1",
      "utterance": "对啊，人生无常，过好每一天",
      "response_candidates": [
        "是啊，过一天少一天",
        "是的，活在当下",
        "嗯，是的，活在当下嘛",
        "是啊，活在当下",
        "嗯，你的心态很好，继续保持",
        "嗯，你是大学生？",
        "是呀，活在当下嘛"
      ]
    },
    {
      "role": "speaker2",
      "utterance": "你这么年纪轻轻就参悟生命的真谛哈哈哈",
      "response_candidates": [
        "是啊，我是一个有故事的男人",
        "因为我觉得我现在就活的很无常",
        "我是老了老了",
        "我已经是老人家了好嘛",
        "哈哈哈，你也很年轻啊",
        "因为我是小仙女，哈哈哈",
        "哈哈哈，不年轻了，只是觉得每一天都很重要"
      ]
    }
  ]
}
```

![](img/Pasted%20image%2020220906142240.png)

![](img/Pasted%20image%2020220906142431.png)

![](img/Pasted%20image%2020220906142444.png)

实验

![](img/Pasted%20image%2020220906142753.png)

![](img/Pasted%20image%2020220906142804.png)

