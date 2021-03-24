文本摘要

假设摘要其实就是基于句子对读者的重要性和意义，对句子进行排序。

在通常情况下，有相对较多实体和名词的句子比其他句子更重要。

```python
import sys
f = open('nyt.txt', 'r')
news_content = f.read()

import nltk
results=[]
for sent_no,sentence in enumerate(nltk.sent_tokenize(news_content)):
    no_of_tokens=len(nltk.word_tokenize(sentence))
    # Let's do POS tagging
    tagged=nltk.pos_tag(nltk.word_tokenize(sentence))
    # Count the no of Nouns in the sentence
    no_of_nouns=len([word for word,pos in tagged if pos in ["NN","NNP"] ])
    #Use NER to tag the named entities.
    ners=nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sentence)), binary=False)
    no_of_ners= len([chunk for chunk in ners if hasattr(chunk, 'node')])
    score=(no_of_ners+no_of_nouns)/float(no_of_toekns)
    results.append((sent_no,no_of_tokens,no_of_ners,\
no_of_nouns,score,sentence))

for sent in sorted(results,key=lambda x: x[4],reverse=True):
    print sent[5]
```

使用TF-IDF

```python
>>>import nltk
>>>from sklearn.feature_extraction.text import TfidfVectorizer
>>>results=[]
>>>sentences=nltk.sent_tokenize(news_content)
>>>vectorizer = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True)
>>>sklearn_binary=vectorizer.fit_transform(sentences)
>>>print countvectorizer.get_feature_names()
>>>print sklearn_binary.toarray()
>>>for sent_no,i in enumerate(sklearn_binary.toarray()):
>>>	results.append(sent_no,i.sum()/float(len(i.nonzero()[0])))
```

