# Elasticsearch

版本：7.0以上

## 基本操作

**查询所有索引：** `get  _cat/indices`

**查询某个索引的字段** ：`get gs-recommend-v2/_mappings`

新建索引

例如：

```python
put /gs-recommend-v2
{
    "mappings": {
        "properties": {
          "_update": {
            "type": "text",
            "fields": {
              "keyword": {
                "type": "keyword",
                "ignore_above": 256
              }
            }
          },
          "child_count": {
            "type": "float"
          },
          "collection_count": {
            "type": "float"
          },
          "comment_rating": {
            "type": "float"
          },
          "commentcount": {
            "type": "float"
          },
          "commentcount_within_1year": {
            "type": "float"
          },
          "districtid": {
            "type": "keyword"
          },
          "districtpath": {
            "type": "text"
          },
          "officialpoiid": {
            "type": "keyword"
          },
          "parentdistrictid": {
            "type": "keyword"
          },
          "poiscore": {
            "type": "float"
          },
          "price": {
            "type": "float"
          },
          "recallscore": {
            "type": "float"
          },
          "share_count": {
            "type": "float"
          },
          "sight_rank": {
            "type": "float"
          },
          "sortno": {
            "type": "float"
          },
          "t_score_no_add": {
            "type": "text",
            "fields": {
              "keyword": {
                "type": "keyword",
                "ignore_above": 256
              }
            }
          },
          "total_picnum": {
            "type": "float"
          },
          "totalsales": {
            "type": "float"
          },
          "uv_section": {
            "type": "text",
            "fields": {
              "keyword": {
                "type": "keyword",
                "ignore_above": 256
              }
            }
          },
          "uvsum": {
            "type": "float"
          },
          "feature": {
            "type": "dense_vector",
            "dims": 13
          }
        }
      }
}
```


### **检索数据** 

全局检索

```python
POST gs-recommend-v2/_search
{
  "query": {
    "match_all": {}
  }
}
```


条件检索

```python
POST gs-recommend-v2/_search
{
  "query": {
    "match": {"poiid": 112}
  }
}
```


向量检索

```python
POST gs-recommend-v2/_search
{
  "query": {
    "script_score": {
      "query": {
        "match_all": {}
      },
      "script": {
        "source": "cosineSimilarity(params.queryVector,doc['feature'])+1.0",
        "params": {
          "queryVector": [
            0,
            4.7418923,
            12781,
            330,
            37332,
            30236,
            5,
            3,
            1549.6129,
            13,
            0,
            0,
            42
          ]
        }
      }
    }
  }
}
```


### 删除索引和数据

删除索引：`Delete 索引名称1，索引名称2`

根据匹配条件删除数据

```python
POST 索引名称/_delete_by_query   
{
  "query":{
    "term":{
      "_id":111
    }
  }
}
```


删除所有数据

```python
POST /testindex/_delete_by_query?pretty
{
    "query": {
        "match_all": {
        }
    }
}
```


## Python操作ES

安装elasticsearch包：`pip install elasticsearch`

```python
from elasticsearch import Elasticsearch

index_name = 'testindex'
http_auth = ("username", "password")
es = Elasticsearch("access url", http_auth=http_auth)

# 批量插入数据
actions = []
action = {
        "_index": index_name,
        "_id": "111",
        "_source": {
            "poiid": 112,
            "feature": [0,4.7418923,12781,330,37332,30236,5,3,1549.6129,13,0,0,42]
        }
    }
    actions.append(action)
a = helpers.bulk(es, actions)

# 查询数据，和ES语法一样
query_json = {
     "query": {
     "match": {"poiid": 112}
   }
}
res = es.search(index=index_name, body=query_json)
hits = res['hits']['hits']
for hit in hits:
    print(str(hit['_id']) + '\t' + str(hit['_score']))
```


查询ES里的所有数据

```python
body = {
        "_source": ["_id"],
        "query": {
            "match_all": {}}
  }

query = es.search(index=index_name, body=body, scroll='5m', size=10000)
results = query['hits']['hits']  # es查询出的结果第一页
total = query['hits']['total']  # es查询出的结果总量
print(len(results), len(total))
scroll_id = query['_scroll_id']  # 游标用于输出es查询出的所有结果
espoiid_list = [doc["_id"] for doc in results]

for i in range(100):
    # scroll参数必须指定否则会报错
    query = es.scroll(scroll_id=scroll_id, scroll='5m')
    query_scroll = query['hits']['hits']
    results += query_scroll
    espoiid_list += [doc["_id"] for doc in query_scroll]
    # print(query_scroll)
    scroll_id = query['_scroll_id']
    print(len(results))
print(len(results))
espoiid_list = set(espoiid_list)
```


删除满足条件的数据

```python
body = {
        "query": {
            "terms": {
                "_id": diff_poiid
            }
        }
    }
es.delete_by_query(index=index_name, body=body)
```


## Java操作ES

```Java
@Autowired
private ElasticsearchQConfig elasticsearchQConfig;

private List<PoiRecsysPoifeaturesV2> getPoiFeatureFromEs(String districtId) throws IOException {
      EsConfig.Query query = elasticsearchQConfig.getConfig().getQuery();
      SearchRequest searchRequest = new SearchRequest(query.getIndexName());
      SearchSourceBuilder searchSourceBuilder = new SearchSourceBuilder();
      BoolQueryBuilder boolQuery = QueryBuilders.boolQuery();
      boolQuery.must(QueryBuilders.rangeQuery("t_score_no_add").gte(query.getScoreNoAdd()));
      boolQuery.filter(QueryBuilders.rangeQuery("total_picnum").gt(query.getTotalPicture()));
      if (!districtId.isEmpty()){
          boolQuery.filter(QueryBuilders.matchQuery("districtpath", districtId));
      }

//        if (districtId > 0) {
//            boolQuery.filter(QueryBuilders.matchQuery("districtpath", districtId));
//        }

      searchSourceBuilder.query(boolQuery)
              .sort("recallscore", SortOrder.DESC)
              .size(query.getSize());

      searchRequest.source(searchSourceBuilder);
      SearchResponse searchResponse = restHighLevelClient.search(searchRequest, RequestOptions.DEFAULT);
      SearchHits hits = searchResponse.getHits();
      org.elasticsearch.search.SearchHit[] searchHits = hits.getHits();
      List<PoiRecsysPoifeaturesV2> res = new ArrayList<>(searchHits.length);

      for (org.elasticsearch.search.SearchHit hit : searchHits) {
          res.add(JSON.parseObject(hit.getSourceAsString(), PoiRecsysPoifeaturesV2.class));
      }

      return res;
  }

```


### BoolQuery 的用法与特性

BoolQuery 作为一个主要的Query结构一般用于整个Query的根。Query的结果会根据score默认排序

```Java
searchSourceBuilder.query(QueryBuilders.boolQueryBuilder()); 
searchRequest.source(searchSourceBuilder); 
```


BoolQuery 拥有4个不同的Query类型。

```Java
// 必须匹配，匹配会增加score。MatchQuery 这里表示"名字"必须包含“大卫”或者“David”其中一个, 多个匹配会增加其结果的score。
boolQueryBuilder.must(QueryBuilders.matchQuery("名字", "大卫 David")) 
// 每一个query类型都可以添加多个子Query，其关系是“AND”。并且可以是BoolQuery
boolQueryBuilder.must(QueryBuilders.boolQuery()) 
// 必须匹配，匹配不影响score。TermsQuery 这里表示"id"必须等于“123”或者“321”
boolQueryBuilder.filter(QueryBuilders.termsQuery("id", "123"， "321"))
// 非必须匹配，匹配增加score。
boolQueryBuilder.should()
// 设定最少匹配“should”的数量，如果没有规定“must”/“filter”默认值是1，有的话默认值是0
boolQueryBuilder.minimalShouldMatch(1) 
// 必须不包含，不影响score。
boolQueryBuilder.must_not()
```


默认的排序是按照score来进行的。我们也可以自己添加用来排序的键。

```Java
// 按日期的排序，默认是正序
searchSourceBuilder.sort(SortBuilders.fieldSort("日期"))
// 按score排序，默认是倒序。当多个sort被提供的时候，按其先后顺序优先排列
searchSourceBuilder.sort(SortBuilders.scoreSort())
```


当搜索请求中包含sort时，搜索结果中会包含sortValues。例如以上sort的结果中，每一个hit会有如下Array

```Java
result.getHits().getHits()[0].getSortValues();
// ["2020-01-01", 2.2]
```


## 参考资料

[【最佳实践】阿里云 Elasticsearch 向量检索4步搭建“以图搜图”搜索引擎](https://developer.aliyun.com/article/750481)

[Elasticsearch 索引创建 / 数据检索](https://segmentfault.com/a/1190000018661035)

[python 查询 elasticsearch 常用方法（Query DSL）](https://www.cnblogs.com/ExMan/p/11323984.html)

[Elastic Search 搜索](https://sourberrycat.com/elastic-search/elastic-search-learnings/)

[Java操作Elasticsearch6实现单个字段多值匹配](https://blog.csdn.net/hu_zhiting/article/details/110956879)


