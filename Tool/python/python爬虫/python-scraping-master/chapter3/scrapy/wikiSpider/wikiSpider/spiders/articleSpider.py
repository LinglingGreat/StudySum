from scrapy.spiders import CrawlSpider, Rule
from wikiSpider.items import Article
from scrapy.linkextractors import LinkExtractor

class ArticleSpider(CrawlSpider):
    # name决定了调用爬虫所需输入的条目名称，这里是在wikiSpider主目录中用命令scrapy crawl article运行这个类
    # 这个爬虫先进入start_urls里面的页面，收集信息，然后停止。
    # 定义规则让scrapy可以在每个页面查找URL链接
    name = "article"
    allowed_domains = ["en.wikipedia.org"]
    start_urls = ["http://en.wikipedia.org/wiki/Python_%28programming_language%29"]
    rules = [
        Rule(LinkExtractor(allow=('(/wiki/)((?!:).)*$'),), callback="parse_item", follow=True)
    ]

    def parse_item(self, response):
        item = Article()
        title = response.xpath('//h1/text()')[0].extract()
        print("Title is: "+title)
        item['title'] = title
        return item
