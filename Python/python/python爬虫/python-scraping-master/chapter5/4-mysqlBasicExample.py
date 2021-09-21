import pymysql
# 连接对象conn
conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', password=None, db='mysql')
# 光标对象cur
cur = conn.cursor()
cur.execute("USE scraping")
cur.execute("SELECT * FROM pages WHERE id=1")
# 获取查询结果
print(cur.fetchone())
cur.close()
conn.close()