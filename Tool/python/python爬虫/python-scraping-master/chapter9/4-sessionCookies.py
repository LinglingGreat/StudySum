import requests

session = requests.Session()
# 会话对象会持续跟踪会话信息，像cookie,header,甚至包括运行HTTP协议的信息比如HTTPAdapter(为HTTP和HTTPS的链接会话提供统一接口)
params = {'username': 'username', 'password': 'password'}
s = session.post("http://pythonscraping.com/pages/cookies/welcome.php", params)
print("Cookie is set to:")
print(s.cookies.get_dict())
print("-----------")
print("Going to profile page...")
s = session.get("http://pythonscraping.com/pages/cookies/profile.php")
print(s.text)