import requests

params = {'username': 'Ryan', 'password': 'password'}
# 向欢迎页面发送一个登录参数
r = requests.post("http://pythonscraping.com/pages/cookies/welcome.php", params)
print("Cookie is set to:")
# 从请求结果中获取cookie，打印登录状态的验证结果，然后再通过cookies参数把cookie发送到简介页面
print(r.cookies.get_dict())
print("-----------")
print("Going to profile page...")
r = requests.get("http://pythonscraping.com/pages/cookies/profile.php", cookies=r.cookies)
print(r.text)