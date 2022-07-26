# -*- coding: utf-8 -*-
# @Author: LiLing
# @Date:   2019-08-03 00:57:55
# @Last Modified by:   LL
# @Last Modified time: 2019-08-03 00:59:57
#!/usr/bin/python
# -*- coding: utf-8 -*-
if __name__=='__main__':
    user_items=[]
    items=[]
    with open('u.data') as f:
        for line in f:
            user_items.append(line.split('\t'))

    with open('u.item') as f:
        for line in f:
            items.append(line.split('|'))
    print('user_items[0] = ',user_items[0])
    print('items[0] = ',items[0])

    items_hash={}
    for i in items:
        items_hash[i[0]]=i[1]

    print('items_hash[1] = ',items_hash['1'])

    for ui in user_items:
        ui[1]=items_hash[ui[1]]

    print('user_items[0] = ',user_items[0])

    with open('ratings.csv','w') as f:
        for ui in user_items:
            f.write(ui[0]+'|'+ui[1]+'|'+ui[2]+'\n')
