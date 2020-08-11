import json
import re

'''
illegal_char=[']', '[', '（', '）', '{', '}', '：', '《', '》','！','？','_','\n']
dataset_pre=open('./dataset/dataset.txt','r',encoding='utf-8')
dataset_post=open('./dataset/data.txt','w',encoding='utf-8')
for line in dataset_pre:
    ss=re.split('[，。]',line)
    for s in  ss:
        for ic in illegal_char:
            s=s.replace(ic,'')
        if len(s)==7:
            dataset_post.write(s+'\n')
'''

dataset=open('./dataset/data.txt','r',encoding='utf-8')
counted_c={}

for line in dataset:
    for c in line:
        if c!='\n':
            if c in counted_c:
                counted_c[c]+=1
            else:
                counted_c[c]=1
erase=[]
for c in counted_c:
    if counted_c[c]<=2:
        erase.append(c)
for c in erase:
    del counted_c[c]

counted_c=sorted(counted_c.items(),key=lambda x: -x[-1])
cs,_=zip(*counted_c)
c2n = dict((c, i + 1) for i, c in enumerate(cs))
n2c = dict((i, c) for i, c in enumerate(cs))

char2num=json.dumps(c2n)
num2char=json.dumps(n2c)

with open('./dataset/char2num.json','w',encoding='utf-8') as f:
    f.write(char2num)

with open('./dataset/num2char.json','w',encoding='utf-8') as f:
    f.write(num2char)
