import json
import keras 
from keras.models import load_model
from keras.layers import Lambda,Input
from gan import GAN, dataGenerator
import numpy as np

num2char=json.load(fp=open('./dataset/num2char.json','r',encoding='utf-8'))
gan=GAN()
gan.build()
print('=== Temperature = 0.75 ===')
print(gan.sample(5, 0.75))
print('=== Temperature = 1.00 ===')
print(gan.sample(5, 1.00))
print('=== Temperature = 1.25 ===')
print(gan.sample(5, 1.25))

print(gan.generate_poem(count=8,temperature=1.00,seed='苟利国家生死以'))
