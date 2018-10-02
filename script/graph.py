# -*- coding:utf-8 -*-
import os
from matplotlib import pyplot as plt
import pandas as pd
from PIL import Image

'''
df_womix = pd.read_csv('classifier.log')
df_wmix = pd.read_csv('classifier_mix.log')

plt.plot(df_womix['epoch'], df_womix['train/accuracy'], label='w/o mix: train', marker='o')
plt.plot(df_wmix['epoch'], df_wmix['train/accuracy'], label='w/ mix: train', marker='o')

plt.plot(df_womix['epoch'], df_womix['test/accuracy'], label='w/o mix: test', marker='o')
plt.plot(df_wmix['epoch'], df_wmix['test/accuracy'], label='w/ mix: test', marker='o')

plt.legend()
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()
'''

#images = ['dcgan/fake_samples_epoch_{:03d}.png'.format(i) for i in range(1, 51, 1)]
#print(images)

image_list = []
for i in range(50):
    image_name = 'dcgan/fake_samples_epoch_{:03d}.png'.format(i+1)
    image = Image.open(image_name)
    image_list.append(image)

image_list[0].save('dcgan/fake_samples.gif', save_all=True, append_images=image_list[1:],
                   optimiza=False, duration=10, loop=0)


for i in range(50):
    image_name = 'dcgan_mix/fake_samples_epoch_{:03d}.png'.format(i+1)
    image = Image.open(image_name)
    image_list.append(image)

image_list[0].save('dcgan_mix/fake_samples.gif', save_all=True, append_images=image_list[1:],
                   optimiza=False, duration=10, loop=0)
