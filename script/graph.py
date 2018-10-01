# -*- coding:utf-8 -*-
from matplotlib import pyplot as plt
import pandas as pd


df_womix = pd.read_csv('train_womix.log')
df_wmix = pd.read_csv('train_wmix.log')

plt.scatter(df_womix['epoch'], df_womix['train/accuracy'], label='w/o mix: train')
plt.scatter(df_wmix['epoch'], df_wmix['train/accuracy'], label='w/ mix: train')

plt.scatter(df_womix['epoch'], df_womix['test/accuracy'], label='w/o mix: test')
plt.scatter(df_wmix['epoch'], df_wmix['test/accuracy'], label='w/ mix: test')

plt.legend()
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()