# -*- coding:utf-8 -*-
from matplotlib import pyplot as plt
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np

dataset = dset.STL10(root='../stl10_root', download=True)

# image = np.asarray(dataset[10][0])

image1 = np.asarray(dataset[2][0]) / 255.
image2 = np.asarray(dataset[19][0]) / 255.
image3 = 0.7 * image1 + 0.3 * image2

plt.subplot(131)
plt.imshow(image1)
plt.subplot(132)
plt.imshow(image2)
plt.subplot(133)
plt.imshow(image3)

plt.show()

for i, data in enumerate(dataset):
    print(i, data[1])
    if i == 100:
        break

