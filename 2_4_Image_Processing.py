#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 14:15:21 2017

@author: fabian
"""
import glob, os
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image as pilImage
from sklearn.feature_extraction import image as skImage
from sklearn.decomposition import PCA

natureArray = np.empty((0,256))

for file in glob.glob("imgpca/b*.jpg"):
    img = pilImage.open(file)
    imgArray = np.array(img)
    tempPatches = skImage.extract_patches_2d(imgArray, (16,16), max_patches=500)
    patch = np.array(tempPatches).reshape(500,256)
    natureArray = np.vstack([natureArray, patch])

print(natureArray.shape)

pca = PCA()
pca.fit(natureArray)
np.set_printoptions(threshold=np.nan)
#print(pca.explained_variance_ratio_)
#print("nxt")
pc = pca.components_
pcRatio = pca.explained_variance_ratio_
print(len(pc))

for i in range(0, 24):
    plt.subplot(4,6,i+1)
    #print(pc[i,:])
    patch = np.array(pc[i,:]).reshape(16,16)
    plt.imshow(patch, cmap="gray")
plt.show()

#plt.plot(pcRatio, ".-")
#plt.show()
#
#plt.plot(pcRatio[:10], ".-")
#plt.show

# We would keep the first 5 PCs
# compression ratio:
# (N_Patches * n_components) + n_components * patchSize / (N_Patches * patchSize)


#b2, b7, b9
image1 = pilImage.open("imgpca/b2.jpg")
imgArray = np.array(image1)
height = imgArray.shape[0]
width = imgArray.shape[1]
print(width, height)

width = 35
height = 35
x = 1

count = 1

patches = np.empty((0,(width%16)*16))
for i in range(0, height - 16, 16):
    for j in range(0, width - 16, 16):
        patches = imgArray[j:j+16,i:i+16]
        print(patches.shape)
        
        newPatch = np.dot(pc[1].T, patches.flatten()) * pc[1]
        newPatch = newPatch.reshape(16,16)
        print(i, j, count)
#        print(newPatch)
        plt.subplot(2, 2, count)
        plt.imshow(newPatch, cmap="gray")
        count += 1

plt.show
