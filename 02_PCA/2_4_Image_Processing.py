#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 14:15:21 2017

@author: fabian
"""

# 2.4 a)
import glob, os
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image as pilImage
from sklearn.feature_extraction import image as skImage
from sklearn.decomposition import PCA

loc = ["./imgpca/n*.jpg", "./imgpca/b*.jpg"]
#iterate over both image sets
data = []
for idx in [0,1]:
	path = loc[idx]
	tmpData = np.empty((0,256))
	#iterate over all files of the set
	for file in glob.glob(path):
		#load the image
		img = np.array(pilImage.open(file))
		#extract 500 patches of size 16x16
		patch = np.array(skImage.extract_patches_2d(img, (16,16), max_patches=500))
		#append patch to the data array
		tmpData = np.vstack([tmpData, patch.reshape(500,256)])
	data.append(tmpData)
	
# 2.4 b)
pcs = []
pcvs = []
labels = ["nature", "buildings"]
for idx in [0,1]:
	#compute the PCA
	pca = PCA()
	pca.fit(data[idx])
	#get the results
	pcs.append(pca.components_)
	pcvs.append(pca.explained_variance_ratio_)
	
	#plot the first 24 pcs
	for i in range(0, 24):
		plt.subplot(4,6,i+1)
		patch = np.array(pcs[idx][i,:]).reshape(16,16)
		plt.imshow(patch, cmap="gray")
		plt.axis('off')
		plt.title(i+1)
		
	plt.suptitle(labels[idx] +  ' - first 24 PCs')
	plt.show()

# as the 2 figures for the first 24 pcs for nature and building images 
# show, they are very different

# 2.4 c)
cnt = 1
for idx in [0,1]:
	plt.subplot(2,2,cnt)
	plt.plot(pcvs[idx], ".-")
	plt.title(labels[idx] + ' - all')
	cnt += 1
	
	plt.subplot(2,2,cnt)
	plt.plot(pcvs[idx][:10], ".-")
	plt.title(labels[idx] + ' - first 10')
	cnt += 1

plt.suptitle('scree plot of pcvs')
plt.show()

# We would keep the first 5 PCs
# compression ratio:
# n_components *(N_Patches + patchSize) / (N_Patches * patchSize)
#
# For the nature set with 5 PCs and 5000 patches that would yield:
# 5 * (5000 + 256) / (5000 * 256) = 0.0205 (or inverse 48.706)
# For the building set with 5 PCs and 6500 patches that would yield:
# 5 * (6500 + 256) / (6500 * 256) = 0.0203 (or inverse 49.260)

# 2.4 d)

#3 images from both datasets
imgs = ["./imgpca/n2.jpg","./imgpca/n7.jpg","./imgpca/n9.jpg","./imgpca/b2.jpg","./imgpca/b7.jpg","./imgpca/b9.jpg"]
for idx in range(len(imgs)):
	img = np.array(pilImage.open(imgs[idx]))
	height = img.shape[0]
	width = img.shape[1]
	count = 1
	for h in [1,2,4,8,16,100]:
		#only use blocks that fit in the image
		imgRecNat = np.zeros(shape=((height//16)*16,(width//16)*16))
		imgRecBui = np.zeros(shape=((height//16)*16,(width//16)*16))
		#reconstruct all the blocks
		for i in range(0, height - 16, 16):
			for j in range(0, width - 16, 16):
				patches = img[i:i+16,j:j+16].flatten()
				#reconstruct with the nature pcs
				imgRecNat[i:i+16,j:j+16] = np.sum(np.dot(pcs[0][:h], patches) * pcs[0][:h].T, axis=1).reshape(16,16)
				#reconstruct with the building pcs
				imgRecBui[i:i+16,j:j+16] = np.sum(np.dot(pcs[1][:h], patches) * pcs[1][:h].T, axis=1).reshape(16,16)
		plt.subplot(2, 6, count)
		plt.imshow(imgRecNat, cmap="gray")
		plt.title(str(h) + 'n PCs')
		plt.axis('off')
		plt.subplot(2, 6, count+6)
		plt.imshow(imgRecBui, cmap="gray")
		plt.title(str(h) + 'b PCs')
		plt.axis('off')
		count += 1
	plt.suptitle(imgs[idx])
	plt.show()

# The reconstruction is somewhat better when the PCs that were derived from the same imageset are used.
# The influence of the chosen PCs is most visible when only few PCs are used for the reconstruction.
# With a growing number of PCs the results of the reconstruction become more and more alike
