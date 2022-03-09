import pickle
import os

rootPath = r'D:\OneDrive - The Ohio State University\choroColorRead'
# read detection results from pickle file

with open(rootPath + '\\' + 'sp_300_spatialPattern_images_quantiles.pickle', 'rb') as f:
	pickleRead = pickle.load(f)

print('test')