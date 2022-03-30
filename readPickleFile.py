import pickle
import os

rootPath = r'C:\Users\jiali\Desktop\choroColorRead\mapAreaDetection\groundTruthMapArea'
# read detection results from pickle file

with open(rootPath + '\\' + 'ohioBoundAnnotation.pickle', 'rb') as f:
	pickleRead = pickle.load(f)

print('test')


