import os
import cv2
import numpy as np
import matplotlib.pyplot as plt  

import pandas as pd
import seaborn as sns
import pickle
from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely.geometry import box
import shapely.geometry

import numpy as np
from scipy.signal import argrelextrema

def rgb2Grey(dominantColor):
    rgb_weights = [0.2989, 0.5870, 0.1140]
    dominantColorGrey = int(np.dot(dominantColor, rgb_weights))
    return dominantColorGrey
def bgr2Grey(dominantColor):
    rgb_weights = [ 0.1140,0.5870, 0.2989]
    dominantColorGrey = int(np.dot(dominantColor, rgb_weights))
    return dominantColorGrey

def most_frequent(List):
    return max(set(List), key = List.count)
def unique_count_app(a):
    a = np.asarray(a)
    colors, count = np.unique(a.reshape(-1,a.shape[-1]), axis=0, return_counts=True)
    return colors[count.argmax()]

def main():
	# read image data
    imagePath = r'C:\Users\jiali\Desktop\choroColorRead\generatedMaps\quantiles'
    testImages = os.listdir(imagePath)

    # read detection results from pickle file
    detectResultsPath = r'D:\OneDrive - The Ohio State University\choroColorRead'
    detectResultFileName = 'detectResultSpatialPattern.pickle'
    with open(detectResultsPath + '\\' + detectResultFileName, 'rb') as f:
        detectResults = pickle.load(f)

    testImage = testImages[0]
    for i,testImage in enumerate(testImages): # 3,8,22,214

        if testImage[-4:] == 'json':
            continue
        elif testImage[-4:] == 'jpeg':
            detectResultFile = testImage[:-4]
        else:
            detectResultFile = testImage[:-3]
        
        print('testImage: ' + testImage)
        img = cv2.imread(imagePath + '\\'+testImage)
        cv2.imshow('test', img)

        imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        detectResult = detectResults[i]
        property = detectResult[1]
        boxes = property['rois']
        masks = property['masks']
        class_ids = property['class_ids']

        N = boxes.shape[0]
        if not N:
            print("\n*** No instances to detected by MaskRCNN *** \n")
        else:
            assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

        mapBoxList = []
        mapMaskList = []
        for i in range(N):
            if i >0:
                continue
            if class_ids[i] != 3:
                continue
            # Bounding box
            y1, x1, y2, x2 = boxes[i]
            bboxMap = shapely.geometry.box(x1, y1, x2, y2)
            mapBoxList.append(bboxMap)
            # Mask
            mask = masks[:, :, i]
            mapMaskList.append(mask)

        imgListList = imgGrey.tolist()
        imgList = [item for sublist in imgListList for item in sublist]
        bg = most_frequent(imgList)
        # imgListNoBG = [i for i in imgList if i != bg ]
        imgListNoBG = []
        for i in range(imgGrey.shape[0]):
            for j in range(imgGrey.shape[1]):
                pixel = int(imgGrey[i,j])
                point = Point(j,i)
                if pixel == bg:
                    continue
                inMappingArea = False
                for box in mapBoxList: # try both box and mask polygon
                    if box.contains(point):
                        inMappingArea = True
                        break
                if inMappingArea:
                    imgListNoBG.append(pixel)

        # imgPD = pd.Series(imgList, name = testImage)
        # sns.histplot(data = imgPD, binwidth = 1, kde = True)

        imgPDNoBG = pd.Series(imgListNoBG, name = testImage)
        kde = sns.histplot(data = imgPDNoBG, binwidth = 1, kde = True)
        plt.close()
        # line= kde.lines[0]
        # data = line.get_data()
        # xData, yData = data

        # xDataList = xData.tolist()

        # # for local maxima
        # localMax = argrelextrema(yData, np.greater)
        # localMaxList = localMax[0].tolist()
        # indiceLocalMaxima = []
        # for i in localMaxList:
        #     grey = xDataList[i]
        #     indiceLocalMaxima.append(round(grey))
        
        # # find RGB value for the grey local maxima
        # pixelDict = dict()
        # for i in range(img.shape[0]):
        #     for j in range(img.shape[1]):
        #         pixelBGR = img[i,j]
        #         # pixelBGR_tuple = tuple(map(tuple, pixelBGR))
        #         pixelGrey = bgr2Grey(pixelBGR)
        #         if str(pixelGrey) in pixelDict:
        #             pixelDict[str(pixelGrey)].append(pixelBGR)
        #             continue
        #         elif pixelGrey in indiceLocalMaxima:
        #             pixelDict[str(pixelGrey)] = [pixelBGR]
        # greyBGRDict = dict()
        # for pixel in pixelDict:
        #     value = unique_count_app(pixelDict[pixel])
        #     greyBGRDict[pixel] = value
        # print(greyBGRDict)

        print('test')

if __name__ == "__main__":    main()