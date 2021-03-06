import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt  
import pandas as pd
import seaborn as sns
import pickle
from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely.geometry import box
import shapely.geometry
import time
# from sklearn.datasets import fetch_mldata
# from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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

def pixelEqualToPixel(pixel1, pixel2):
    if abs(pixel1[0] - pixel2[0])<=5 and abs(pixel1[1] - pixel2[1])<=5 and \
        abs(pixel1[2] - pixel2[2])<=5:
        return True
    else:
        return False

def main():
	# read image data
    imagePath = r'D:\OneDrive - The Ohio State University\choroMapThemeAnalysis' + '\\dataCollection'
    testImagePath = imagePath + '\\originalSizeChoroMaps_standard'
    testImages = os.listdir(testImagePath)

    rootPath = r'D:\OneDrive - The Ohio State University\choroMapThemeAnalysis'
    maskRcnnPath = rootPath + '\\maskRCNNResults'
    # read detection results from pickle file
    detectResultsPath = maskRcnnPath + '\\detectPickleResults'
    detectResultsDir = os.listdir(detectResultsPath)
    detectResultsDir.sort()

    testImage = testImages[0]
    unclassifiedImages = ['05_choropleth_65_0.png','0095-choropleth.png','06_africa.png','0_lzCKe_IRiQwe1LsT.png']
    targetImage = '119.Choropleth+Map.jpg'
    afterTarget = False
    for i,testImage in enumerate(testImages): # 3,8,22,214
        # testImage = testImages[3]
        # if i not in [3,8,22,214]:
        #     continue

        # if testImage == targetImage:
        #     afterTarget = True
        #     continue
        # if afterTarget == False:
        #     continue

        if testImage in unclassifiedImages:
            continue

        if testImage[-4:] == 'json':
            continue
        elif testImage[-4:] == 'jpeg':
            detectResultFile = testImage[:-4]
        else:
            detectResultFile = testImage[:-3]
        
        print('testImage: ' + testImage)
        img = cv2.imread(testImagePath + '\\'+testImage)
        cv2.imshow('test', img)

        imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        detectResultFile = detectResultFile + 'pickle'
        with open(detectResultsPath + '\\' + detectResultFile, 'rb') as f:
            detectResult = pickle.load(f)
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

        

        # imgArray = np.asarray(imgList)
        # labelArray = np.ones(imgArray.shape[0],)

        # feat_cols = [ 'pixel'+str(i) for i in range(imgArray.shape[1]) ]
        # df = pd.DataFrame(imgArray,columns=feat_cols)
        # df['y'] = labelArray
        # df['label'] = df['y'].apply(lambda i: str(i))

        # np.random.seed(42)
        # rndperm = np.random.permutation(df.shape[0])

        # N = 10000
        # df_subset = df.loc[rndperm[:N],:].copy()
        # data_subset = df_subset[feat_cols].values

        # time_start = time.time()
        # tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        # tsne_results = tsne.fit_transform(data_subset)
        # print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

        # df_subset['tsne-2d-one'] = tsne_results[:,0]
        # df_subset['tsne-2d-two'] = tsne_results[:,1]
        # plt.figure(figsize=(10,10))
        # sns.scatterplot(
        #     x="tsne-2d-one", y="tsne-2d-two",
        #     hue="y",
        #     # palette=sns.color_palette("hls", 10),
        #     data=df_subset,
        #     legend="full",
        #     alpha=0.3
        # )

        imgListList = (img).tolist()
        imgList = [item for sublist in imgListList for item in sublist]
        # count number of occurence of pixel value in a dictionary
        pixelCounterDict = dict()
        for pixel in imgList:
            pixelTuple = (pixel[0],pixel[1],pixel[2])
            if pixelTuple in pixelCounterDict:
                pixelCounterDict[pixelTuple] += 1
            else:
                pixelCounterDict[pixelTuple] = 0


        bg = max(pixelCounterDict, key=pixelCounterDict.get)

        # bg = most_frequent(imgList)
        # imgListNoBG = [i for i in imgList if i != bg ]
        imgListNoBG = []
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                pixel = img[i,j]
                pixel = pixel/255
                pixel = pixel.tolist()
                point = Point(j,i)
                if pixelEqualToPixel(pixel, bg):
                    continue
                inMappingArea = False
                for box in mapBoxList: # try both box and mask polygon
                    if box.contains(point):
                        inMappingArea = True
                        break
                if inMappingArea:
                    imgListNoBG.append(pixel)

        # imgListList = imgGrey.tolist()
        # imgListList = (img/255).tolist()
        # imgList = [item for sublist in imgListList for item in sublist]

        mappingImgListNoBG = random.sample(imgListNoBG, min(10000, len(imgListNoBG)))

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        # plot path lines of the colors

        bList, gList, rList = [], [], []
        for bgr in mappingImgListNoBG:
            bList.append(bgr[0])
            gList.append(bgr[1])
            rList.append(bgr[2])
            
        ax.scatter(bList, gList, rList, s = 1)
            
        ax.set_xlabel('b')
        ax.set_ylabel('g')
        ax.set_zlabel('r')
        plt.show()

        print('test')
        
        # count number of occurence of pixel value in a dictionary
        # mapPixelCounterDict = dict()
        # for pixel in imgListNoBG:
        #     pixelTuple = (pixel[0],pixel[1],pixel[2])
        #     if pixelTuple in mapPixelCounterDict:
        #         mapPixelCounterDict[pixelTuple] += 1
        #     else:
        #         mapPixelCounterDict[pixelTuple] = 1

        # imgPD = pd.Series(imgList, name = testImage)
        # sns.histplot(data = imgPD, binwidth = 1, kde = True)
        # sortedDict = {k: v for k, v in sorted(mapPixelCounterDict.items(), key=lambda item: item[1], reverse = True)}



        # imgPDNoBG = pd.Series(imgListNoBG, name = testImage)
        # kde = sns.histplot(data = imgPDNoBG, binwidth = 1, kde = True)
        # plt.close()
        
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