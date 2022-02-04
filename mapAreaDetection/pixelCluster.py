from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import os
import cv2
import pickle
from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely.geometry import box
import shapely.geometry
import numpy as np
import matplotlib.pyplot as plt  
import random
random.seed(42)

def pixelEqualToPixel(pixel1, pixel2):
    if abs(pixel1[0] - pixel2[0])<=5 and abs(pixel1[1] - pixel2[1])<=5 and \
        abs(pixel1[2] - pixel2[2])<=5:
        return True
    else:
        return False

def rgb2Grey(dominantColor):
    rgb_weights = [0.2989, 0.5870, 0.1140]
    dominantColorGrey = int(np.dot(dominantColor, rgb_weights))
    return dominantColorGrey

def main():
	# read image data
    imagePath = r'D:\OneDrive - The Ohio State University\choroMapThemeAnalysis' + '\\dataCollection'
    testImagePath = imagePath + '\\originalSizeChoroMaps_standard_classified'
    testImages = os.listdir(testImagePath)

    rootPath = r'D:\OneDrive - The Ohio State University\choroMapThemeAnalysis'
    maskRcnnPath = rootPath + '\\maskRCNNResults'
    # read detection results from pickle file
    detectResultsPath = maskRcnnPath + '\\detectPickleResults'
    # detectResultsDir = os.listdir(detectResultsPath)
    # detectResultsDir.sort()

    afterTarget = False
    for i,testImage in enumerate(testImages): # 3,8,22,214
        if testImage == '1639px-India_population_density_map_en.svg.png':
            afterTarget = True
        
        if afterTarget == False:
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

        # imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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
        if len(imgListNoBG) == 0:
            imgListNoBG = img.tolist()
            mappingImgListNoBG = random.sample(imgListNoBG, min(10000, len(imgListNoBG)))
            imgArrayNoBG = np.asarray(mappingImgListNoBG).reshape(len(mappingImgListNoBG) * len(mappingImgListNoBG[0]),3)
        else:
            mappingImgListNoBG = random.sample(imgListNoBG, min(10000, len(imgListNoBG)))
            imgArrayNoBG = np.asarray(mappingImgListNoBG)

        kmax = 10
        # Create dataset with 3 random cluster centers and 1000 datapoints
        # x, y = make_blobs(n_samples = 1000, centers = 3, n_features=2, shuffle=True, random_state=31)
        x = imgArrayNoBG

        # # function returns WSS score for k values from 1 to kmax
        # sse = []
        # for k in range(1, kmax+1):
        #     kmeans = KMeans(n_clusters = k).fit(x)
        #     centroids = kmeans.cluster_centers_
        #     pred_clusters = kmeans.predict(x)
        #     curr_sse = 0
            
        #     # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
        #     for i in range(len(x)):
        #         curr_center = centroids[pred_clusters[i]]
        #         curr_sse += (x[i, 0] - curr_center[0]) ** 2 + (x[i, 1] - curr_center[1]) ** 2
                
        #         sse.append(curr_sse)

        sil = []
        difSilList = []
        # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
        for k in range(2, kmax+1):
            kmeans = KMeans(n_clusters = k).fit(x)
            labels = kmeans.labels_
            silScore = silhouette_score(x, labels, metric = 'euclidean')
            difSil = 0
            if len(sil) > 0:
                difSil = silScore - sil[-1]
            sil.append(silScore)
            difSilList.append(difSil)
        print(sil)

        # find the best index of difSil
        index = 0
        

        maxValue = max(sil)
        maxIndex = sil.index(maxValue)
        maxDif = max(difSilList)

        while maxIndex >= 3:
            maxValue = sil[maxIndex]
            i = maxIndex - 1
            if difSilList[maxIndex] < maxDif / 10:
                maxIndex = maxIndex - 1
            else:
                break

        # for i in range(1, len(difSilList)):
        #     difSilLeft = difSilList[i - 1]
        #     difSilRight = difSilList[i]
        #     if difSilRight < difSilLeft / 5:
        #         index = i - 1
        #         break
        #     else:
        #         index = i

        numClusters = maxIndex + 2
        kmeans = KMeans(n_clusters = numClusters).fit(x)

        fig = plt.figure()
        ax = fig.add_subplot()
        # plot path lines of the colors

        bList, gList, rList = [], [], []
        colorList = []
        xList = []
        yList = []
        for i,bgr in enumerate(kmeans.cluster_centers_):

            bList.append(bgr[0])
            gList.append(bgr[1])
            rList.append(bgr[2])
            
            b = bgr[0]
            if b > 1:
                b = 1
            elif b < 0:
                b = 0
                
            g = bgr[1]
            if g > 1:
                g = 1
            elif g < 0:
                g = 0

            r = bgr[2]
            if r > 1:
                r = 1
            elif r < 0:
                r = 0

            colorList.append((r, g, b))

            print("r: {r:.2f} g: {g:.2f} b: {b:.2f}!".format(r = r * 255, g = g * 255, b = b * 255))

            xList.append(i+1)
            yList.append(1)
            
        colorList.sort(key=lambda x: rgb2Grey((x[0],x[1],x[2])))
        ax.scatter(xList, yList, s = 20,c = colorList)
            
        # ax.set_xlabel('b')
        # ax.set_ylabel('g')
        # ax.set_zlabel('r')

        # plt.show()
        plt.savefig('colors_'+testImage)

        print('test')

if __name__ == "__main__":    main()