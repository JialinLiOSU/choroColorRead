# The goal is to identify US states 
# test achieve this goal in two steps
# 1. edge detection for the original images
# 2. feature matching by SIFT descriptor
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io

from numpy import array, array_equal, allclose
import numpy as np
import cv2
from matplotlib import pyplot as plt
import pickle
import numpy as np
import pickle
import os
import sys
# sys.path.append(r'C:\Users\jiali\Desktop\Map_Identification_Classification\world map generation\getCartoCoordExtent')
# from shapex import *
# from geom.point import *
# from geom.centroid import *
# from shapely.geometry import Polygon
# from shapely.geometry import box
from collections import deque
import random
import pickle

def edgeDetector(img):
    # img = cv2.imread(testImagePath + '\\'+imgName)
    font = cv2.FONT_HERSHEY_COMPLEX
    # legendPath = 'C:\\Users\\jiali\\Desktop\\MapElementDetection\\code\\Legend Analysis\\enhanced legend images'
    height = img.shape[0]
    width = img.shape[1]
    enlargeRatio = 1
    dim = (width*enlargeRatio, height*enlargeRatio)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray,(3,3),0)
    edge = cv2.Canny(gray, 50,200)
    # laplacian = cv2.Laplacian(gray,cv2.CV_8UC1)
    # Taking a matrix of size 5 as the kernel 
    kernel = np.ones((3,3), np.uint8) 

    n= 2
    for i in range(n):
        edge = cv2.dilate(edge, kernel, iterations=1) 
        edge = cv2.erode(edge, kernel, iterations=1) 
    
    cv2.imshow("shapes", edge)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return edge

def removeText(img, ocrResults):
    for ocr in ocrResults:
        bbox = ocr[0]
        xMin = int(bbox[0][0])
        yMin = int(bbox[0][1])
        xMax = int(bbox[2][0])
        yMax = int(bbox[2][1])
        crop_img = img[yMin:yMax, xMin:xMax]
        dominantColor = unique_count_app(crop_img)
        value = dominantColor

        for x in range(xMin,xMax):
            for y in range(yMin,yMax):
                img[y][x] = value
    # cv2.imshow("shapes", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return img
# find the dominant color value for each legend rectangle
def unique_count_app(a):
    colors, count = np.unique(a.reshape(-1,a.shape[-1]), axis=0, return_counts=True)
    return colors[count.argmax()]
def removeTitleLegend(img1,imgDetectResult):
    imgName = imgDetectResult[0]
    finalLegendBox = imgDetectResult[1]
    xMin = int(finalLegendBox.bounds[0])
    xMax = int(finalLegendBox.bounds[2])
    yMin = int(finalLegendBox.bounds[1])
    yMax = int(finalLegendBox.bounds[3])
    xMin = max(xMin, 0) 
    xMax = max(xMax-1, 0)
    yMin = max(yMin, 0)
    yMax = max(yMax-1, 0)
    crop_img = img1[yMin:yMax, xMin:xMax]
    dominantColor = unique_count_app(crop_img)
    value = dominantColor
    for x in range(xMin,xMax):
            for y in range(yMin,yMax):
                img1[y][x] = value
    return img1

def arreq_in_list(myarr, list_arrays):
    return next((True for elem in list_arrays if array_equal(elem, myarr)), False)

def edgeDetector(img):
    # img = cv2.imread(testImagePath + '\\'+imgName)
    font = cv2.FONT_HERSHEY_COMPLEX
    # legendPath = 'C:\\Users\\jiali\\Desktop\\MapElementDetection\\code\\Legend Analysis\\enhanced legend images'
    height = img.shape[0]
    width = img.shape[1]
    enlargeRatio = 1
    dim = (width*enlargeRatio, height*enlargeRatio)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray,(3,3),0)
    edge = cv2.Canny(gray, 50,200)
    # laplacian = cv2.Laplacian(gray,cv2.CV_8UC1)
    # Taking a matrix of size 5 as the kernel 
    kernel = np.ones((3,3), np.uint8) 

    n= 2
    for i in range(n):
        edge = cv2.dilate(edge, kernel, iterations=1) 
        edge = cv2.erode(edge, kernel, iterations=1) 

    # contours, _ = cv2.findContours(
    #     edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # rectList = []  # used to save the rectangles
    # rectIndList = []  # save the min max XY value for extraction

    # for cnt in contours:
    #     approx = cv2.approxPolyDP(cnt, 0.1 * cv2.arcLength(cnt, True), True)
    #     cv2.drawContours(img, [approx], 0, (120, 120, 120), 1)
    #     x = approx.ravel()[0]
    #     y = approx.ravel()[1]

    #     if len(approx) >=3:

    #         test1 = approx[0][0][1]
    #         test2 = approx[2][0][1]
    #         if abs(test1 - test2) > 10:
    #             cv2.putText(img, "Rectangle", (x, y), font, 0.5, (0))
    #             if x >263 *3 and x < 270 * 3:
    #                 print(len(rectList))
    #             rectList.append(approx)
    # rectShapeBoxList = rectListToShapeBoxList(rectList)

    # # find out all rects intersecting with legendBbox and not intersecting with texts
    # legendRectShapeBoxList = []
    # for rectBox in rectShapeBoxList:
    #     isInterText = intersectText(rectBox,legendTextShapeBoxList)
    #     isInterLegend = legendShapeBox.intersects(rectBox)
    #     if isInterLegend and not isInterText:
    #         legendRectShapeBoxList.append(rectBox)

    # legendRectShapeBoxList = removeOverlappedBox(legendRectShapeBoxList) # postprocess to remove overlapped rect boxes
    
    # cv2.imshow("shapes", edge)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return edge

def getBackgroundColor(img1,imgGrey):
    # pick up background color
    (height, width, channel) = img1.shape
    heightList = range(10, height - 10, int((height - 20)/10))
    widthList = range(10, width - 10, int((width - 20)/10))
    samplePoints1 = [[10, w] for w in widthList]
    samplePoints2 = [[height - 10, w] for w in widthList]
    samplePoints3 = [[h, 10] for h in heightList]
    samplePoints4 = [[h, width - 10] for h in heightList]
    samplePoints = samplePoints1 + samplePoints2 + samplePoints3 + samplePoints4

    colorValues = []
    colorCounts = []
    for sp in samplePoints:
        colorValue = imgGrey[sp[0],sp[1]]
        if arreq_in_list(colorValue, colorValues) == False:
            colorValues.append(colorValue)
            colorCounts.append(1)
        else:
            index = colorValues.index(colorValue)
            colorCounts[index] += 1
    colorCounts.sort()
    indexColorMost = colorCounts.index(colorCounts[0])
    bgColorValue1 = colorValues[indexColorMost]
    bgColorValue2 = None
    if len(colorCounts)>=2:
        indexColorMostSec = colorCounts.index(colorCounts[1])
        bgColorValue2 = colorValues[indexColorMostSec]
    return bgColorValue1,bgColorValue2

def getXofPointWithMinY(usBound,yMinImg):
    for coordXY in usBound:
        if coordXY[0][1] == yMinImg:
            return coordXY[0][0]
    return None

def findIndexIntersectingBboxes(i,spShapelyBoxList):
    intersectBboxes = []
    for j in range(len(spShapelyBoxList)):
        if i == j:
            continue
        if spShapelyBoxList[i].intersects(spShapelyBoxList[j]):
            intersectBboxes.append(j)
    return intersectBboxes

def bfs(root,spShapelyBoxList):
    result = []
    
    if root is None:
        return result
    
    q = deque([root])
    while q:
        level = []
        for i in range(len(q)):
            node = q.popleft()
            result.append(node)
            indexIintersectBboxes = findIndexIntersectingBboxes(node,spShapelyBoxList)
            for indxInterBbox in indexIintersectBboxes:
                if (not indxInterBbox in q) and (not indxInterBbox in result):
                    q.append(indxInterBbox)
    return result

def main():
    # read detection results from pickle fil
    computerName = 'jiali'
    # computerName = 'li.7957'
    savePath = 'C:\\Users\\'+ computerName +'\\Desktop\\choroMapThemeAnalysis\\dataCollection\\choroSuperPixel_challenging'

    path = 'C:\\Users\\' + computerName + '\\Desktop\\choroMapThemeAnalysis\\dataCollection\\originalSizeChoroMaps_challenging'
    imageNames = os.listdir(path)  

    # imgName = '1921_462_mask.tif'
    superPixelResultsPairList = []
    imgContinue = False
    imageNames.sort()
    for i,imgName in enumerate(imageNames):
        # if imgName == 'live_birth_correct.png':
        #     continue
        # if imgContinue == False:
        #     continue

        # img1Name = usBoundResult[0]
        # img1Name = "115936742_3317778874932504_6087693444288850821_o.jpg"
        
        if '.json' in imgName:
            continue
        print('index: ' + str(i))
        print('image name: ' + imgName)


        # read images and remove texts on the images
        img = cv2.imread(path + '\\' + imgName) # Image1 to be matched
        img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # loop over the number of segments
        # apply SLIC and extract (approximately) the supplied number
        # of segments
        try:
            image = img_as_float(img1)
            numSegments = 300
            # get segments from the segmentation results
            segments = slic(image, n_segments = numSegments, sigma = 5)
        except:
            print("not working image: " + imgName + '\n')
            continue
        superPixelResultsPairList.append([imgName, segments])
        # edgeSegments = edgeDetectorGrey(segments)
        # show the output of SLIC
        fig = plt.figure("Superpixels -- %d segments" % (numSegments), dpi = 1000)
        ax = plt.gca()
        bounds = mark_boundaries(image, segments)
        ax.imshow(bounds)
        plt.axis("off")
        # show the plots
        
        # plt.show()
        # print('test')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.savefig(savePath + '\\' + 'sp_300_' + imgName)
        imgNameShort = imgName.split('.')[:-1]
        with open('E:\\sp_challenging\\sp_300_challenging_' + '-'.join(imgNameShort) + '.pickle', "wb") as f:
            pickle.dump(superPixelResultsPairList, f)

    segments = superPixelResultsPairList[0][1]
    # print('test')

    # bgColor = 0

    # bgColor,bgColorSec = getBackgroundColor(img1Proc, imgGrey)

        # get the list of pairs of coords of pixels with a specific superpixel segmentid
        # index of the list means superpixel segmentid
    maxSegmentationID = np.amax(segments)
    minSegmentationID = np.amin(segments)

    coordPairsList = [] 
    for id in range(minSegmentationID,maxSegmentationID + 1):
        results = np.where(segments == id)
        coordPairs = np.asarray(results).T.tolist()
        # results = zip(*np.where(segments == id))
        # coord = zip(results)
        coordPairsList.append(coordPairs)

    # identify whether the superpixel is with bg color
    mapRegionSuperPixels = []
    for coordPairs in coordPairsList:
        colorValueList = []
        for coordPair in coordPairs:
            colorValue = imgGrey[coordPair[0],coordPair[1]]
            colorValueList.append(colorValue)
        count0 = colorValueList.count(0)
        count255 = colorValueList.count(255)
        if count0 >= count255:
            maxOccurValue = 0
        else:
            maxOccurValue = 255
        if abs(maxOccurValue - bgColor) > 10:
            mapRegionSuperPixels.append(coordPairs)
    with open(r'd:\superPixelMask_462_4800_buildingCoordPairs.pickle', 'rb') as f:
	    mapRegionSuperPixels = pickle.load(f)
    



    generate bboxes of the super pixels
    spShapelyBoxList = []
    for sp in mapRegionSuperPixels:
        maxCoordSpBbox = np.amax(sp,0)
        minCoordSpBbox = np.amin(sp,0)
        xMaxSpBbox, yMaxSpBbox = maxCoordSpBbox[0],maxCoordSpBbox[1]
        xMinSpBbox, yMinSpBbox = minCoordSpBbox[0], minCoordSpBbox[1]
        spShapelyBoxList.append([xMinSpBbox, yMinSpBbox,xMaxSpBbox, yMaxSpBbox])
    # ax = plt.gca()
    for box in spShapelyBoxList:
        
        xMinSpBbox, yMinSpBbox,xMaxSpBbox, yMaxSpBbox = box[0], box[1], box[2], box[3]
        startPoint = (int(yMinSpBbox),int(xMinSpBbox))
        endPoint = (int(yMaxSpBbox),int(xMaxSpBbox))
        cv2.rectangle(img,startPoint,endPoint,(255, 0, 0),2)
    # cv2.imshow("Bboxes", img)
    cv2.imwrite(r'C:\Users\jiali\Desktop' + '\\' + 'spBbox_1951_462.tif', img)
    print('test')
            
    #         # for each super pixel bbox, for it doesn't intersect with other bboxes
    #         # record the index and remove
    #         noIntersectIndexes = []
    #         for i in range(len(spShapelyBoxList)):
    #             hasIntersecting = 0
    #             for j in range(len(spShapelyBoxList)):
    #                 if i == j:
    #                     continue
    #                 if spShapelyBoxList[i].intersects(spShapelyBoxList[j]):
    #                     hasIntersecting = 1
    #                     break
    #             if hasIntersecting == 0:
    #                 noIntersectIndexes.append(i)

    #         noIntersectIndexes.sort(reverse=True)
    #         if len(noIntersectIndexes)!=0:
    #             for niIndex in noIntersectIndexes:
    #                 del mapRegionSuperPixels[niIndex]
    #                 del spShapelyBoxList[niIndex]

    #         # construct the clusters for continuous US, Alaska
    #         # clusterList = []
    #         # clusterIndexList = []
    #         clusterIndexBboxes = bfs(0,spShapelyBoxList)
    #         initialIndex = 0
    #         while True:
    #             clusterIndexBboxes = bfs(initialIndex,spShapelyBoxList)
    #             if len(clusterIndexBboxes) < len(spShapelyBoxList) / 2:
    #                 remainingIndexes = np.setdiff1d([range(len(spShapelyBoxList))], clusterIndexBboxes)
    #                 remainingIndexes = list(remainingIndexes)

    #                 initialIndex = random.choice(remainingIndexes)
    #             else:
    #                 break

    #         remainingIndexes = np.setdiff1d([range(len(spShapelyBoxList))], clusterIndexBboxes)
    #         remainingIndexes = list(remainingIndexes)

    #         remainingIndexes.sort(reverse = True)

    #         if len(remainingIndexes)!=0:
    #             for reIndex in remainingIndexes:
    #                 del mapRegionSuperPixels[reIndex]
    #                 del spShapelyBoxList[reIndex]
                

    #         # find super-pixels on the corners of US continent
    #         maxXcoordList = []
    #         maxXcoordPairList = []
    #         maxYcoordList = []
    #         maxYcoordPairList = []
    #         minXcoordList = []
    #         minXcoordPairList = []
    #         minYcoordList = []
    #         minYcoordPairList = []
    #         for mapSuperPixel in mapRegionSuperPixels:
    #             maxXcoord = 0
    #             maxYcoord = 0
    #             minXcoord = 999999
    #             minYcoord = 999999
    #             for pairCoord in mapSuperPixel:
    #                 if pairCoord[1] > maxXcoord:
    #                     maxXcoord = pairCoord[1]
    #                     maxXcoordPairList.append(pairCoord)
    #                 if pairCoord[0] > maxYcoord and pairCoord[1] > image.shape[1]/4:
    #                     maxYcoord = pairCoord[0]
    #                     maxYcoordPairList.append(pairCoord)
    #                 if pairCoord[1] < minXcoord:
    #                     minXcoord = pairCoord[1]
    #                     minXcoordPairList.append(pairCoord)
    #                 if pairCoord[0] < minYcoord:
    #                     minYcoord = pairCoord[0]
    #                     minYcoordPairList.append(pairCoord)
    #             maxXcoordList.append(maxXcoord)
    #             maxYcoordList.append(maxYcoord)
    #             minXcoordList.append(minXcoord)
    #             minYcoordList.append(minYcoord)

    #         maxXCoord = max(maxXcoordList)
    #         indexRightSuperPixel = maxXcoordList.index(maxXCoord)
    #         maxXCoordPair = maxXcoordPairList[indexRightSuperPixel]

    #         maxYCoord = max(maxYcoordList)
    #         indexBottomSuperPixel = maxYcoordList.index(maxYCoord)
    #         maxYCoordPair = maxYcoordPairList[indexBottomSuperPixel]

    #         minXCoord = min(minXcoordList)
    #         indexLeftSuperPixel = minXcoordList.index(minXCoord)
    #         minXCoordPair = minXcoordPairList[indexLeftSuperPixel]

    #         minYCoord = min(minYcoordList)
    #         indexTopSuperPixel = minYcoordList.index(minYCoord)
    #         minYCoordPair = minYcoordPairList[indexTopSuperPixel]

    #         deltaImgX =  maxXCoord - minXCoord
    #         deltaImgY =  maxYCoord - minYCoord

    #         xofPointWithMinY = minYCoordPair[1]
    #         if xofPointWithMinY < minXCoord + deltaImgX/5 or xofPointWithMinY > maxXCoord - deltaImgX/5: # conic CRS
    #             deltaGeoX = deltaGeoConicX
    #             deltaGeoY = deltaGeoConicY
    #             x1 = x1c
    #             y2 = y2c
    #             shapefilePath = r'C:\Users\jiali\Desktop\MapElementDetection\code\shpFiles\USA_Contiguous_Albers_Equal_Area_Conic'
    #             fileName = 'USA_Contiguous_Albers_Equal_Area_Conic.shp'
    #             print('conic')
    #         else:
    #             deltaGeoX = deltaGeoWGSX
    #             deltaGeoY = deltaGeoWGSY
    #             x1 = x1w
    #             y2 = y2w
    #             shapefilePath = r'C:\Users\jiali\Desktop\MapElementDetection\code\shpFiles\USA_WGS84'
    #             fileName = 'USA_WGS84.shp'
    #             print('WGS')

    #         MaineSuperPixel = mapRegionSuperPixels[indexRightSuperPixel]
    #         WashingtonSuperPixel = mapRegionSuperPixels[indexTopSuperPixel]
    #         TexasSuperPixel = mapRegionSuperPixels[indexBottomSuperPixel]

    #         for coordPair in MaineSuperPixel+WashingtonSuperPixel+TexasSuperPixel:
    #             imgGrey[coordPair[0],coordPair[1]] = 0

    #         fig = plt.figure()
    #         ax = plt.gca()
    #         # ax.imshow(imgGrey)
    #         plt.show()

    #         # get the three corner image coordinates for Washington, Maine and Texas
    #         # print(maxXCoordPair)
    #         # print(maxYCoordPair)
    #         # print(minYCoordPair)

            
    #         ax.scatter(minXCoord, minYCoord, color='blue', marker='o', alpha=0.8)
    #         ax.scatter(maxXCoord, maxYCoord, color='blue', marker='o', alpha=0.8)

    #         # get the geographic coordinate and image coordinate of a selected state
    #         for state in list(short_state_names.values()):
    #             # if state == 'South Carolina':
    #             #     continue
    #             # xGeoState, yGeoState = getStateExtent(shp, state)
    #             shp = shapex(shapefilePath + '\\' + fileName)

    #             pointList = getPointList(shp,state)
    #             centroidGeo = centroid(pointList)[1]

    #             xCentroidGeo = centroidGeo.x
    #             # print(xCentroidGeo)
    #             yCentroidGeo = centroidGeo.y
    #             # print(yCentroidGeo)

    #             xImgState = minXCoord + (xCentroidGeo - x1) / deltaGeoX * deltaImgX
    #             yImgState = minYCoord + (y2 - yCentroidGeo ) / deltaGeoY * deltaImgY

    #             ax.scatter(xImgState, yImgState, color='red', marker='o', alpha=0.8)
    #             centroidStateCoordList.append((xImgState,yImgState))

    #         ax.imshow(img1)
    #         # plt.axis("off")
    #         # plt.title(img1Name) 
    #         # plt.show()
    #         fig.savefig(savePath + '\\' + img1Name)
    #         print('superpixel')
    #     centroidImgCoordList.append((img1Name,centroidStateCoordList))
    # with open(r'C:\Users\jiali\Desktop\shuaichen\processed_images' + '\\' + 'centroidImgCoordList.pickle', 'wb') as f:
	#     pickle.dump(centroidImgCoordList,f)

    
if __name__ == "__main__":    main()