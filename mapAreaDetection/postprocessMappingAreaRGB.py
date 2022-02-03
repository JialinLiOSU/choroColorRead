# combine detected mapping area by Mask RCNN and superpixel
import pickle
import os
import numpy as np
import json
from PIL import Image,ImageDraw
import sys
# import tensorflow as tf
# import keras

import cv2
import matplotlib.pyplot as plt  
import matplotlib.patches as patches

from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
from skimage.measure import find_contours

from shapely.geometry import Polygon
from shapely.geometry import box

class superPixel:
	def __init__(self, bbox, isInMapBbox, isInMapPolygon, intersectsMapPolygon, maxOccurValue):
		self.bbox = bbox
		self.isInMapBbox = isInMapBbox
		self.isInMapPolygon = isInMapPolygon
		self.intersectsMapPolygon = intersectsMapPolygon
		self.maxOccurValue = maxOccurValue

# find the dominant color value for each legend rectangle
def unique_count_app(a):
    colors, count = np.unique(a.reshape(-1,a.shape[-1]), axis=0, return_counts=True)
    return colors[count.argmax()]

def unique(list1):
 
	# initialize a null list
	unique_list = []

	# traverse for all elements
	for x in list1:
		# check if exists in unique_list or not
		if x not in unique_list:
			unique_list.append(x)
	return unique_list

# get the most likely legend bbox for imgName from detectResults
def getLegendBboxImage(imgName,legendResults):
    # legendResults.append((imgName,finalLegendBox,legendRectShapeBoxList,legendTextShapelyBoxList,legendTextBboxes))
    legendShapeBbox = None
    legendTextShapelyBoxList = None
    legendTextBboxes = None
    for result in legendResults: # result[0]: image name,
        if result[0] == imgName:
            legendShapeBbox = result[1]
            legendTextShapelyBoxList = result[2]
            legendTextBboxes = result[3]
            break
    
    return legendShapeBbox,legendTextShapelyBoxList,legendTextBboxes

def rgb2Grey(dominantColor):
    rgb_weights = [0.2989, 0.5870, 0.1140]
    dominantColorGrey = int(np.dot(dominantColor, rgb_weights))
    return dominantColorGrey

def removeText(imgRGBLegend,legendTextShapelyBoxList,dominantColorLegend,xMinLeg,yMinLeg):
    for legTextBox in legendTextShapelyBoxList:
        xMin = int(legTextBox.bounds[0]) - xMinLeg
        yMin = int(legTextBox.bounds[1]) - yMinLeg
        xMax = int(legTextBox.bounds[2]) - xMinLeg
        yMax = int(legTextBox.bounds[3]) - yMinLeg

        for x in range(xMin,xMax):
            for y in range(yMin,yMax):
                imgRGBLegend[y][x] = dominantColorLegend
    # cv2.imshow("shapes", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return imgRGBLegend

def getSPLegendNonBackground(legendShapelyBbox,imgRGB,legendTextShapelyBoxList):
    # get dominant color in legend area
    xMin = int(legendShapelyBbox.bounds[0])
    yMin = int(legendShapelyBbox.bounds[1])
    xMax = int(legendShapelyBbox.bounds[2])
    yMax = int(legendShapelyBbox.bounds[3])
    imgRGBLegend = imgRGB[yMin:yMax, xMin:xMax]
    dominantColorLegend = unique_count_app(imgRGBLegend)
    dominantGreyLegend = rgb2Grey(dominantColorLegend)
        
    # remove texts in legend   
    if len(legendTextShapelyBoxList) != 0:
        imgRGBLegendClean = removeText(imgRGBLegend,legendTextShapelyBoxList,dominantColorLegend,xMin,yMin)
    else:
        imgRGBLegendClean = imgRGBLegend
    imgGreyLegendClean = cv2.cvtColor(imgRGBLegendClean, cv2.COLOR_RGB2GRAY)

    # cv2.imshow('test', imgGreyLegendClean)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    # loop over the number of segments
    # apply SLIC and extract (approximately) the supplied number
    # of segments
    imgRGBLegendCleanFloat = img_as_float(imgRGBLegendClean)
    numSegments = 50
    # get segments from the segmentation results
    segments = slic(imgRGBLegendCleanFloat, n_segments = numSegments, sigma = 5)
    # edgeSegments = edgeDetectorGrey(segments)
    # show the output of SLIC
    fig = plt.figure("Superpixels -- %d segments" % (numSegments))
    ax = fig.add_subplot(1, 1, 1)
    bounds = mark_boundaries(imgRGBLegendCleanFloat, segments)
    ax.imshow(bounds)
    # plt.show()

    # get the list of pairs of coords of pixels with a specific superpixel segmentid
    # index of the list means superpixel segmentid
    maxSegmentationID = np.amax(segments)
    minSegmentationID = np.amin(segments)

    coordPairsList = [] 
    for id in range(minSegmentationID,maxSegmentationID + 1):
        results = np.where(segments == id)
        coordPairs = np.asarray(results).T.tolist()
        coordPairsList.append(coordPairs)

    # identify whether the superpixel is with bg color
    mapRegionSuperPixels = []
    dominantLegendColorList = []


	# identify whether the superpixel is with bg color
    mapRegionSuperPixels = []
    for i, coordPairs in enumerate(coordPairsList):
        colorValueRGBList = dict()
        for coordPair in coordPairs:
            colorValueRGB = imgRGBLegendClean[coordPair[0],coordPair[1]]
            colorValueRGB = tuple(colorValueRGB) # convert to tuple to be the key of dict
            if colorValueRGB in colorValueRGBList:
                colorValueRGBList[colorValueRGB] +=1
            else:
                # if there is no same color in colorValueRGBList, compare with the colors in the list
                closeToColorList, closeColor = colorCloseToColorList(colorValueRGB,colorValueRGBList)
                if closeToColorList == False:
                    colorValueRGBList[colorValueRGB] = 1
                else:
                    colorValueRGBList[closeColor] += 1
        maxOccurValue = max(colorValueRGBList,key=colorValueRGBList.get)
        if not areTwoColorClose(maxOccurValue, dominantColorLegend) and not colorCloseToColorList(maxOccurValue, dominantLegendColorList):
            mapRegionSuperPixels.append(coordPairs)
            dominantLegendColorList.append(maxOccurValue)

    return dominantLegendColorList, (xMin,yMin)

def combineTwoColorList(superPixelValueList, dominantLegendColorList):
	colorList = superPixelValueList
	for legendColor in dominantLegendColorList:
		if legendColor in colorList:
			continue
		closeToColorList = False
		for color in colorList:
			d1, d2, d3 = abs(int(legendColor[0]) - int(color[0])),abs(int(legendColor[1]) - int(color[1])),abs(int(legendColor[2]) - int(color[2]))
			distance = [d1, d2, d3]
			isClose = [d < 10 for d in distance]
			numCloseChannel = isClose.count(True)
			if numCloseChannel == 3 or d1+d2+d3 < 30:
				closeToColorList = True
			break

		if closeToColorList == False:
			colorList.append(legendColor)
	return colorList

def colorCloseToColorList(colorValueRGB,colorValueRGBList):
	closeToColorList = False
	closeColor = None
	for color in colorValueRGBList:
		d1, d2, d3 = abs(int(colorValueRGB[0]) - int(color[0])),abs(int(colorValueRGB[1]) - int(color[1])),abs(int(colorValueRGB[2]) - int(color[2]))
		distance = [d1, d2, d3]
		isClose = [d < 10 for d in distance]
		numCloseChannel = isClose.count(True)
		if numCloseChannel == 3 or d1+d2+d3 < 30:
			closeToColorList = True
			closeColor = color
			break
	return closeToColorList, closeColor

def areTwoColorClose(color1,color2):
	close = False
	d1,d2,d3 = abs(int(color1[0]) - int(color2[0])),abs(int(color1[1]) - int(color2[1])),abs(int(color1[2]) - abs(color2[2]))
	distance = [d1,d2,d3]
	isClose = [d <= 10 for d in distance]
	numCloseChannel = isClose.count(True)
	if numCloseChannel == 3 or  d1+d2+d3 < 30:
		close = True
	return close


def main():
	rootPath = r'D:\OneDrive - The Ohio State University\choroMapThemeAnalysis'
	maskRcnnPath = rootPath + '\\maskRCNNResults'
	# read detection results from pickle file
	imagePath = rootPath + '\\dataCollection\originalSizeChoroMaps_standard'
	detectResultsPath = maskRcnnPath + '\\detectPickleResults'
	detectResultsDir = os.listdir(detectResultsPath)
	detectResultsDir.sort()

	priorResultsPath = r'C:\Users\jiali\Desktop\choroMapThemeAnalysis' + '\\' + 'intermediateResults'
	# read legend post processing results to get colors from legend area
	with open(priorResultsPath + '\\legendResultsEnlargedStandard_11_11_2021.pickle', 'rb') as f:
		legendResults = pickle.load(f)

	unclassifiedImages = ['05_choropleth_65_0.png','0095-choropleth.png','06_africa.png','0_lzCKe_IRiQwe1LsT.png']

	for detectResultFile in detectResultsDir:
		with open(detectResultsPath + '\\' + detectResultFile, 'rb') as f:
			detectResult = pickle.load(f)

		imageName = detectResult[0]
		if imageName in unclassifiedImages:
			continue
		print("imageName: "+imageName)
		image = io.imread(imagePath + '\\' + imageName)
		if image.shape[2]==4:
			image = image[:,:,:3]
		height = image.shape[0]
		width = image.shape[1]

		imgtemp = cv2.imread(imagePath + '\\' + imageName)
		imageGray = cv2.cvtColor(imgtemp, cv2.COLOR_BGR2GRAY)
		property = detectResult[1]
		boxes = property['rois']
		masks = property['masks']
		class_ids = property['class_ids']

		# extract mask for mapping area
		N = boxes.shape[0]
		if not N:
			print("\n*** No instances to display *** \n")
		else:
			assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

		masked_image = image.astype(np.uint32).copy()
		maskPolyList = []
		for i in range(N):
			if i >0:
				continue
			if class_ids[i] != 3:
				continue
			# Bounding box
			if not np.any(boxes[i]):
				# Skip this instance. Has no bbox. Likely lost in image cropping.
				continue
			y1, x1, y2, x2 = boxes[i]
			bboxMap = box(x1, y1, x2, y2)
			# Mask
			mask = masks[:, :, i]

			# Mask Polygon
			# Pad to ensure proper polygons for masks that touch image edges.
			padded_mask = np.zeros(
				(mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
			padded_mask[1:-1, 1:-1] = mask
			contours = find_contours(padded_mask, 0.5)
			for verts in contours:
				# Subtract the padding and flip (y, x) to (x, y)
				verts = np.fliplr(verts) - 1
				vertList = [(vert[0],vert[1]) for vert in verts]
				maskPolygon = Polygon(vertList)
				maskPolyList.append(maskPolygon)

		# super pixel results
		superPixelResultsPath = r'D:\OneDrive - The Ohio State University\choroMapThemeAnalysis\superPixelResults'

		with open(superPixelResultsPath + '\\' + 'superPixelResultsStandard.pickle', 'rb') as f:
			superPixelResults = pickle.load(f)

		for spr in superPixelResults:
			if spr[0] == imageName:
				segments = spr[1]
				break
		# segments = superPixelResults[107][1]

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
		maxOccurValueList = []
		for i, coordPairs in enumerate(coordPairsList):
			colorValueList = []
			colorValueRGBList = dict()
			for coordPair in coordPairs:
				colorValue = imageGray[coordPair[0],coordPair[1]]
				colorValueList.append(colorValue)
				colorValueRGB = image[coordPair[0],coordPair[1]]
				colorValueRGB = tuple(colorValueRGB) # convert to tuple to be the key of dict
				if colorValueRGB in colorValueRGBList:
					colorValueRGBList[colorValueRGB] +=1
				else:
					# if there is no same color in colorValueRGBList, compare with the colors in the list
					closeToColorList, closeColor = colorCloseToColorList(colorValueRGB,colorValueRGBList)
					if closeToColorList == False:
						colorValueRGBList[colorValueRGB] = 1
					else:
						colorValueRGBList[closeColor] += 1
					
			maxOccurValue = max(colorValueRGBList,key=colorValueRGBList.get)
			maxOccurValueList.append(maxOccurValue)
			# print('index: ' + str(i))
		# with open('maxOccurValueList.pickle', 'rb') as f:
		#     maxOccurValueList = pickle.load(f)
		
		# generate bboxes of the super pixels
		spShapelyBoxList = []
		for sp in coordPairsList:
			maxCoordSpBbox = np.amax(sp,0)
			minCoordSpBbox = np.amin(sp,0)
			xMaxSpBbox, yMaxSpBbox = maxCoordSpBbox[0],maxCoordSpBbox[1]
			xMinSpBbox, yMinSpBbox = minCoordSpBbox[0], minCoordSpBbox[1]
			spShapelyBoxList.append(box(xMinSpBbox, yMinSpBbox,xMaxSpBbox, yMaxSpBbox))

		# for each super pixel, save its bbox, isInMapBbox, isInMapPolygon, maxOccurValue in object
		superPixelList=[]
		for i, spbox in enumerate(spShapelyBoxList):
			isInMapBbox = bboxMap.contains(spbox)
			# if isInMapBbox:
			# 	print(i)
			isInMapPolygon, intersectsMapPolygon = False, False
			# maskPolyList.sort(key = lambda x:x.area)
			# maskPolygon = maskPolyList[-1]
			for maskPolygon in maskPolyList:
				if maskPolygon.contains(spbox):
					isInMapPolygon = True
					# break
				elif maskPolygon.intersects(spbox):
					intersectsMapPolygon = True
					# break

			maxOccurValue = maxOccurValueList[i]
			# maxOccurValue = 0
			spObject = superPixel(spbox, isInMapBbox, isInMapPolygon,intersectsMapPolygon, maxOccurValue)
			superPixelList.append(spObject)
		
		# calculate background color
		bgSuperpixelList = [] # super pixels in map bbox but not in map polygon
		for sp in superPixelList:
			if sp.isInMapBbox and not sp.isInMapPolygon:
				bgSuperpixelList.append(sp)

		bgMaxOccurValueList = []
		for bgsp in bgSuperpixelList:
			bgMaxOccurValueList.append(bgsp.maxOccurValue)
		bgColorValue = max(bgMaxOccurValueList,key=bgMaxOccurValueList.count)
		bgColor= bgColorValue

		mapSuperpixelList = [] # super pixels in map polygon 
		mapMaxOccurValueList = []
		for sp in superPixelList:
			if sp.maxOccurValue == (246, 246, 246):
				print("debug")
			intersects = (sp.isInMapPolygon or sp.intersectsMapPolygon)
			closeToBG = areTwoColorClose(sp.maxOccurValue, bgColorValue)
			closeToSaved = colorCloseToColorList(sp.maxOccurValue,mapMaxOccurValueList)[0]
			if intersects and not closeToBG and not closeToSaved:
				mapSuperpixelList.append(sp)
				mapMaxOccurValueList.append(sp.maxOccurValue)
		# mapMaxOccurValueList = [mapSP.maxOccurValue for mapSP in mapSuperpixelList]
		
		rgb_weights = [0.2989, 0.5870, 0.1140]
		superPixelValueList = []
		superPixelGreyValueList = []
		for mapSP in mapSuperpixelList:
			shapelybox = mapSP.bbox
			bounds = shapelybox.bounds
			dominantColor = mapSP.maxOccurValue
			dominantColorGrey = int(np.dot(dominantColor, rgb_weights))

			# need to identify whether new value is close to value in list
			closeToValueInList = colorCloseToColorList(dominantColor,superPixelValueList)[0]
			diffList = [abs(dominantColorGrey - pv)>10 for pv in superPixelGreyValueList]
			if not closeToValueInList :
				# if bgColorSec != None and abs(dominantColorGrey - bgColorSec) <= 10:
				#     continue
				superPixelValueList.append(dominantColor)
			# superPixelValueList.append(dominantColor.tolist())
		
		# uniqueDominantColors = unique(dominantColorList)
		print(superPixelValueList)
		# rect = patches.Rectangle((50, 100), 40, 30, linewidth=1, edgecolor='r', facecolor='none')

		# visualize results
		for i,color in enumerate(superPixelValueList):#color rgb
			startPoint = (10+20 * i, 0)
			endPoint = (20 * i + 20, 10)
			color = list(color) # tuple to list
			BGR_B, BGR_G, BGR_R = int(color[2]), int(color[1]), int(color[0])
			
			cv2.rectangle(imgtemp,startPoint,endPoint,(BGR_B,BGR_G,BGR_R),5) #  BGR

		cv2.imshow('test', imgtemp)
		# get legend bboxes, text bboxes, and rectangles
		legendShapelyBbox,legendTextShapelyBoxList,legendTextBboxes = getLegendBboxImage(imageName,legendResults)
		# legendShapelyBbox,legendTextShapelyBoxList,legendTextBboxes = legendResults[1],legendResults[2],legendResults[3]
		# if legendShapelyBbox == None:
		# 	print('no legend detected!')
		# 	continue

		dominantLegendColorList, (xMinLeg,yMinLeg) = getSPLegendNonBackground(legendShapelyBbox,image,legendTextShapelyBoxList)
		# spLegendResults = (spLegendShapelyBoxList, dominantLegendGreyColorList,(xMinLeg,yMinLeg))
		print(dominantLegendColorList)

		colorsMappingArea = combineTwoColorList(superPixelValueList, dominantLegendColorList)
		print(colorsMappingArea)

		# visualize results
		for i,color in enumerate(colorsMappingArea):#color rgb
			startPoint = (10+20 * i, 0)
			endPoint = (20 * i + 20, 10)
			color = list(color) # tuple to list
			BGR_B, BGR_G, BGR_R = int(color[2]), int(color[1]), int(color[0])
			cv2.rectangle(imgtemp,startPoint,endPoint,(BGR_B, BGR_G, BGR_R),5) #  BGR
		cv2.imshow('test', imgtemp)
		print('test')


		# Create a Rectangle patch
		

	print('test')

if __name__ == "__main__":    main()

