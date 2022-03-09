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
    dominantLegendGreyColorList = []
    for coordPairs in coordPairsList:
        colorValueList = []
        for coordPair in coordPairs:
            colorValue = imgGreyLegendClean[coordPair[0],coordPair[1]]
            colorValueList.append(colorValue)
        
        maxOccurValue = max(colorValueList,key=colorValueList.count)
        if abs(maxOccurValue - dominantGreyLegend) > 5:
            mapRegionSuperPixels.append(coordPairs)
            dominantLegendGreyColorList.append(maxOccurValue)

    # generate bboxes of the super pixels
    spLeggendShapelyBoxList = []
    dominantLegendColorList = []
    for sp in mapRegionSuperPixels:
        maxCoordSpBbox = np.amax(sp,0)
        minCoordSpBbox = np.amin(sp,0)
        yMaxSpBbox, xMaxSpBbox = maxCoordSpBbox[0],maxCoordSpBbox[1]
        yMinSpBbox, xMinSpBbox = minCoordSpBbox[0], minCoordSpBbox[1]
        croppedImg = imgRGBLegend[ yMinSpBbox:yMaxSpBbox,xMinSpBbox:xMaxSpBbox]
        dominantColor = unique_count_app(croppedImg)
        dominantColorGrey = rgb2Grey(dominantColor)
        dominantLegendColorList.append(dominantColor.tolist())
        spLeggendShapelyBoxList.append(box(xMinSpBbox, yMinSpBbox, xMaxSpBbox, yMaxSpBbox))
    # till now, spLeggendShapelyBoxList and dominantLegendGreyColorList are used 
    return spLeggendShapelyBoxList,dominantLegendGreyColorList, dominantLegendColorList, (xMin,yMin)

def combineTwoColorList(superPixelValueList, dominantLegendColorList):
	colorList = superPixelValueList
	for legendColor in dominantLegendColorList:
		if legendColor in colorList:
			continue
		closeToColorList = False
		for color in colorList:
			distance = [abs(legendColor[0] - color[0]),abs(legendColor[1] - color[1]),abs(legendColor[2] - color[2])]
			isDiff = [d > 5 for d in distance]
			numDiffChannel = isDiff.count(True)
			if numDiffChannel < 2:
				closeToColorList = True
				break

		if closeToColorList == False:
			colorList.append(legendColor)
	return colorList

def main():

	# read image data
	imagePath = r'C:\Users\jiali\Desktop\choroColorRead\generatedMaps\quantiles'
	testImages = os.listdir(imagePath)
	savePath = r'C:\Users\jiali\Desktop\choroColorRead\mapAreaDetection'

    # read detection results from pickle file
	detectResultsPath = r'D:\OneDrive - The Ohio State University\choroColorRead'
	detectResultFileName = 'detectResultSpatialPattern.pickle'
	with open(detectResultsPath + '\\' + detectResultFileName, 'rb') as f:
		detectResults = pickle.load(f)

	# super pixel results
	superPixelResultsPath = detectResultsPath
	with open(superPixelResultsPath + '\\' + 'sp_300_spatialPattern_images_quantiles.pickle', 'rb') as f:
		superPixelResults = pickle.load(f)
	afterTarget = False
	for i, imageName in enumerate(testImages):
		
		detectResult = detectResults[i]
		print("imageName: "+imageName)

		if imageName == 'us_Blues_4_nonAuto1.jpg':
			afterTarget = True

		if afterTarget == False:
			continue

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

		
		# super pixels
		spr = superPixelResults[i]
		segments = spr[1]

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
			for coordPair in coordPairs:
				colorValue = imageGray[coordPair[0],coordPair[1]]
				colorValueList.append(colorValue)
					
			maxOccurValue = max(colorValueList,key=colorValueList.count)
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
		for sp in superPixelList:
			if (sp.isInMapPolygon or sp.intersectsMapPolygon) and abs(sp.maxOccurValue - bgColorValue) > 5:
				mapSuperpixelList.append(sp)
		mapMaxOccurValueList = [mapSP.maxOccurValue for mapSP in mapSuperpixelList]
		
		rgb_weights = [0.2989, 0.5870, 0.1140]
		superPixelValueList = []
		superPixelGreyValueList = []
		for mapSP in mapSuperpixelList:
			shapelybox = mapSP.bbox
			bounds = shapelybox.bounds
			xMin = int(bounds[0])
			yMin = int(bounds[1])
			xMax = int(bounds[2])
			yMax = int(bounds[3])
			croppedImg = image[xMin:xMax,yMin:yMax]
			dominantColor = unique_count_app(croppedImg)
			dominantColorGrey = int(np.dot(dominantColor, rgb_weights))

			# need to identify whether new value is close to value in list
			diffList = [abs(dominantColorGrey - pv)>10 for pv in superPixelGreyValueList]
			if  abs(dominantColorGrey - bgColor) > 5:
				if all(diffList):
					superPixelGreyValueList.append(dominantColorGrey)
					superPixelValueList.append(dominantColor.tolist())
				else:
					# check whether the new RGB balue is different from the exisiting value in the list
					sameGreyIndex = [i for i, x in enumerate(diffList) if x == False]
					# compare rgb value with all existing grey values' corresponding rgb value
					rgbDiff = True
					for ind in sameGreyIndex:
						existSuperPixelValue = superPixelValueList[ind]
						espR, espG, espB = existSuperPixelValue[0],existSuperPixelValue[1],existSuperPixelValue[2]
						dcR, dcG, dcB = dominantColor[0],dominantColor[1],dominantColor[2]
						if abs(espR - dcR) <= 15 and abs(espG - dcG) <= 15 and abs(espG - dcG) <= 15:
							rgbDiff = False
							break
					if rgbDiff:
						superPixelGreyValueList.append(dominantColorGrey)
						superPixelValueList.append(dominantColor.tolist())
					
		
		# uniqueDominantColors = unique(dominantColorList)
		print(superPixelValueList)
		# rect = patches.Rectangle((50, 100), 40, 30, linewidth=1, edgecolor='r', facecolor='none')

		# visualize results
		# for i,color in enumerate(superPixelValueList):#color rgb
		# 	startPoint = (20 * i, 0)
		# 	endPoint = (20 * i + 10, 10)
		# 	cv2.rectangle(imgtemp,startPoint,endPoint,(color[2],color[1],color[0]),5) #  BGR

		# cv2.imshow('test', imgtemp)
		# get legend bboxes, text bboxes, and rectangles
		# legendShapelyBbox,legendTextShapelyBoxList,legendTextBboxes = getLegendBboxImage(imageName,legendResults)
		# # legendShapelyBbox,legendTextShapelyBoxList,legendTextBboxes = legendResults[1],legendResults[2],legendResults[3]
		# # if legendShapelyBbox == None:
		# # 	print('no legend detected!')
		# # 	continue

		# spLegendShapelyBoxList,dominantLegendGreyColorList,dominantLegendColorList, (xMinLeg,yMinLeg) = getSPLegendNonBackground(legendShapelyBbox,image,legendTextShapelyBoxList)
		# spLegendResults = (spLegendShapelyBoxList, dominantLegendGreyColorList,(xMinLeg,yMinLeg))
		# print(dominantLegendColorList)

		# colorsMappingArea = combineTwoColorList(superPixelValueList, dominantLegendColorList)
		# print(colorsMappingArea)

		# # visualize results
		# for i,color in enumerate(colorsMappingArea):#color rgb
		# 	startPoint = (20 * i, 0)
		# 	endPoint = (20 * i + 10, 10)
		# 	cv2.rectangle(imgtemp,startPoint,endPoint,(color[2],color[1],color[0]),5) #  BGR
		# cv2.imshow('test', imgtemp)


		# Create a Rectangle patch
		

	print('test')

if __name__ == "__main__":    main()

