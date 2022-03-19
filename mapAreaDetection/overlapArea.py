import pickle
import os
import cv2
import matplotlib.pyplot as plt  
import matplotlib.patches as patches

import numpy as np

from skimage.measure import find_contours

from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely.geometry import box
from random import sample


def findDetectResult(detectResults, imageName):
    for dr in detectResults:
        name = dr[0]
        if name == imageName:
            return dr
    return None

def main():
    rootPath = r'C:\Users\jiali\Desktop\choroColorRead\mapAreaDetection\groundTruthMapArea'
    # read detection results from pickle file

    with open(rootPath + '\\' + 'ohioBoundAnnotation.pickle', 'rb') as f:
        ohioBoundContour = pickle.load(f)
    with open(rootPath + '\\' + 'usBoundAnnotation.pickle', 'rb') as f:
        usBoundContour = pickle.load(f)

    ohioBoundContourList = ohioBoundContour.reshape(1051,2).tolist()
    ohioBoundContourList.append(ohioBoundContourList[0])
    ohioBoundPoly = Polygon(ohioBoundContourList)
    usBoundContourList = usBoundContour.reshape(2701,2).tolist()
    usBoundContourList.append(usBoundContourList[0])
    print(usBoundContourList[0])
    print(usBoundContourList[-1])
    usBoundPoly = Polygon(usBoundContourList)
    usBoundPoly = usBoundPoly.buffer(0)

    # read detection results from pickle file
    detectResultsPath = r'D:\OneDrive - The Ohio State University\choroColorRead'
    detectResultFileName = 'detectResultSpatialPattern.pickle'
    with open(detectResultsPath + '\\' + detectResultFileName, 'rb') as f:
        detectResults = pickle.load(f)

    imagePath = r'C:\Users\jiali\Desktop\choroColorRead\generatedMaps\quantiles'
    # imageName = 'ohio_Blues_4_neg1.jpg'
    testImages = os.listdir(imagePath)
    afterTarget = False
    mappedAreaEvaluationList = []
    recallList, precisionList = [],[]
    for imageName in testImages:
        print('imageName: ' + imageName)
        detectResult = findDetectResult(detectResults, imageName)
        region = imageName.split('_')[0]

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

        image = cv2.imread(imagePath + '\\' + imageName)
        imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        height = image.shape[0]
        width = image.shape[1]

        maskPolyList = []
        # get the polygon for mapping area
        for i in range(N):
        #     if i > 0:
        #         continue
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
            padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                vertList = [(vert[0],vert[1]) for vert in verts]
                maskPolygon = Polygon(vertList) # this is the predicted boundary polygon
                if len(maskPolyList)>0:
                    if maskPolygon.area > maskPolyList[0].area:
                        maskPolyList[0] = maskPolygon
                else:
                    maskPolyList.append(maskPolygon)
            
            
        # identify the map region, based on which to calculate overlap area
        if region == 'ohio':
            trueAreaPolygon = ohioBoundPoly
        elif region == 'us':
            trueAreaPolygon = usBoundPoly
        
        maskPolygon = maskPolyList[0]

        overlapArea = trueAreaPolygon.intersection(maskPolygon).area
        # overlapArea1 = maskPolygon.intersection(trueAreaPolygon).area
        recall = overlapArea / trueAreaPolygon.area
        precision = overlapArea / maskPolygon.area
        print(recall)
        print(precision)
        if recall < 0.9 or precision < 0.9:
            print('debug')
        recallList.append(recall)
        precisionList.append(precision)
        mappedAreaEvaluationList.append([imageName,recall,precision, overlapArea, maskPolygon.area, trueAreaPolygon.area])

    print('average recall: '+ str(sum(recallList)/len(recallList)))
    print('average precision: ' + str(sum(precisionList)/len(precisionList)))

    with open(r'C:\Users\jiali\Desktop\choroColorRead\mapAreaDetection' + '\\' + 'mappedAreaEvaluationList.pickle', 'wb') as f:
        pickle.dump(mappedAreaEvaluationList,f)

        print('test')



            

        

if __name__ == "__main__":    main()