import pickle
import os
import numpy as np
import json
from PIL import Image,ImageDraw
import sys

import cv2
import matplotlib.pyplot as plt  
import matplotlib.patches as patches

from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
from skimage.measure import find_contours

from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely.geometry import box
from random import sample

import math
import random
from sklearn.neighbors import KDTree
import alphashape
import statistics
from scipy.stats import ranksums
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def bgr2rgb(value):
    return value[2],value[0],value[1]
def valueEqualColor(value,color):
    colorR, colorG, colorB = color[0], color[1], color[2]
    valueR, valueG, valueB = value[0], value[1], value[2]
    if abs(colorR - valueR) > 10 or abs(colorB - valueB) > 10 or abs(colorG - valueG) > 10:
        return False
    else:
        return True
def rgb2Grey(dominantColor):
    rgb_weights = [0.2989, 0.5870, 0.1140]
    dominantColorGrey = int(np.dot(dominantColor, rgb_weights))
    return dominantColorGrey
def distance(coord1, coord2):
    x1, y1 = coord1[0], coord1[1]
    x2, y2 = coord2[0], coord2[1]
    dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    return dist

def silKmeans(kmax, x):
    sil = []
    difSilList = []
    # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
    for k in range(3, kmax):
    #     print('K: '+str(k))
        kmeans = KMeans(n_clusters = k,random_state=0).fit(x)
        labels = kmeans.labels_
        silScore = silhouette_score(x, labels, metric = 'euclidean')
        difSil = 0
        if len(sil) > 0:
            difSil = silScore - sil[-1]
        sil.append(silScore)
        difSilList.append(difSil)

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
    numClusters = maxIndex + 2
    kmeansResults = KMeans(n_clusters = numClusters).fit(x)
    return kmeansResults

def moransi(z, w):
    """
    Input
      z: list of values
      w: weight matrix
    Output
      I: Moran's I measure
    """
    n = len(z)
    mean = float(sum(z))/n
    S0 = 0
    d = 0
    var = 0
    for i in range(n):
        var += (z[i]-mean)**2
        for j in range(i):
            if w[i][j]:
                S0 += w[i][j]
                d += w[i][j] * (z[i]-mean) * (z[j]-mean)
    I = n*d/S0/var
    return I

spColorExtraction = {
'ohio_Blues_4_neg.jpg':[[[191, 214, 230], [107, 174, 216], [33, 114, 180], [239, 243, 255]],],
'ohio_Blues_4_neg1.jpg':[[[239, 243, 255], [33, 114, 180], [107, 174, 216], [191, 214, 230]],],
'ohio_Blues_4_nonAuto.jpg':[[[191, 214, 230], [33, 114, 180], [239, 243, 255], [107, 174, 216]],],
'ohio_Blues_4_nonAuto1.jpg':[[[107, 174, 216], [191, 214, 230], [33, 114, 180], [239, 243, 255]],],
'ohio_Blues_4_pos_large.jpg':[[[191, 214, 230], [33, 114, 180], [239, 243, 255], [107, 174, 216]],],
'ohio_Blues_4_pos_large1.jpg':[[[191, 214, 230], [33, 114, 180], [239, 243, 255], [107, 174, 216]],],
'ohio_Blues_4_pos_small.jpg':[[[191, 214, 230], [239, 243, 255], [107, 174, 216], [33, 114, 180]],],
'ohio_Blues_4_pos_small1.jpg':[[[191, 214, 230], [107, 174, 216], [239, 243, 255], [33, 114, 180]],],
'ohio_Blues_6_neg.jpg':[[[157, 202, 225], [50, 130, 189], [107, 174, 216], [239, 243, 255], [198, 219, 240], [8, 82, 157]],],
'ohio_Blues_6_neg1.jpg':[[[157, 202, 225], [239, 243, 255], [107, 174, 216], [198, 219, 240], [8, 82, 157], [50, 130, 189]],],
'ohio_Blues_6_nonAuto.jpg':[[[157, 202, 225], [50, 130, 189], [107, 174, 216], [239, 243, 255], [198, 219, 240], [8, 82, 157]],],
'ohio_Blues_6_nonAuto1.jpg':[[[157, 202, 225], [50, 130, 189], [107, 174, 216], [239, 243, 255], [198, 219, 240], [8, 82, 157]],],
'ohio_Blues_6_pos_large.jpg':[[[157, 202, 225], [50, 130, 189], [107, 174, 216], [239, 243, 255], [198, 219, 240], [8, 82, 157]],],
'ohio_Blues_6_pos_large1.jpg':[[[157, 202, 225], [239, 243, 255], [8, 82, 157], [50, 130, 189], [107, 174, 216], [198, 219, 240]],],
'ohio_Blues_6_pos_small.jpg':[[[157, 202, 225], [239, 243, 255], [107, 174, 216], [198, 219, 240], [8, 82, 157], [50, 130, 189]],],
'ohio_Blues_6_pos_small1.jpg':[[[157, 202, 225], [50, 130, 189], [239, 243, 255], [198, 219, 240], [8, 82, 157], [107, 174, 216]],],
'ohio_Blues_8_neg.jpg':[[[157, 202, 225], [66, 146, 197], [33, 114, 180], [198, 219, 240], [247, 250, 255], [107, 174, 216], [222, 234, 246], [8, 69, 149]],],
'ohio_Blues_8_neg1.jpg':[[[247, 250, 255], [8, 69, 149], [66, 146, 197], [33, 114, 180], [222, 234, 246], [107, 174, 216], [198, 219, 240], [157, 202, 225]],],
'ohio_Blues_8_nonAuto.jpg':[[[157, 202, 225], [8, 69, 149], [247, 250, 255], [198, 219, 240], [222, 234, 246], [107, 174, 216], [33, 114, 180], [66, 146, 197]],],
'ohio_Blues_8_nonAuto1.jpg':[[[107, 174, 216], [198, 219, 240], [66, 146, 197], [157, 202, 225], [8, 69, 149], [247, 250, 255], [222, 234, 246], [33, 114, 180]],],
'ohio_Blues_8_pos_large.jpg':[[[157, 202, 225], [33, 114, 180], [247, 250, 255], [198, 219, 240], [8, 69, 149], [222, 234, 246], [66, 146, 197], [107, 174, 216]],],
'ohio_Blues_8_pos_large1.jpg':[[[157, 202, 225], [247, 250, 255], [33, 114, 180], [66, 146, 197], [198, 219, 240], [222, 234, 246], [8, 69, 149], [107, 174, 216]],],
'ohio_Blues_8_pos_small.jpg':[[[157, 202, 225], [222, 234, 246], [247, 250, 255], [66, 146, 197], [33, 114, 180], [198, 219, 240], [107, 174, 216], [8, 69, 149]],],
'ohio_Blues_8_pos_small1.jpg':[[[157, 202, 225], [66, 146, 197], [198, 219, 240], [247, 250, 255], [222, 234, 246], [8, 69, 149], [33, 114, 180], [107, 174, 216]],],
'ohio_RdBu_4_neg.jpg':[[[146, 198, 222], [244, 166, 130], [201, 0, 32], [7, 112, 177]],],
'ohio_RdBu_4_neg1.jpg':[[[146, 198, 222], [244, 166, 130], [201, 0, 32], [7, 112, 177]],],
'ohio_RdBu_4_nonAuto.jpg':[[[146, 198, 222], [244, 166, 130], [201, 0, 32], [7, 112, 177]],],
'ohio_RdBu_4_nonAuto1.jpg':[[[146, 198, 222], [244, 166, 130], [201, 0, 32], [7, 112, 177]],],
'ohio_RdBu_4_pos_large.jpg':[[[146, 198, 222], [244, 166, 130], [201, 0, 32], [7, 112, 177]],],
'ohio_RdBu_4_pos_large1.jpg':[[[146, 198, 222], [244, 166, 130], [201, 0, 32], [7, 112, 177]],],
'ohio_RdBu_4_pos_small.jpg':[[[244, 166, 130], [201, 0, 32], [146, 198, 222], [7, 112, 177]],],
'ohio_RdBu_4_pos_small1.jpg':[[[244, 166, 130], [146, 198, 222], [201, 0, 32], [7, 112, 177]],],
'ohio_RdBu_6_neg.jpg':[[[254, 219, 199], [104, 169, 207], [209, 229, 240], [177, 24, 44], [240, 138, 98], [33, 101, 172]],],
'ohio_RdBu_6_neg1.jpg':[[[177, 24, 44], [104, 169, 207], [209, 229, 240], [33, 101, 172], [240, 138, 98], [254, 219, 199]],],
'ohio_RdBu_6_nonAuto.jpg':[[[254, 219, 199], [33, 101, 172], [177, 24, 44], [209, 229, 240], [240, 138, 98], [104, 169, 207]],],
'ohio_RdBu_6_nonAuto1.jpg':[[[209, 229, 240], [104, 169, 207], [177, 24, 44], [254, 219, 199], [33, 101, 172], [240, 138, 98]],],
'ohio_RdBu_6_pos_large.jpg':[[[254, 219, 199], [177, 24, 44], [240, 138, 98], [33, 101, 172], [104, 169, 207], [209, 229, 240]],],
'ohio_RdBu_6_pos_large1.jpg':[[[254, 219, 199], [177, 24, 44], [104, 169, 207], [33, 101, 172], [209, 229, 240], [240, 138, 98]],],
'ohio_RdBu_6_pos_small.jpg':[[[254, 219, 199], [177, 24, 44], [209, 229, 240], [33, 101, 172], [240, 138, 98], [104, 169, 207]],],
'ohio_RdBu_6_pos_small1.jpg':[[[254, 219, 199], [104, 169, 207], [177, 24, 44], [240, 138, 98], [33, 101, 172], [209, 229, 240]],],
'ohio_RdBu_8_neg.jpg':[[[254, 219, 199], [146, 198, 222], [244, 166, 130], [177, 24, 44], [209, 229, 240], [67, 148, 195], [215, 95, 78], [33, 101, 172]],],
'ohio_RdBu_8_neg1.jpg':[[[177, 24, 44], [33, 101, 172], [67, 148, 195], [215, 95, 78], [146, 198, 222], [209, 229, 240], [244, 166, 130], [254, 219, 199]],],
'ohio_RdBu_8_nonAuto.jpg':[[[254, 219, 199], [33, 101, 172], [177, 24, 44], [146, 198, 222], [215, 95, 78], [244, 166, 130], [209, 229, 240], [67, 148, 195]],],
'ohio_RdBu_8_nonAuto1.jpg':[[[209, 229, 240], [146, 198, 222], [254, 219, 199], [33, 101, 172], [177, 24, 44], [215, 95, 78], [244, 166, 130], [67, 148, 195]],],
'ohio_RdBu_8_pos_large.jpg':[[[254, 219, 199], [215, 95, 78], [67, 148, 195], [177, 24, 44], [244, 166, 130], [33, 101, 172], [146, 198, 222], [209, 229, 240]],],
'ohio_RdBu_8_pos_large1.jpg':[[[254, 219, 199], [177, 24, 44], [67, 148, 195], [146, 198, 222], [244, 166, 130], [215, 95, 78], [33, 101, 172], [209, 229, 240]],],
'ohio_RdBu_8_pos_small.jpg':[[[254, 219, 199], [215, 95, 78], [209, 229, 240], [67, 148, 195], [177, 24, 44], [146, 198, 222], [244, 166, 130], [33, 101, 172]],],
'ohio_RdBu_8_pos_small1.jpg':[[[254, 219, 199], [146, 198, 222], [244, 166, 130], [177, 24, 44], [215, 95, 78], [33, 101, 172], [67, 148, 195], [209, 229, 240]],],
'us_Blues_4_neg.jpg':[[[107, 175, 212], [239, 243, 255], [191, 214, 230], [33, 114, 180]],],
'us_Blues_4_neg1.jpg':[[[107, 175, 212], [239, 243, 255], [191, 214, 230], [33, 114, 180]],],
'us_Blues_4_nonAuto.jpg':[[[33, 114, 180], [107, 175, 214], [239, 243, 255], [191, 214, 230]],],
'us_Blues_4_nonAuto1.jpg':[[[107, 174, 216], [33, 114, 180], [239, 243, 255], [191, 214, 230]],],
'us_Blues_4_pos_large.jpg':[[[239, 243, 255], [107, 174, 216], [191, 214, 230], [33, 114, 180]],],
'us_Blues_4_pos_large1.jpg':[[[239, 243, 255], [107, 174, 216], [107, 126, 140], [189, 215, 232]],(33, 114, 180)],
'us_Blues_4_pos_small.jpg':[[[33, 114, 180], [239, 243, 255], [191, 214, 230], [107, 174, 216], [49, 111, 162]],],
'us_Blues_4_pos_small1.jpg':[[[33, 114, 180], [107, 175, 214], [191, 214, 230], [239, 243, 255]],],
'us_Blues_6_neg.jpg':[[[107, 175, 212], [239, 243, 255], [198, 219, 240], [8, 82, 157], [157, 202, 225], [50, 130, 189], [110, 131, 150]],],
'us_Blues_6_neg1.jpg':[[[107, 175, 212], [239, 243, 255], [157, 202, 225], [8, 82, 157], [50, 130, 191], [111, 132, 149], [200, 218, 240]],],
'us_Blues_6_nonAuto.jpg':[[[8, 82, 153], [107, 175, 214], [239, 243, 255], [198, 219, 238], [50, 130, 189], [157, 202, 225], [132, 130, 131], [115, 132, 148]],],
'us_Blues_6_nonAuto1.jpg':[[[107, 175, 212], [8, 82, 157], [198, 218, 242], [157, 202, 225], [50, 130, 189], [241, 248, 255], [123, 127, 136]],],
'us_Blues_6_pos_large.jpg':[[[239, 243, 255], [54, 129, 186], [107, 174, 216], [157, 202, 225], [8, 82, 157], [198, 219, 240], [125, 134, 143]],],
'us_Blues_6_pos_large1.jpg':[[[239, 243, 255], [198, 219, 238], [107, 174, 216], [73, 126, 166], [157, 202, 225],[8, 82, 157]],],
'us_Blues_6_pos_small.jpg':[[[50, 130, 191], [239, 243, 255], [157, 202, 225], [200, 218, 240], [8, 82, 157], [107, 174, 216]],],
'us_Blues_6_pos_small1.jpg':[[[50, 130, 191], [107, 175, 214], [157, 202, 225], [239, 243, 255], [118, 132, 143], [0, 84, 171], [198, 219, 240]],(8, 82, 157)],
'us_Blues_8_neg.jpg':[[[107, 175, 212], [222, 234, 246], [198, 219, 240], [247, 250, 255], [33, 114, 180], [65, 147, 197], [157, 202, 225], [8, 69, 149], [95, 140, 159]],],
'us_Blues_8_neg1.jpg':[[[107, 175, 212], [222, 234, 246], [157, 202, 225], [8, 69, 149], [33, 114, 180], [66, 146, 197], [247, 250, 255], [198, 219, 240]],],
'us_Blues_8_nonAuto.jpg':[[[8, 69, 149], [107, 175, 214], [247, 250, 255], [198, 219, 238], [222, 234, 246], [66, 146, 197], [157, 202, 225], [33, 114, 180]],],
'us_Blues_8_nonAuto1.jpg':[[[107, 175, 212], [8, 69, 149], [222, 234, 246], [157, 202, 225], [33, 114, 180], [198, 219, 240], [65, 146, 199], [244, 248, 251], [92, 140, 160]],],
'us_Blues_8_pos_large.jpg':[[[66, 146, 197], [107, 174, 216], [157, 202, 225], [33, 114, 180], [247, 250, 255], [198, 219, 240], [121, 132, 136], [14, 68, 138]],],
'us_Blues_8_pos_large1.jpg':[[[222, 234, 246], [247, 250, 255], [66, 146, 197], [157, 202, 225], [198, 219, 240], [127, 132, 138], [108, 134, 149]],],
'us_Blues_8_pos_small.jpg':[[[33, 114, 180], [247, 250, 255], [157, 202, 225], [222, 234, 246], [8, 69, 149], [66, 146, 197], [107, 174, 216], [118, 127, 132]],(8, 69, 149)],
'us_Blues_8_pos_small1.jpg':[[[33, 114, 180], [107, 175, 214], [247, 250, 255], [157, 202, 225], [226, 233, 243], [80, 142, 183], [115, 132, 142], [11, 64, 158], [198, 219, 240]],],
'us_RdBu_4_neg.jpg':[[[146, 198, 220], [5, 113, 178], [244, 166, 130], [201, 0, 32], [109, 137, 141]],],
'us_RdBu_4_neg1.jpg':[[[146, 198, 220], [4, 114, 177], [244, 166, 130], [200, 1, 34], [117, 133, 146]],],
'us_RdBu_4_nonAuto.jpg':[[[2, 114, 180], [146, 198, 220], [201, 0, 32], [244, 166, 130], [120, 131, 117]],],
'us_RdBu_4_nonAuto1.jpg':[[[146, 198, 220], [7, 112, 177], [206, 0, 32], [244, 166, 130]],],
'us_RdBu_4_pos_large.jpg':[[[204, 0, 32], [146, 198, 220], [244, 165, 132], [5, 113, 177], [137, 122, 125]],],
'us_RdBu_4_pos_large1.jpg':[[[204, 0, 32], [244, 166, 128], [146, 198, 222], [0, 115, 188]],],
'us_RdBu_4_pos_small.jpg':[[[2, 114, 180], [146, 198, 220], [201, 0, 32], [244, 166, 130], [158, 131, 122]],],
'us_RdBu_4_pos_small1.jpg':[[[2, 114, 180], [146, 198, 220], [201, 0, 32], [244, 166, 128]],],
'us_RdBu_6_neg.jpg':[[[209, 229, 240], [176, 25, 44], [240, 138, 98], [34, 101, 172], [104, 170, 205], [254, 219, 199], [81, 113, 151]],],
'us_RdBu_6_neg1.jpg':[[[209, 229, 240], [179, 23, 44], [254, 219, 199], [33, 101, 172], [104, 170, 205], [123, 136, 144], [138, 55, 65], [110, 126, 141], [238, 139, 100]],],
'us_RdBu_6_nonAuto.jpg':[[[33, 101, 172], [209, 229, 240], [177, 24, 44], [240, 138, 98], [105, 169, 207], [254, 219, 199]],],
'us_RdBu_6_nonAuto1.jpg':[[[209, 229, 240], [241, 138, 95], [254, 219, 199], [104, 169, 207], [33, 101, 172], [112, 141, 157]],],
'us_RdBu_6_pos_large.jpg':[[[176, 25, 44], [104, 169, 207], [209, 228, 242], [254, 219, 199], [33, 101, 172],[112, 130, 144]],],
'us_RdBu_6_pos_large1.jpg':[[[176, 25, 44], [252, 220, 199], [209, 229, 240], [240, 138, 98], [121, 129, 132]],],
'us_RdBu_6_pos_small.jpg':[[[102, 170, 205], [209, 229, 240], [254, 219, 199], [177, 24, 44], [33, 101, 172], [240, 138, 98], [141, 125, 138], [116, 129, 137]],],
'us_RdBu_6_pos_small1.jpg':[[[102, 170, 207], [209, 229, 240], [254, 219, 199], [36, 100, 174], [177, 24, 44], [240, 138, 98], [117, 136, 134]],],
'us_RdBu_8_neg.jpg':[[[209, 229, 240], [33, 101, 172], [244, 166, 130], [67, 148, 195], [176, 25, 44], [144, 199, 220], [254, 219, 199], [138, 125, 108], [216, 95, 76]],],
'us_RdBu_8_neg1.jpg':[[[209, 229, 240], [33, 101, 172], [254, 219, 199], [66, 149, 193], [146, 198, 222], [215, 95, 78], [177, 24, 44], [244, 166, 130]],],
'us_RdBu_8_nonAuto.jpg':[[[33, 101, 172], [209, 229, 240], [177, 24, 44], [215, 95, 78], [146, 198, 222], [254, 219, 199], [67, 148, 195], [122, 141, 148], [244, 166, 130]],],
'us_RdBu_8_nonAuto1.jpg':[[[209, 229, 240], [146, 198, 220], [222, 93, 72], [254, 219, 199], [67, 148, 195], [33, 101, 172], [127, 127, 135], [104, 138, 165]],],
'us_RdBu_8_pos_large.jpg':[[[176, 25, 44], [146, 198, 220], [209, 228, 242], [254, 219, 199], [66, 149, 193], [244, 166, 130], [33, 101, 172], [144, 120, 108], [223, 93, 71]],],
'us_RdBu_8_pos_large1.jpg':[[[216, 95, 78], [242, 166, 130], [177, 24, 44], [146, 198, 222], [212, 225, 234], [255, 220, 198], [118, 132, 135], [93, 138, 171], [66, 148, 196]],(33, 101, 172)],
'us_RdBu_8_pos_small.jpg':[[[66, 148, 195], [209, 229, 240], [254, 219, 199], [215, 95, 78], [33, 101, 172], [146, 198, 222], [129, 130, 134], [110, 133, 151], [94, 141, 157]],],
'us_RdBu_8_pos_small1.jpg':[[[66, 148, 195], [209, 229, 240], [254, 219, 199], [146, 198, 222], [177, 24, 44],[108, 124, 140], [223, 91, 76], [37, 98, 179], [244, 166, 130]],],
}

def findDetectResult(detectResults, imageName):
    for dr in detectResults:
        name = dr[0]
        if name == imageName:
            return dr
    return None
# 1/(distance(pixelCoordList_sample[i], pixelCoordList_sample[j]) + 0.0000001)
def calculateKNeighWeight(i, coordList, k = 6): 
    # i is the currently target, k includes the point itself
    # k nearest neighborhood, only consider the k nearest neighbors
    # based on distance to calcuate weight, and normalize weight
    distList = []
    for j in range(len(coordList)):
        dist = distance(coordList[i], coordList[j])
        distList.append(dist)
    
    neighIndice = sorted(range(len(distList)), key = lambda sub: distList[sub])[:k]
    distNeigh = [distList[ind] for ind in neighIndice]

    neighIndice = neighIndice[1:]
    distNeigh = distNeigh[1:]
    wNeigh = [1/(distNeigh[j]+ 0.00000001) for j in range(len(distNeigh))]
    
    # normalize weights
    wNeighNorm = (np.asarray(wNeigh)/sum(wNeigh)).tolist()
    
    # put weights together
    wList = [0 for j in range(len(coordList))]
    for ind, j in enumerate(neighIndice):
        wList[j] = wNeighNorm[ind]
        
    return wList

def calculateKNeigh(targetCoord, coordList, k = 6): 
    # targetCoord is the coordinate of current point
    # k nearest neighborhood, only consider the k nearest neighbors
    # CoordList is the set of coordinates to compare
    distList = []
    for j in range(len(coordList)):
        dist = distance(targetCoord, coordList[j])
        distList.append(dist)
    neighIndice = sorted(range(len(distList)), key = lambda sub: distList[sub])[:k]
    return neighIndice

def most_common(lst):
    return max(set(lst), key=lst.count)

def mostCommonListHLine(y,xMin, xMax, pixelCoordList_sample,zList):
    mostCommonList = []
    for x in range(xMin, xMax + 1):
        coord = (x,y)
        neighIndice = calculateKNeigh(coord, pixelCoordList_sample)
        # find the most common class
        zListNeighIndice = [zList[ind] for ind in neighIndice]
        mostCommon = most_common(zListNeighIndice)
        mostCommonList.append(mostCommon)
    return mostCommonList

def mostCommonListVLine(x,yMin, yMax, pixelCoordList_sample,zList):
    mostCommonList = []
    for y in range(yMin, yMax + 1):
        coord = (x,y)
        neighIndice = calculateKNeigh(coord, pixelCoordList_sample)
        # find the most common class
        zListNeighIndice = [zList[ind] for ind in neighIndice]
        mostCommon = most_common(zListNeighIndice)
        mostCommonList.append(mostCommon)
    return mostCommonList

def countSegmentationFun(mostCommonList):
    countSegmentation = 1
    currentValue = mostCommonList[0]
    countDifferentTemp = 0
    for i in range(len(mostCommonList)):
        if mostCommonList[i] ==  currentValue:
            countDifferentTemp = 0
            continue
        else:
            countDifferentTemp += 1
            if countDifferentTemp > 20:
                currentValue = mostCommonList[i]
                countSegmentation += 1
    return countSegmentation

def main():
    # read detection results from pickle file
    detectResultsPath = r'C:\Users\li.7957\OneDrive - The Ohio State University\choroColorRead'
    detectResultFileName = 'detectResultSpatialPattern.pickle'
    with open(detectResultsPath + '\\' + detectResultFileName, 'rb') as f:
        detectResults = pickle.load(f)

    imagePath = r'C:\Users\li.7957\Desktop\choroColorRead\generatedMaps\classifiedQuantiles\pos_large'
    # imageName = 'ohio_Blues_4_neg1.jpg'
    testImages = os.listdir(imagePath)
    afterTarget = False
    for imageName in testImages:
        print('imageName: ' + imageName)
        detectResult = findDetectResult(detectResults, imageName)
        # if imageName == 'us_Blues_4_pos_small.jpg':
        #     afterTarget = True
        # if afterTarget == False:
        #     continue

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

        # get the polygon for mapping area
        for i in range(N):
            if class_ids[i] != 3:
                continue
            # Bounding box
            if not np.any(boxes[i]):
                # Skip this instance. Has no bbox. Likely lost in image cropping.
                continue
            y1, x1, y2, x2 = boxes[i]
            # Mask
            mask = masks[:, :, i]
            # print('shape of mask: ')
            # print(mask.shape)

        # rgb color list
        # colorList = [[191, 214, 230], [107, 174, 216], [33, 114, 180], [239, 243, 255]]
        colorList = spColorExtraction[imageName][0] # 0: detected colors, 1: missed colors

        # traverse the whole image
        colorSumRGBList = [sum(color) for color in colorList]
        colorGreyList = [rgb2Grey(color) for color in colorList]

        indexMinGreyColor = colorGreyList.index(min(colorGreyList))
        # colorGrey0 = rgb2Grey(colorList[0])
        # colorGrey1 = rgb2Grey(colorList[1])
        # colorGrey2 = rgb2Grey(colorList[2])
        # colorGrey3 = rgb2Grey(colorList[3])
        pixelCoordsList = [[] for i in range(len(colorGreyList)) ] # a set of coords for each color class
        pixelCoordList0, pixelCoordList1, pixelCoordList2, pixelCoordList3 = [],[],[],[]
        for i in range(height):
            # if i % 100 == 0:
            #     print('i: ' + str(i))   
            for j in range(mask.shape[1]):
                # point = Point(j,i)
                if mask[i,j] == True:
                    value =  imageGray[i,j]
                    b, g, r = image[i,j]
                    if not (abs(b - g) < 10 and abs(b-r) < 10 and abs(g - r)<10):
                        # if j == 879 and i == 105:
                        #     print('debug')
                        for k in range(len(colorGreyList)):
                            if abs(value - colorGreyList[k]) <= 10:
                                colorB, colorG, colorR = colorList[k]
                                if abs(b - colorB) > 10 and abs(g - colorG) > 10 and abs(r - colorR) > 10:
                                    continue
                                else:
                                    pixelCoordsList[k].append((j,i))
                                break  
        pixelCoordListSampleList = [[] for i in range(len(pixelCoordsList))]
        pixelCoordListSampleAll = []
        for i, pixelCoords in enumerate(pixelCoordsList):
            pixelCoordListSampleList[i] = sample(pixelCoordsList[i],int(len(pixelCoordsList[i])/100))
            pixelCoordListSampleAll += pixelCoordListSampleList[i]

        # clustering
        kmax = 100
        silKmeansResultList = []
        zListList = []
        zListAll = []
        for i, pixelCoordsSample in enumerate(pixelCoordListSampleList):
            kmax = min(100,len(pixelCoordsSample))
            silKmeansResult = silKmeans(kmax, pixelCoordsSample)
            silKmeansResultList.append(silKmeansResult)
            zList = [colorSumRGBList[i] for j in range(len(pixelCoordListSampleList[i]))]
            zListList.append(zList)
            zListAll += zList 

        # clustering
        # kmax = 100
        # # kmeans0 = silKmeans(kmax, pixelCoordList0_sample)
        # # kmeans1 = silKmeans(kmax, pixelCoordList1_sample)
        # kmeansHighestClass = silKmeans(kmax, pixelCoordListSampleList[indexMinGreyColor])
        # # kmeans3 = silKmeans(kmax, pixelCoordList3_sample)
        zCentersList = []
        zCentersAll = []
        coordCentersList = []
        coordCentersAll = []
        needLineGrid = False
        for i,silKmeansResult in enumerate(silKmeansResultList):
            # print('debug')
            zCentersTemp = [colorSumRGBList[i] for j in range(silKmeansResult.cluster_centers_.shape[0])]
            zCentersList.append(zCentersTemp)
            if len(zCentersTemp) <= 10:
                needLineGrid = True
            zCentersAll += zCentersTemp
            coordCenters = silKmeansResult.cluster_centers_.tolist()
            coordCentersList.append(coordCenters)
            coordCentersAll += coordCenters
        

        ### any number of cluster centers not larger than 10, conduct grid line algorithm
        if needLineGrid:
            xList = [pixelCoord[0] for pixelCoord in pixelCoordListSampleAll]
            yList = [pixelCoord[1] for pixelCoord in pixelCoordListSampleAll]
            xMin, xMax = min(xList), max(xList)
            yMin, yMax = min(yList), max(yList)
            yMid = int((yMin + yMax) / 2)

            mostCommonListLines = []
            for y in range(yMin, yMax + 1, int((yMax + 1 - yMin) / 10)):
                # print(y)
                mostCommonList = mostCommonListHLine(y, xMin, xMax, pixelCoordListSampleAll,zListAll)
                mostCommonListLines.append(mostCommonList)
            
            for x in range(xMin, xMax + 1, int((xMax + 1 - xMin) / 10)):
                # print(x)
                mostCommonList = mostCommonListVLine(x, yMin, yMax, pixelCoordListSampleAll,zListAll)
                mostCommonListLines.append(mostCommonList)


            countSegmentationList = []
            for mostCommonList in mostCommonListLines:
                countSegmentation = countSegmentationFun(mostCommonList)
                countSegmentationList.append(countSegmentation)

            countSegmentation = max(countSegmentationList)
            totalNumCluster = countSegmentation**2
            numClusterEach = math.ceil(totalNumCluster / len(colorGreyList))
            print('new numClusterEach: '+str(numClusterEach))

            kmeansResultList = []
            coordCentersList = []
            for i, pixelCoordsSample in enumerate(pixelCoordListSampleList):
                if len(pixelCoordListSampleList[i]) < numClusterEach:
                    continue
                else:
                    kmeansResult = KMeans(n_clusters = numClusterEach).fit(pixelCoordListSampleList[i])
                # kmeansResult = KMeans(n_clusters = numClusterEach).fit(pixelCoordListSampleList[i])
                kmeansResultList.append(kmeansResult)
                zList = [colorSumRGBList[i] for j in range(len(pixelCoordListSampleList[i]))]
                zListList.append(zList)
                zListAll += zList 

            # kmeansHighestClass = KMeans(n_clusters = numClusterEach).fit(pixelCoordListSampleList[indexMinGreyColor])
            zCentersList = []
            zCentersAll = []
            coordCentersList = []
            coordCentersAll = []
            for i,kmeansResult in enumerate(kmeansResultList):
                # print('debug')
                zCentersTemp = [colorSumRGBList[i] for j in range(kmeansResult.cluster_centers_.shape[0])]
                zCentersList.append(zCentersTemp)
                
                zCentersAll += zCentersTemp
                coordCenters = kmeansResult.cluster_centers_.tolist()
                coordCentersList.append(coordCenters)
                coordCentersAll += coordCenters

        ### Calculate moran's I for cluster centers
        numCenters = len(zCentersAll)
        wMatrix = np.zeros(shape=(numCenters, numCenters))
        for i in range(numCenters):
            wMatrix[i] = calculateKNeighWeight(i, coordCentersAll)
            # print(wMatrix[i])
        w = wMatrix.tolist()
        moranIClusterCenters = moransi(zCentersAll, w) # use line grid to decide number of clusters
        print('moranIClusterCenters:' + str(moranIClusterCenters))

        ### calculate theoretical expected value and variance of Moran's I
        # expected value
        N = numCenters
        print(N)
        expI = -(1/(N - 1))
        print('expI: ' + str(expI))

        # variance
        s0 = sum([sum(wRow) for wRow in w])
        s1 = 0
        for i in range(len(w)):
            for j in range(len(w[0])):
                s1 += (w[i][j] + w[j][i])**2
        s1 = s1 /2
        s2 = 0
        for k in range(N):
            s2kj = 0
            s2ik = 0
            for j in range(N):
                s2kj += w[k][j]
            for i in range(N):
                s2ik += w[i][k]
            s2 += (s2kj + s2ik)**2
        varI = ((N**2)*s1 - N*s2 + 3*(s0**2))/((N**2 -1)*(s0**2)) - expI**2
        sigma = varI**(0.5)
        thresholdLower = expI - 2 * sigma
        print('thresholdLower: ' + str(thresholdLower))
        thresholdUpper = expI + 2 * sigma
        print('thresholdUpper: ' + str(thresholdUpper))

        if moranIClusterCenters < thresholdLower:
            print('Negative spatial autocorrelation')
        elif moranIClusterCenters > thresholdUpper:
            print('Positive spatial autocorrelation')
        else:
            print('No clear spatial autocorrelation')



    

if __name__ == "__main__":    main()