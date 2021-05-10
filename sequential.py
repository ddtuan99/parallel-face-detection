import sys
import cv2 as cv
import numpy as np
import xml.etree.ElementTree as ET
from numba import jit
 
@jit(nopython=True)
def convert_rgb2gray(in_pixels, out_pixels):
    '''
    Convert color image to grayscale image.
 
    in_pixels : numpy.ndarray with shape=(h, w, 3)
                h, w is height, width of image
                3 is colors with BGR (blue, green, red) order
        Input RGB image
    
    out_pixels : numpy.ndarray with shape=(h, w)
        Output image in grayscale
    '''
 
    for r in range(len(in_pixels)):
        for c in range(len(in_pixels[0])):
            out_pixels[r, c] = (in_pixels[r, c, 0] * 0.114 + 
                                in_pixels[r, c, 1] * 0.587 + 
                                in_pixels[r, c, 2] * 0.299)
 
 
@jit(nopython=True)
def calculate_sat(in_pixels, sat):
    '''
    Calculate Summed Area Table (Integral image)
 
    in_pixels : numpy.ndarray with shape=(h, w)
                h, w is height, width of image
        Grayscale image need to calculate SAT
    
    sat : numpy.ndarray with shape=(h, w)
        Summed Area Table of input image
    '''

    sat[0, 0] = in_pixels[0, 0]
    for c in range(1, len(in_pixels[0])):
        sat[0, c] = sat[0, c - 1] + in_pixels[0, c]
    for r in range(1, len(in_pixels)):
        row_sum = 0
        for c in range(len(in_pixels[0])):
            row_sum += in_pixels[r, c]
            sat[r, c] = row_sum + sat[r - 1, c]
 
 
def test_convert_rgb2gray(img, gray_img):
    '''
    Test convert_rgb2gray function
    '''

    gray_img_np = (img @ [0.114, 0.587, 0.299]).astype(np.uint8)
    gray_img_cv = cv.cvtColor(img, code=cv.COLOR_BGR2GRAY)
 
    print('Jitted vs Numpy  error:', 
          np.mean(np.abs(gray_img.astype(np.int16) - gray_img_np)))
    print('Jitted vs Opencv error:', 
          np.mean(np.abs(gray_img.astype(np.int16) - gray_img_cv)))
 
 
def test_calculate_sat(img, sat):
    '''
    Test calculate_sat function
    '''
 
    sat_np = np.cumsum(img, axis=0, dtype=np.int64)
    np.cumsum(sat_np, axis=1, out=sat_np)
 
    total = np.sum(img) 
    assert(total == sat[-1, -1])
    assert(total == sat_np[-1, -1])
    assert(np.array_equal(sat, sat_np))
 
def load_model(file_name):
    '''
    Loads a classifier from a file

    filename: Name of the file from which the classifier is loaded
    '''

    tree = ET.parse('haarcascade_frontalface_default.xml')
    root = tree.getroot()
    #root[0]

    stage_thresholds = np.empty(0, dtype = float)
    tree_counts = np.empty(1, dtype = int)
    feature_vals = np.empty((0,3), dtype = float)
    rectangles = np.empty((0,5), dtype = float)
    rect_counts = np.empty(0, dtype = int)

    for cascade in root:
        for cascade_item in cascade:
            # print(cascade, cascade_item)
            if cascade_item.tag == "stages":
                for stage in cascade_item:
                    # results = [int(i) for i in results]
                    stage_thresholds = np.append(stage_thresholds, [float(i) for i in stage.find('stageThreshold').text.split()])
                    tree_counts = np.append(tree_counts, [int(i) for i in stage.find('maxWeakCount').text.split()][0] + tree_counts[len(tree_counts)-1])

                    for stage_item in stages:
                        # print(stage_item)
                        if stage_item.tag == "weakClassifiers":
                            for trees in stage_item:
                                internal_nodes = [float(i) for i in trees.find('internalNodes').text.split()]
                                leaf_values = [float(i) for i in trees.find('leafValues').text.split()]
                                feature_vals = np.append(feature_vals,np.array([[internal_nodes[3],leaf_values[0],leaf_values[1]]]),0)
                                
            if cascade_item.tag == "features":
                for feature in cascade_item:
                    for rects in feature:
                        rect_counts = np.append(rect_counts, len(rects))                  
                        for rect in rects:
                            # print(rect.text.split())
                            # rect_info = np.array([list(map(float, rect.text.split()))])
                            rectangles = np.vstack([rectangles, [float(i) for i in rect.text.split()]])

    # len(rectangles)

    
    '''
    stage_thresholds: numpy.ndarray with shape=(nStages)
                    nStages is number of stage used in the classifier
                    threshold of each stage to check if whether should we proceed to the next stage or not

    tree_counts: numpy.ndarray with shape=(nStages + 1) 
                tree_counts[i] contains number of tree/feature before stage i or index of the first tree of stage i,
                so range(tree_counts[i], tree_counts[i + 1]) will gives all tree's index of stage i

    feature_vals: numpy.ndarray with shape(nFeatures, 3)
                nFeatures is total number of features used in the classifier
                Contains (threshold, left_val, right_val) of each features, each feature correspond to a tree with the same index

    rectangles: numpy.ndarray with shape(nRectangles, 5)
                nRectangles is total number of rectangles used for features in the classifier
                Contains (x_topleft, y_topleft, width, height, weight) of each rectangle

    rect_counts: numpy.ndarray with shape(nFeatures + 1)
                A feature consists of 2 or 3 rectangles. rect_counts[i] is the index of first rectangle of feature i,
                so range(rect_counts[i], rect_counts[i + 1]) give all rectangle's index (in rectangles array) of feature i
    '''
    return tuple([stage_thresholds, tree_counts, feature_vals, rect_counts, rect_list])

 
def main():
    # Read arguments
    if len(sys.argv) != 3:
        print('python sequential.py INPUT OUTPUT')
        sys.exit(1)
    ifname = sys.argv[1]
    ofname = sys.argv[2]
 
    # Read image
    img = cv.imread(ifname)
 
    # Convert image to grayscale
    gray_img = np.empty((img.shape[0], img.shape[1]), dtype=img.dtype)
    convert_rgb2gray(img, gray_img)
    test_convert_rgb2gray(img, gray_img)
 
    # Calculate summed area table
    sat = np.empty(gray_img.shape, dtype=np.int64)
    calculate_sat(gray_img, sat)
    test_calculate_sat(gray_img, sat)
 
    # Write image
    cv.imwrite(ofname, gray_img)
 
 
# Execute
main()