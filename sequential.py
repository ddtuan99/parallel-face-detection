import sys
from numpy.core.arrayprint import set_string_function

from numpy.lib.function_base import append
import cv2 as cv
import numpy as np
from numba.typed import List
from numba import jit
import xml.etree.ElementTree as ET
import time
import math

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
def calculate_sat(in_pixels, sat, sqsat):
    '''
    Calculate Summed Area Table (SAT) and Squared SAT
 
    in_pixels : numpy.ndarray with shape=(h, w)
                h, w is height, width of image
        Grayscale image need to calculate SAT
    
    sat : numpy.ndarray with shape=(h + 1, w + 1)
        SAT 0-padding at top and left side of input image
 
    sqsat : numpy.ndarray with shape=(h + 1, w + 1)
        Squared SAT 0-padding at top and left side of input image
    '''

    sat[0, :], sqsat[0, :] = 0, 0
    for r in range(len(in_pixels)):
        row_sum, row_sqsum = 0, 0
        sat[r + 1, 0], sqsat[r + 1, 0] = 0, 0
        for c in range(len(in_pixels[0])):
            row_sum += in_pixels[r, c]
            row_sqsum += in_pixels[r, c] ** 2
            sat[r + 1, c + 1] = row_sum + sat[r, c + 1]
            sqsat[r + 1, c + 1] = row_sqsum + sqsat[r, c + 1]
 
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
    assert(np.array_equal(sat[1:, 1:], sat_np))
 
def load_model(file_name):
    '''
    Loads a classifier from a file
    filename: Name of the file from which the classifier is loaded
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

    xmlr = ET.parse(file_name).getroot()
    cascade = xmlr.find('cascade')
    stages = cascade.find('stages')
    features = cascade.find('features')

    window_size = np.array([int(cascade.find('width').text), 
                   int(cascade.find('height').text)])

    num_stages = len(stages)
    num_features = len(features)

    stage_thresholds = np.empty(num_stages)
    tree_counts = np.empty(num_stages + 1, dtype=np.int16)
    feature_vals = np.empty((num_features, 3), dtype=np.float64)

    ft_cnt = 0
    tree_counts[0] = 0
    for stage_idx, stage in enumerate(stages):
        num_trees = stage.find('maxWeakCount').text
        stage_threshold = stage.find('stageThreshold').text
        weak_classifiers = stage.find('weakClassifiers')
        tree_counts[stage_idx + 1] = tree_counts[stage_idx] + np.int16(num_trees)
        stage_thresholds[stage_idx] = np.float64(stage_threshold)
        for tree in weak_classifiers:
            node = tree.find('internalNodes').text.split()
            leaf = tree.find('leafValues').text.split()
            feature_vals[ft_cnt][0] = np.float64(node[3])
            feature_vals[ft_cnt][1] = np.float64(leaf[0])
            feature_vals[ft_cnt][2] = np.float64(leaf[1])
            ft_cnt += 1

    rect_counts = np.empty(num_features + 1, dtype=np.int16)

    rect_counts[0] = 0
    for ft_idx, feature in enumerate(features):
        rect_count = len(feature.find('rects'))
        rect_counts[ft_idx + 1] = rect_counts[ft_idx] + np.int16(rect_count)

    rectangles = np.empty((rect_counts[-1], 5), np.int8)

    rect_cnt = 0
    for feature in features:
        rects = feature.find('rects')
        for rect in rects:
            rect_vals = rect.text.split()
            rectangles[rect_cnt][0] = np.int8(rect_vals[0])
            rectangles[rect_cnt][1] = np.int8(rect_vals[1])
            rectangles[rect_cnt][2] = np.int8(rect_vals[2])
            rectangles[rect_cnt][3] = np.int8(rect_vals[3])
            rectangles[rect_cnt][4] = np.int8(rect_vals[4][:-1])
            rect_cnt += 1

    return (window_size, stage_thresholds, tree_counts, 
            feature_vals, rect_counts, rectangles)


@jit(nopython=True)
def evalute_features(sat, pos, ftr_index, rect_counts, rects, scale):
    '''
    Compute the sum (and squared sum) of the pixel values in the window, and get the mean and variance of pixel values
	in the window.
    '''
    i, j = pos
    rect_sum = 0.0
    # Each feature consists of 2 or 3 rectangle.
    # For each rectangle in the feature
    for idx in range(rect_counts[ftr_index], rect_counts[ftr_index + 1]):
        rx1 = np.int32(scale*rects[idx][0]) + j
        rx2 = np.int32(scale*(rects[idx][2] + rects[idx][0])) + j
        ry1 = np.int32(scale*rects[idx][1]) + i
        ry2 = np.int32(scale*(rects[idx][3] + rects[idx][1])) + i
        # Add the sum of pixel values in the rectangles (weighted by the rectangle's weight) to the total sum 
        rect_sum += np.double(sat[ry2][rx2] - sat[ry2][rx1] - sat[ry1][rx2] + sat[ry1][rx1])*rects[idx][4]

    return rect_sum

@jit(nopython = True)
def stage_pass(base_wd_size, stage, tree_counts, sats, ftr_vals, rect_counts, rects, pos, scale):
    '''
    Calculate sum of a stage by iterating features of the stage.
    The sum then used to compare with stage threshold to return the corresponded node value.
    '''
    ftr_sum = 0.0
    w, h = int(scale*base_wd_size[0]), int(scale*base_wd_size[1])
    inv_area=1.0 / (w*h)
    stg_index, threshold = stage
    sat, sqsat = sats
    i, j = pos

    sat_sum = sat[i + h][j + w] + sat[i][j] - sat[i][j + w] - sat[i + h][j]
    squared_sum = sqsat[i + h][j + w] + sqsat[i][j] - sqsat[i][j + w] - sqsat[i + h][j]
    mean = sat_sum * inv_area
    variance = squared_sum * inv_area - np.double(mean) * mean

    vnorm = np.sqrt(variance)

    for ftr_index in range(tree_counts[stg_index], tree_counts[stg_index + 1]):
        # Implement stump-base decision tree (tree has one feature) for now.
        rect_sum = evalute_features(sat, (i, j), ftr_index, rect_counts, rects, scale)

        # threshold > rect_sum/(area*vnorm) ? left : right
        rect_sum2 = rect_sum*inv_area
        if (rect_sum2 < ftr_vals[ftr_index][0]*vnorm):
            ftr_sum += ftr_vals[ftr_index][1]
        else:
            ftr_sum += ftr_vals[ftr_index][2]

    if ftr_sum < threshold:
        return False

    return True

@jit(nopython=True)
def detect_multi_scale(wd_size, stg_threshold, tree_counts, sat, sqsat, ftr_vals, rect_counts, rects, shift_ratio = 0.1):
    # Slide the detector window
    width, height = len(sat[0]), len(sat)
    max_scale = min([width/wd_size[0], height/wd_size[1]])
    scale_inc = 1.1
    scale = 1.0
    result = List()

    while scale < max_scale:
        cur_wd_size = (scale*wd_size).astype(np.int32)
        step = (cur_wd_size*shift_ratio).astype(np.int32)
        # Compute the sliding step of the window
        for i in range(1, height - cur_wd_size[1], step[1]):
            for j in range(1, width - cur_wd_size[0], step[0]):
                accepted = True
                for stg_idx, threshold in enumerate(stg_threshold):
                    if not stage_pass(wd_size, (stg_idx, threshold), tree_counts, (sat, sqsat), ftr_vals, rect_counts, rects, (i, j), scale):
                        accepted = False
                        break
                    
                if accepted == True:
                    result.append([j, i, int(cur_wd_size[0]), int(cur_wd_size[1])])
    
        scale *= scale_inc
    return result
    
@jit(nopython=True)
def merge(rects, threshold):
    '''
    Merged multiple rectanges that may contain faces into groups and form new rectangles from those group.
    Groups with number of neighbor higher than threshold will be kept.
    '''
    retour = List()
    ret = np.empty(len(rects), dtype = np.int32)
    nb_classes = 0
    for i in range(len(rects)):
        found = False
        for j in range(i):
            if equals(rects[j], rects[i]):
                found = True
                ret[i] = ret[j]
        if found == False:
            ret[i] = nb_classes
            nb_classes += 1
    neighbors = np.empty(nb_classes)
    rect = np.empty((nb_classes, 4), dtype = np.int32)
    for idx in range(nb_classes):
        neighbors[idx] = 0
        rect[idx] = np.array([0, 0, 0, 0])

    for i in range(len(rects)):
        neighbors[ret[i]] += 1
        rect[ret[i]][0] += rects[i][0]
        rect[ret[i]][1] += rects[i][1]
        rect[ret[i]][2] += rects[i][2]
        rect[ret[i]][3] += rects[i][3]

    for idx in range(nb_classes):
        n = neighbors[idx]
        if n >= threshold:
            r = np.array([0, 0, 0, 0], dtype = np.int32)
            r[0] = (rect[idx][0]*2 + n)/(2*n)
            r[1] = (rect[idx][1]*2 + n)/(2*n)
            r[2] = (rect[idx][2]*2 + n)/(2*n)
            r[3] = (rect[idx][3]*2 + n)/(2*n)
            retour.append(r)
    return retour

@jit(nopython=True)
def equals(r1, r2):
    '''
    Check if two rectanges are near eachother of contain eachother in order to group those rectangles
    '''
    distance = np.int32(r1[2]*0.2)
    if r2[0] <= r1[0] + distance and r2[0] >= r1[0] - distance and \
                r2[1] <= r1[1] + distance and \
                r2[1] >= r1[1] - distance and \
                r2[2] <= np.int32( r1[2] * 1.2 ) and \
                np.int32( r2[2] * 1.2) >= r1[2]:
                    return True
    if r1[0] >= r2[0] and r1[0] + r1[2] <= r2[0] + r2[2] and r1[1] >= r2[1] and r1[1] + r1[3] <= r2[1] + r2[3]:
        return True    
    return False

def main():
    # Read arguments
    if len(sys.argv) != 4:
        print('python sequential.py MODEL INPUT OUTPUT')
        sys.exit(1)
    mfname = sys.argv[1]
    ifname = sys.argv[2]
    ofname = sys.argv[3]
 
    # Read image
    img = cv.imread(ifname)
    height, width = img.shape[:-1]
 
    # Convert image to grayscale
    gray_img = np.empty((height, width), dtype=img.dtype)
    convert_rgb2gray(img, gray_img)
    test_convert_rgb2gray(img, gray_img)
 
    # Calculate Summed Area Table (SAT) and squared SAT
    sat = np.empty((height + 1, width + 1), dtype=np.int64)
    sqsat = np.empty((height + 1, width + 1), dtype=np.int64)
    calculate_sat(gray_img, sat, sqsat)
    test_calculate_sat(gray_img, sat)
    test_calculate_sat(np.power(gray_img, 2, dtype=np.int64), sqsat)

    # Load model
    wd_size, stg_threshold, tree_counts, ftr_vals, rect_counts, rects = load_model(mfname)

    shift_ratio = 0.05
    start = time.time()
    result = detect_multi_scale(wd_size, stg_threshold, tree_counts, sat, sqsat, ftr_vals, rect_counts, rects, shift_ratio)
    end = time.time()
    print("Time: " + str(end - start))
    # result = [tuple(r) for r in result]
    # merged_result = cv.groupRectangles(result, 3)[0]

    result = np.array(result)
    merged_result = merge(result, 0)

    color = (255, 255, 255)
    thickness = 2
    detected_img = img.copy()

    for rect in merged_result:
        start_point = (rect[0], rect[1])
        end_point = (rect[0] + rect[2], rect[1] + rect[3])
        detected_img = cv.rectangle(detected_img, start_point, end_point, color, thickness)

    # Write image
    cv.imwrite(ofname, detected_img)

# Execute
main()