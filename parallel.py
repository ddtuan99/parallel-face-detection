from re import S
import numpy as np
import math
from numba import jit, prange, cuda, int64
import numba
from numba import config
import sys
import cv2 as cv
import time

@cuda.jit()
def convert_rgb2gray_kernel(in_pixels, width, height, out_pixels):
  # gray = 0.299*red + 0.587*green + 0.114*blue  
  x = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x
  y = cuda.threadIdx.y + cuda.blockIdx.y*cuda.blockDim.y

  if x < width and y < height:
    red = float(in_pixels[y][x][0])
    green = float(in_pixels[y][x][1])
    blue = float(in_pixels[y][x][2])
    out_pixels[y][x] = 0.299*red + 0.587*green + 0.114*blue

@cuda.jit()
def calculate_sat_kernel_x(in_pixels, sat, sqsat):
    y = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x

    if y < len(in_pixels):
        for c in range(len(in_pixels[0])):
            sat[y + 1, c + 1] = sat[y + 1, c] + in_pixels[y, c]
            sqsat[y + 1, c + 1] = sqsat[y + 1, c] + in_pixels[y, c]**2
        
@cuda.jit()
def calculate_sat_kernel_y(in_pixels, sat, sqsat):
    x = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x

    if x < len(in_pixels[0]):
        for r in range(len(in_pixels)):
            sat[r + 1, x + 1] += sat[r, x + 1]
            sqsat[r + 1, x + 1] += sqsat[r, x + 1]
 
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

@cuda.jig
def detect_kernel(wd_size, stg_threshold, tree_counts, sat, sqsat, ftr_vals, rect_counts, rects):
    # Detect multiple windows asynchronously
    pass

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
    d_gray_img = cuda.device_array((height, width), dtype = img.dtype)

    block_size_rgb2gray = (32, 32)
    grid_size_rgb2gray = (math.ceil(img.shape[1] / block_size_rgb2gray[0]), 
                math.ceil(img.shape[0] / block_size_rgb2gray[1]))
    convert_rgb2gray_kernel[grid_size_rgb2gray, block_size_rgb2gray](img, img.shape[1], img.shape[0], d_gray_img)
    gray_img = d_gray_img.copy_to_host().astype(img.dtype)
    test_convert_rgb2gray(img, gray_img)    

    # Calculate Summed Area Table (SAT) and squared SAT
    d_sat = cuda.device_array((height + 1, width + 1), dtype=np.int64)
    d_sqsat = cuda.device_array((height + 1, width + 1), dtype=np.int64)
    d_sat[:-1], d_sqsat[:-1] = 0, 0

    block_size_sat = 32
    grid_size_sat_y = math.ceil(img.shape[1]/block_size_sat)
    grid_size_sat_x = math.ceil(img.shape[0]/block_size_sat)

    calculate_sat_kernel_x[grid_size_sat_x, block_size_sat](gray_img.astype(np.int64), d_sat, d_sqsat)
    calculate_sat_kernel_y[grid_size_sat_y, block_size_sat](gray_img.astype(np.int64), d_sat, d_sqsat)
    sat = d_sat.copy_to_host().astype(np.int64)
    sqsat = d_sqsat.copy_to_host().astype(np.int64)
    
    test_calculate_sat(gray_img, sat)
    test_calculate_sat(np.power(gray_img, 2, dtype=np.int64), sqsat)

    # Load model
    wd_size, stg_threshold, tree_counts, ftr_vals, rect_counts, rects = load_model(mfname)

    # shift_ratio = 0.05
    # start = time.time()
    # result = detect_multi_scale(wd_size, stg_threshold, tree_counts, sat, sqsat, ftr_vals, rect_counts, rects, shift_ratio)
    # end = time.time()
    # print("Time: " + str(end - start))
    # # result = [tuple(r) for r in result]
    # # merged_result = cv.groupRectangles(result, 3)[0]

    # result = np.array(result)
    # merged_result = merge(result, 4)

    # color = (255, 255, 255)
    # thickness = 2
    # detected_img = img.copy()

    # for rect in result:
    #     start_point = (rect[0], rect[1])
    #     end_point = (rect[0] + rect[2], rect[1] + rect[3])
    #     detected_img = cv.rectangle(detected_img, start_point, end_point, color, thickness)

    # Write image
    cv.imwrite(ofname, gray_img)

# Execute
main()