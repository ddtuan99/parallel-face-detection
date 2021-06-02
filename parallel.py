import xml.etree.ElementTree as ET
import numpy as np
import math
from numba import jit, prange, cuda, int64
from numba.typed import List
from numba import config
import numba
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

@cuda.jit(device=True)
def calc_sum_rect(sat, loc, rect, scale):
    '''
    Evaluate feature.
    '''

    tlx = loc[0] + np.int32(rect[0] * scale)
    tly = loc[1] + np.int32(rect[1] * scale)
    brx = loc[0] + np.int32((rect[0] + rect[2]) * scale)
    bry = loc[1] + np.int32((rect[1] + rect[3]) * scale)
    return (sat[tly, tlx] + sat[bry, brx] - sat[bry, tlx] - sat[tly, brx]) * rect[4]


 
@cuda.jit()
def detect_kernel(base_win_sz, stage_thresholds, tree_counts, \
                    feature_vals, rect_counts, rectangles, sats, scale, scale_idx, sld_w, sld_h, step, is_pass):
    # Detect multiple windows asynchronously
    x = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y*cuda.blockDim.y
    x_sat = x * step
    y_sat = y * step

    sat, sqsat = sats
    w, h = np.int32(base_win_sz[0] * scale), np.int32(base_win_sz[1] * scale)
    inv_area = 1 / (w * h)

    if x_sat < sld_w and y_sat < sld_h:
        win_sum = sat[y_sat][x_sat] + sat[y_sat+h][x_sat+w] - sat[y_sat][x_sat+w] - sat[y_sat+h][x_sat]
        win_sqsum = sqsat[y_sat][x_sat] + sqsat[y_sat+h][x_sat+w] - sqsat[y_sat][x_sat+w] - sqsat[y_sat+h][x_sat]
        variance = win_sqsum * inv_area - (win_sum * inv_area) ** 2

        # Reject low-variance intensity region, also take care of negative variance
        # value due to inaccuracy floating point operation.
        if variance < 100:
            return 

        std = math.sqrt(variance)

        num_stages = len(stage_thresholds)
        for stg_idx in prange(num_stages):
            stg_sum = 0.0
            for tr_idx in prange(tree_counts[stg_idx], tree_counts[stg_idx+1]): 
                # Implement stump-base decision tree (tree has one feature) for now.
                # Each feature consists of 2 or 3 rectangle.
                rect_idx = rect_counts[tr_idx]
                ft_sum = (calc_sum_rect(sat, (x_sat, y_sat), rectangles[rect_idx], scale) + 
                        calc_sum_rect(sat, (x_sat, y_sat), rectangles[rect_idx+1], scale))
                if rect_idx + 2 < rect_counts[tr_idx+1]:
                    ft_sum += calc_sum_rect(sat, (x_sat, y_sat), rectangles[rect_idx+2], scale)
                
                # Compare ft_sum/(area*std) with threshold to choose return value.
                stg_sum += (feature_vals[tr_idx][1] 
                            if ft_sum * inv_area < feature_vals[tr_idx][0] * std 
                            else feature_vals[tr_idx][2])

            if stg_sum < stage_thresholds[stg_idx]:
                return 
        print((x_sat + y_sat * sld_w) + base_win_sz[0] * base_win_sz[1] * scale_idx)
        is_pass[(x_sat + y_sat * sld_w) + base_win_sz[0] * base_win_sz[1] * scale_idx][0] = True


def detect_multi_scale(model, sats, out_img, base_scale=1.0, 
                       scale_factor=1.1):
    # Slide the detector window
    win_size = model[0]
    scale = base_scale
    scale_idx = 0
    height, width = out_img.shape[:2]
    max_scale = min(width / win_size[0], height / win_size[1])
    rect_size = 0

    base_win_sz, stage_thresholds, tree_counts, \
        feature_vals, rect_counts, rectangles = model

    base_win_size = (win_size * base_scale).astype(np.int32)       
    base_sld_w = (width - base_win_size[0])
    base_sld_h = (height - base_win_size[1])
    base_rect_size = base_sld_w * base_sld_h
    
    while scale < max_scale:
        rect_size += base_rect_size
        scale *= scale_factor

    d_is_pass = cuda.device_array((rect_size, 1), dtype=np.bool)

    d_sats = numba.cuda.to_device(sats)
    scale = base_scale 
    block_size = (32, 32)

    while scale < max_scale:
        cur_win_size = (win_size * scale).astype(np.int32)
        step = int((2 if scale < 2 else 1) * scale)
        grid_size = (math.ceil((width - cur_win_size[0]) / block_size[0]), 
                math.ceil((height - cur_win_size[1]) / block_size[1]))        
        detect_kernel[grid_size, block_size](base_win_sz, stage_thresholds, tree_counts, \
                                                    feature_vals, rect_counts, rectangles, d_sats, scale, scale_idx, width - cur_win_size[0], height - cur_win_size[1], step, d_is_pass)
        scale *= scale_factor
        scale_idx += 1
    is_pass = d_is_pass.copy_to_host().astype(np.bool)
    end = time.time()
    
    is_pass = List(is_pass)

    return is_pass

@cuda.jit(device = True)
def sum_rec(rec):
    return rec[0] + rec[1] + rec[2] + rec[3]

@jit(nopython = True)
def delete_zero(win_size, height, width, base_scale, scale_factor, is_pass):
    scale = base_scale
    max_scale = min(width / win_size[0], height / win_size[1])    
    rec_list = List()
    scale_idx = 0

    while scale < max_scale:
        cur_win_size = (win_size * scale).astype(np.int32)
        step = int((2 if scale < 2 else 1) * scale)
        sld_h = height - cur_win_size[1]
        sld_w = width - cur_win_size[0]
        for y in range(0, sld_h, step):
            for x in range(0, sld_w, step):
                if is_pass[(x + y * sld_w) + win_size[0] * win_size[1] * scale_idx][0] == True:
                    is_pass[(x + y * sld_w) + win_size[0] * win_size[1] * scale_idx][0] = False
                    rec_list.append(np.array([x, y, cur_win_size[0], cur_win_size[1]]))
        scale *= scale_factor
        scale_idx += 1
    return rec_list


@jit(nopython=True)
def group_rectangles(rectangles, min_neighbors=3, eps=0.2):
    '''
    Group object candidate rectangles.

    Parameters
    ----------
    rectangles: list(np.array(4))
        List of rectangles

    min_neighbors: int
        Minimum neighbors each candidate rectangle should have to retain it.

    eps: float
        Relative difference between sides of the rectangles to merge them into 
    a group.

    Return
    ------
    A list of grouped rectangles.
    '''
    if min_neighbors == 0:

        final_list = List(rectangles)
        return final_list

    num_rects = len(rectangles)
    num_classes = 0

    groups = List()
    num_members = List()
    labels = np.empty(num_rects, dtype=np.int32)
    for i in range(num_rects):
        if rectangles[i][0] == -1:
            continue
        r1 = rectangles[i]
        new_group = True
        for j in range(i):
            if rectangles[j][0] == -1:
                continue
            r2 = rectangles[j]
            delta = eps * (min(r1[2], r2[2]) + min(r1[3], r2[3])) * 0.5
            if (abs(r1[0] - r2[0]) <= delta and 
                abs(r1[1] - r2[1]) <= delta and 
                abs(r1[0] + r1[2] - r2[0] - r2[2]) <= delta and 
                abs(r1[1] + r1[3] - r2[1] - r2[3]) <= delta):
                new_group = False
                labels[i] = labels[j]
                groups[labels[j]] += r1
                num_members[labels[j]] += 1
                break
        if new_group:
            groups.append(r1)
            num_members.append(1)
            labels[i] = num_classes
            num_classes += 1

    # Filter out groups which don't have enough rectangles
    i = 0
    while i < num_classes:
        while num_members[i] <= min_neighbors and i < num_classes:
            num_classes -= 1
            groups[i] = groups[num_classes]
            num_members[i] = num_members[num_classes]
        groups[i] //= num_members[i]
        i += 1

    # Filter out small rectangles inside large rectangles
    final_list = List()
    for i in range(num_classes):
        r1 = groups[i]
        m1 = max(3, num_members[i])
        is_good = True
        for j in range(num_classes):
            if i == j:
                continue
            r2 = groups[j]
            dx, dy = r2[2] * 0.2, r2[3] * 0.2
            if (r1[0] >= r2[0] - dx and 
                r1[1] >= r2[1] - dy and 
                r1[0] + r1[2] <= r2[0] + r2[2] + dx and 
                r1[1] + r1[3] <= r2[1] + r2[3] + dy and 
                num_members[j] > m1):
                is_good = False
                break
        if is_good:
            final_list.append(r1)

    return final_list

@jit(nopython=True)
def draw_rect(img, rect, color=0, thick=1):
    '''
    Draw bounding box on image.
    '''

    (x, y, w, h), t = rect, thick
    img[y:y+h, x-t:x+t+1] = color
    img[y:y+h, x+w-t:x+w+t+1] = color
    img[y-t:y+t+1, x:x+w] = color
    img[y+h-t:y+h+t+1, x:x+w] = color    

def run(model, in_img, out_img, debug):
    '''
    Implement object detection workflow.
    '''


    height, width = in_img.shape[:2]

    # Convert image to grayscale
    d_gray_img = cuda.device_array((height, width), dtype = in_img.dtype)

    block_size_rgb2gray = (32, 32)
    grid_size_rgb2gray = (math.ceil(in_img.shape[1] / block_size_rgb2gray[0]), 
                math.ceil(in_img.shape[0] / block_size_rgb2gray[1]))

    convert_rgb2gray_kernel[grid_size_rgb2gray, block_size_rgb2gray](in_img, in_img.shape[1], in_img.shape[0], d_gray_img)
    gray_img = d_gray_img.copy_to_host().astype(in_img.dtype)
    test_convert_rgb2gray(in_img, gray_img)    

    # Calculate Summed Area Table (SAT) and squared SAT
    d_sat = cuda.device_array((height + 1, width + 1), dtype=np.int64)
    d_sqsat = cuda.device_array((height + 1, width + 1), dtype=np.int64)
    d_sat[:-1], d_sqsat[:-1] = 0, 0

    block_size_sat = 32
    grid_size_sat_y = math.ceil(in_img.shape[1]/block_size_sat)
    grid_size_sat_x = math.ceil(in_img.shape[0]/block_size_sat)

    calculate_sat_kernel_x[grid_size_sat_x, block_size_sat](gray_img.astype(np.int64), d_sat, d_sqsat)
    calculate_sat_kernel_y[grid_size_sat_y, block_size_sat](gray_img.astype(np.int64), d_sat, d_sqsat)

    sat = d_sat.copy_to_host().astype(np.int64)
    sqsat = d_sqsat.copy_to_host().astype(np.int64)
    
    test_calculate_sat(gray_img, sat)
    test_calculate_sat(np.power(gray_img, 2, dtype=np.int64), sqsat)
    # if debug: test_calculate_sat(gray_img, sat)
    # if debug: test_calculate_sat(np.power(gray_img, 2, dtype=np.int64), sqsat)

    # Detect object
    start = time.time()
    candidates = detect_multi_scale(model, (sat, sqsat), out_img)
    print(len(candidates))
    
    end = time.time()
    print(end - start)
    unique, counts = np.unique(candidates, return_counts=True)
    print(dict(zip(unique, counts)))

    result = delete_zero(model[0], height, width, 1.0, 1.1, candidates)
    print(len(result))
    end3 = time.time()
    print(end3 - end)
    # Group candidates

    print(result[:100])

    final_detections = group_rectangles(result, 0)

    # Draw bounding box on output image
    for rec in final_detections:
        draw_rect(out_img, rec)


def main(_argv):
    argv = _argv if _argv else sys.argv

    # Read arguments
    if len(argv) != 4:
        print('python sequential.py MODEL INPUT OUTPUT')
        sys.exit(1)
    mfname = argv[1]
    ifname = argv[2]
    ofname = argv[3]
 
    #Load Haar Cascade model
    model = load_model(mfname)
 
    # Read input image
    in_img = cv.imread(ifname)
    out_img = in_img.copy()
 
    # Run object detection workflow
    run(model, in_img, out_img, False)
 
    # Write output image
    cv.imwrite(ofname, out_img)

# Execute
main(None)