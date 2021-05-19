import sys
import cv2 as cv
import numpy as np
from numba import jit
import xml.etree.ElementTree as ET
import math
 
def load_model(file_name):
    '''
    Load Opencv's Haar Cascade pre-trained model.
    
    Parameter
    ---------
    filename: Name of the file from which the classifier is loaded.
 
    Returns
    -------
    A tuple contains below numpy arrays:

    window_size : tuple with shape=(2)
        Base width and height of detection window.

    stage_thresholds : numpy.ndarray with shape=(num_stages)
        num_stages is number of stage used in the classifier.
        threshold of each stage to check if we proceed to the next stage.
 
    tree_counts : numpy.ndarray with shape=(num_stages + 1) 
        `tree_counts[i]` is the number of tree/feature before stage `i` i.e. 
        index of the first tree of stage `i`. Therefore, 
        `range(tree_counts[i], tree_counts[i + 1])` will give range of trees' 
        index (in `feature_vals` array) of stage `i`.
 
    feature_vals : numpy.ndarray with shape(num_features, 3)
        num_features is the total number of feature used in the classifier.
        3 is (threshold, left_val, right_val) of each tree.
        Each feature correspond to a tree.
    
    rect_counts : numpy.ndarray with shape(num_features + 1)
        A feature consists of 2 or 3 rectangles. `rect_counts[i]` is the index 
        of the first rectangle of feature `i`. Therefore, 
        `range(rect_counts[i], rect_counts[i + 1])` give all rectangle's index 
        (in `rectangles` array) of feature `i`.
 
    rectangles : numpy.ndarray with shape(num_rectangles, 5)
        num_rectangles is the total number of rectangle of all features in the 
        classifier.
        5 is (x_topleft, y_topleft, width, height, weight) of each rectangle.
    '''
 
    xmlr = ET.parse(file_name).getroot()
    cascade = xmlr.find('cascade')
    stages = cascade.find('stages')
    features = cascade.find('features')
 
    window_size = (int(cascade.find('width').text), 
                   int(cascade.find('height').text))
 
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
 
'''
decide to turn left or right
return value is 0 or 1, respective to LEFT or RIGHT
'''

@jit(nopython=True)
def get_left_or_right(x, y, gray_img, sqsat, i, j, feature_id, feature_vals, rect_counts, rectangles, scale):
    w, h = int(scale*x), int(scale*y)
    inv_area = 1/(w*h)
    total_x = gray_img[i + h][j + w]+ gray_img[i][j] - gray_img[i][j + w] - gray_img[i + h][j]
    total_x2 = sqsat[i + h][j + w] + sqsat[i][j] - sqsat[i][j + w] - sqsat[i + h][j]
    moy = total_x * inv_area
    vnorm = total_x2 * inv_area - moy * moy
    vnorm = math.sqrt(vnorm) if vnorm > 1 else 1
        
    rect_sum = 0
    for rect_id in range(rect_counts[feature_id], rect_counts[feature_id + 1]):
        r = rectangles[rect_id]
        rx1 = j + np.int16(scale*r[0])
        rx2 = j + np.int16(scale*(r[2] + r[0]))
        ry1 = i + np.int16(scale*r[1])
        ry2 = i + np.int16(scale*(r[3] + r[1]))
        rect_sum += np.int32((gray_img[ry2][rx2] - gray_img[ry2][rx1] - gray_img[ry1][rx2] + gray_img[ry1][rx1])*r[4])

    rect_sum2 = np.float(rect_sum)*inv_area
    threshold = feature_vals[0]
    return feature_vals[1] if rect_sum2 < threshold*vnorm else feature_vals[2]

'''

If the sum of these values exceeds the threshold, the stage passes
else it fails (the window is not the object looked for).

'''
@jit(nopython=True)
def check_passed(i, j, model, sat, sqsat, stage_thresholds_id,scale):
    window_size, stage_thresholds, tree_counts, feature_vals, rect_counts, rectangles = model
    sum = 0
    for feature_id in range(tree_counts[stage_thresholds_id],tree_counts[stage_thresholds_id+1]):
        sum += get_left_or_right(window_size[0], window_size[1], sat, sqsat, i, j, feature_id, feature_vals[feature_id], rect_counts, rectangles,scale)

    return sum>stage_thresholds[stage_thresholds_id]
        

@jit(nopython=True)
def detect(gray_image, model, sat, sqsat, scale_inc):
    window_size, stage_thresholds, tree_counts, feature_vals, rect_counts, rectangles = model
    face_list = np.empty((0, 4), np.int32)
    width = gray_image.shape[1]
    height = gray_image.shape[0]
    max_scale = min(np.float(width)/window_size[0], np.float(height)/window_size[1])
    print("max_scale: ", max_scale)

    scale = 1.0
    while scale < max_scale:
        size_x, size_y = np.int16(scale*window_size[0]), np.int16(scale*window_size[1])
        step = 1
        i = 0
        while i < height - size_y:
            j = 0
            while j < width - size_x:
                passed = True
                for stage_thresholds_id in range(len(stage_thresholds)):
                    if not check_passed(i, j, model, sat, sqsat, stage_thresholds_id,scale):
                        passed = False
                        break
                
                if passed:
                    print("Passed!")
                    # face_list.append([j,j,size_x,size_y])
                    face_list = np.append(face_list, np.array([[j, i, size_x, size_y]], dtype = np.int16), axis = 0)

                j += step
            i += step
        scale *= scale_inc
    

    return face_list


def main():
    # Read arguments
    if len(sys.argv) != 3:
        print('python sequential.py INPUT OUTPUT')
        sys.exit(1)
    # mfname = sys.argv[1]
    ifname = sys.argv[1]
    ofname = sys.argv[2]
 
    #Load Haar Cascade model
    model = load_model('haarcascade_frontalface_default.xml')
 
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
 
    # Write image
    cv.imwrite(ofname, gray_img)

    window_size, stage_thresholds, tree_counts, feature_vals, rect_counts, rectangles = model

    # print(feature_vals)

    # value = get_left_or_right(window_size[0], window_size[1], gray_img, sqsat, 0, 0, 0, feature_vals, rect_counts, rectangles, 1.5)
    # print(value)

    # passed = check_passed(0, 0, window_size[0], window_size[1], 2, sat, sqsat, feature_vals, rect_counts, rectangles, tree_counts, stage_thresholds, 0)

    face_list = detect(gray_img, model, sat, sqsat, 1.5)
    print("face_list:", face_list)
    
    
# Execute
main()