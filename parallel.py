import sys
import cv2 as cv
import numpy as np
from numba import jit, cuda
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
 
 
@cuda.jit
def convert_rgb2gray_kernel(in_pixels, out_pixels, width, height):
    '''
    Convert color image to grayscale image.
 
    in_pixels : numpy.ndarray with shape=(h, w, 3)
                h, w is height, width of image
                3 is colors with BGR (blue, green, red) order
        Input RGB image
    
    out_pixels : numpy.ndarray with shape=(h, w)
        Output image in grayscale
    '''
    r = cuda.blockIdx.x*cuda.blockDim.x+cuda.threadIdx.x
    c = cuda.blockIdx.y*cuda.blockDim.y+cuda.threadIdx.y

    if r<height and c<width:
        out_pixels[r][c] = (0.114*in_pixels[r, c, 0] + 
                            0.587*in_pixels[r, c, 1] + 
                            0.299*in_pixels[r, c, 2])

def test_convert_rgb2gray_kernel(img, gray_img):
    '''
    Test convert_rgb2gray function
    '''
 
    gray_img_np = (img @ [0.114, 0.587, 0.299]).astype(np.uint8)
    gray_img_cv = cv.cvtColor(img, code=cv.COLOR_BGR2GRAY)
 
    print('Jitted vs Numpy  error:', 
          np.mean(np.abs(gray_img.astype(np.int16) - gray_img_np)))
    print('Jitted vs Opencv error:', 
          np.mean(np.abs(gray_img.astype(np.int16) - gray_img_cv)))

 
@cuda.jit
def calculate_sat_kernel_x(in_pixels, sat, sqsat):
    y = cuda.blockIdx.x*cuda.blockDim.x + cuda.threadIdx.x

    if y < len(in_pixels):
        for c in range(len(in_pixels[0])):
            sat[y + 1, c + 1] = sat[y + 1, c] + in_pixels[y, c]
            sqsat[y + 1, c + 1] = sqsat[y + 1, c] + in_pixels[y, c]**2
        
@cuda.jit
def calculate_sat_kernel_y(in_pixels, sat, sqsat):
    x = cuda.blockIdx.x*cuda.blockDim.x + cuda.threadIdx.x

    if x < len(in_pixels[0]):
        for r in range(len(in_pixels)):
            sat[r + 1, x + 1] += sat[r, x + 1]
            sqsat[r + 1, x + 1] += sqsat[r, x + 1]
 
def test_calculate_sat_kernel(img, sat):
    '''
    Test calculate_sat function
    '''
 
    sat_np = np.cumsum(img, axis=0, dtype=np.int64)
    np.cumsum(sat_np, axis=1, out=sat_np)
 
    total = np.sum(img)
    assert(total == sat[-1, -1])
    assert(total == sat_np[-1, -1])
    assert(np.array_equal(sat[1:, 1:], sat_np))
 

@jit(nopython=True)
def get_left_or_right(window_size, gray_img, sqsat, i, j, feature_id, feature_vals,rect_counts, rectangles, scale):
    '''
    Decide to turn left or right
    Return value is 0 or 1, respective to LEFT or RIGHT
    
    '''

    w, h = int(scale*window_size[0]), int(scale*window_size[1])
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


@jit(nopython=True)
def check_passed(i, j, model, sat, sqsat, stage_thresholds_id,scale):
    '''

    If the sum of these values exceeds the threshold, the stage passes
    else it fails (the window is not the object looked for).

    '''
    window_size, stage_thresholds, tree_counts, feature_vals, rect_counts, rectangles = model
    sum = 0
    for feature_id in range(tree_counts[stage_thresholds_id],tree_counts[stage_thresholds_id+1]):
        sum += get_left_or_right(window_size, sat, sqsat, i, j, feature_id, feature_vals[feature_id], rect_counts, rectangles, scale)

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

@jit(nopython=True)
def equals(r1, r2):
    distance = np.int16(r1[2]*0.2)
    if r2[0] <= r1[0] + distance and \
        r2[0] >= r1[0] - distance and \
        r2[1] <= r1[1] + distance and \
        r2[1] >= r1[1] - distance and \
        r2[2] <= np.int16( r1[2] * 1.2 ) and \
        np.int16( r2[2] * 1.2) >= r1[2]:
        return True

    if r1[0] >= r2[0] and r1[0] + r1[2] <= r2[0] + r2[2] and \
         r1[1] >= r2[1] and r1[1] + r1[3] <= r2[1] + r2[3]:
        return True
    return False

@jit(nopython=True)
def merge(rects, threshold):
    retour = []
    rects_size = rects.shape[0]
    ret = np.empty(rects_size, dtype = np.int16)
    nb_classes = 0
    for i in range(rects_size):
        found = False
        for j in range(i):
            if equals(rects[j], rects[i]):
                found = True
                ret[i] = ret[j]
        if not found:
            ret[i] = nb_classes
            nb_classes += 1
    
    neighbors = np.empty(nb_classes)
    rect = np.empty((nb_classes, 4), dtype = np.int32)
    for i in range(nb_classes):
        neighbors[i] = 0
        rect[i] = np.array([0, 0, 0, 0])

    for i in range(rects_size):
        neighbors[ret[i]] += 1
        rect[ret[i]][0] += rects[i][0]
        rect[ret[i]][1] += rects[i][1]
        rect[ret[i]][2] += rects[i][2]
        rect[ret[i]][3] += rects[i][3]

    for idx in range(nb_classes):
        n = neighbors[idx]
        if n >= threshold:
            r = np.array([0, 0, 0, 0], dtype = np.int16)
            r[0] = (rect[idx][0]*2 + n)/(2*n)
            r[1] = (rect[idx][1]*2 + n)/(2*n)
            r[2] = (rect[idx][2]*2 + n)/(2*n)
            r[3] = (rect[idx][3]*2 + n)/(2*n)
            retour.append(r)
    return retour

def draw(merge_img, color, thickness, out_img):
    for rect in merge_img:
        start_point = (rect[0], rect[1])
        end_point = (rect[0] + rect[2], rect[1] + rect[3])
        out_img = cv.rectangle(out_img, start_point, end_point, color, thickness)

def evaluate_convert_rgb2gray_kernel(in_img, block_size):
    height, width = in_img.shape[:-1]
    
    # Allocate device memories
    d_gray_img = cuda.device_array((height, width), dtype = in_img.dtype)

    # Set grid size and invoke kernel
    grid_size = ((height-1) // block_size[1]+1, (width-1) // block_size[0]+1)
    convert_rgb2gray_kernel[grid_size,block_size](in_img,d_gray_img,width,height)

    # Copy data to host
    gray_img = d_gray_img.copy_to_host().astype(in_img.dtype)
    print(gray_img)

    # Compute error
    test_convert_rgb2gray_kernel(in_img,gray_img)

    # Write image
    cv.imwrite("output.jpg", gray_img)

    return gray_img

def evaluate_calculate_sat_kernel(gray_img, block_size):
    height, width = gray_img.shape[0], gray_img.shape[1]

    d_sat = cuda.device_array((height + 1, width + 1), dtype=np.int64)
    d_sqsat = cuda.device_array((height + 1, width + 1), dtype=np.int64)

    d_sat[:-1], d_sqsat[:-1] = 0, 0

    calculate_sat_kernel_x[(width-1) // block_size+1, block_size](gray_img.astype(np.int64), d_sat, d_sqsat)
    calculate_sat_kernel_y[(height-1) // block_size+1, block_size](gray_img.astype(np.int64), d_sat, d_sqsat)

    sat = d_sat.copy_to_host().astype(np.int64)
    sqsat = d_sqsat.copy_to_host().astype(np.int64)

    test_calculate_sat_kernel(gray_img, sat)
    test_calculate_sat_kernel(np.power(gray_img, 2, dtype=np.int64), sqsat)

def main():
    # Read arguments
    if len(sys.argv) != 3:
        print('python sequential.py INPUT OUTPUT')
        sys.exit(1)
    ifname = sys.argv[1]
    ofname = sys.argv[2]
 
    # Read image
    in_img = cv.imread(ifname)

    # Run grayscale kernel
    gray_img = evaluate_convert_rgb2gray_kernel(in_img,(32,32))

    # Run sat kernel
    evaluate_calculate_sat_kernel(gray_img, 32)


    # out_img = in_img.copy()

    # Run 
    # run(model,in_img,out_img)

    # Write image
    # cv.imwrite(ofname, out_img)
    
# Execute
main()
