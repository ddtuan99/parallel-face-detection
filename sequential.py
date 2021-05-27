import sys
import cv2 as cv
import numpy as np
from numba import jit
from numba.typed import List
import xml.etree.ElementTree as ET


def load_model(file_name):
    '''
    Load Opencv's Haar Cascade pre-trained model.
    
    Parameter
    ---------
    file_name: Name of the file from which the classifier is loaded.
 
    Returns
    -------
    A tuple contains below numpy arrays:

    window_size : numpy.ndarray with shape=(2)
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


@jit(nopython=True)
def calc_sum_rect(sat, loc, rect, scale):
    '''
    Evaluate feature.
    '''

    tlx = loc[0] + np.int32(rect[0] * scale)
    tly = loc[1] + np.int32(rect[1] * scale)
    brx = loc[0] + np.int32((rect[0] + rect[2]) * scale)
    bry = loc[1] + np.int32((rect[1] + rect[3]) * scale)
    return (sat[tly, tlx] + sat[bry, brx] - sat[bry, tlx] - sat[tly, brx]) * rect[4]


@jit(nopython=True)
def detect_at(model, sats, point, scale):
    '''
    Detect object with a specific scale.
    '''

    base_win_sz, stage_thresholds, tree_counts, \
        feature_vals, rect_counts, rectangles = model
    sat, sqsat = sats

    w, h = int(base_win_sz[0] * scale), int(base_win_sz[1] * scale)
    inv_area = 1 / (w * h)

    x, y = point
    win_sum = sat[y][x] + sat[y+h][x+w] - sat[y][x+w] - sat[y+h][x]
    win_sqsum = sqsat[y][x] + sqsat[y+h][x+w] - sqsat[y][x+w] - sqsat[y+h][x]
    variance = win_sqsum * inv_area - (win_sum * inv_area) ** 2

    # Reject low-variance intensity region, also take care of negative variance
    # value due to inaccuracy floating point operation.
    if variance < 100:
        return -1

    std = np.sqrt(variance)

    num_stages = len(stage_thresholds)
    for stg_idx in range(num_stages):
        stg_sum = 0.0
        for tr_idx in range(tree_counts[stg_idx], tree_counts[stg_idx+1]): 
            # Implement stump-base decision tree (tree has one feature) for now.
            # Each feature consists of 2 or 3 rectangle.
            rect_idx = rect_counts[tr_idx]
            ft_sum = (calc_sum_rect(sat, point, rectangles[rect_idx], scale) + 
                      calc_sum_rect(sat, point, rectangles[rect_idx+1], scale))
            if rect_idx + 2 < rect_counts[tr_idx+1]:
                ft_sum += calc_sum_rect(sat, point, rectangles[rect_idx+2], scale)
            
            # Compare ft_sum/(area*std) with threshold to choose return value.
            stg_sum += (feature_vals[tr_idx][1] 
                        if ft_sum * inv_area < feature_vals[tr_idx][0] * std 
                        else feature_vals[tr_idx][2])

        if stg_sum < stage_thresholds[stg_idx]:
            return stg_idx
    
    return num_stages

 
@jit(nopython=True)
def detect_multi_scale(model, sats, out_img, base_scale=1.0, 
                       scale_factor=1.1, min_neighbors=3):
    '''

    '''

    win_size = model[0]
    num_stages = len(model[1])
    scale = base_scale
    height, width = out_img.shape[:2]
    max_scale = min(width / win_size[0], height / win_size[1])

    rec_list = List()
    while scale < max_scale:
        cur_win_size = (win_size * scale).astype(np.int32)
        step = int((2 if scale < 2 else 1) * scale)
        for y in range(0, height - cur_win_size[1], step):
            for x in range(0, width - cur_win_size[0], step):
                if detect_at(model, sats, (x, y), scale) == num_stages:
                    rec_list.append(np.array([x, y, cur_win_size[0], cur_win_size[1]]))
        scale *= scale_factor

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
        return rectangles

    num_rects = len(rectangles)
    num_classes = 0

    groups = List()
    num_members = List()
    labels = np.empty(num_rects, dtype=np.int32)
    for i in range(num_rects):
        r1 = rectangles[i]
        new_group = True
        for j in range(i):
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
def run(model, in_img, out_img, debug):
    '''
    Implement object detection workflow.
    '''

    height, width = in_img.shape[:2]

    # Convert image to grayscale
    gray_img = np.empty((height, width), dtype=in_img.dtype)
    convert_rgb2gray(in_img, gray_img)
    # if debug: test_convert_rgb2gray(in_img, gray_img)
 
    # Calculate Summed Area Table (SAT) and squared SAT
    sat = np.empty((height + 1, width + 1), dtype=np.int64)
    sqsat = np.empty((height + 1, width + 1), dtype=np.int64)
    calculate_sat(gray_img, sat, sqsat)
    # if debug: test_calculate_sat(gray_img, sat)
    # if debug: test_calculate_sat(np.power(gray_img, 2, dtype=np.int64), sqsat)

    # Detect object
    candidates = detect_multi_scale(model, (sat, sqsat), out_img)

    # Group candidates
    final_detections = group_rectangles(candidates, 4)

    # Draw bounding box on output image
    for rec in final_detections:
        draw_rect(out_img, rec)

 
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

    sat_np = np.array(img, dtype=np.int64)
    sat_np.cumsum(axis=0, out=sat_np).cumsum(axis=1, out=sat_np)
 
    total = np.sum(img)
    assert(total == sat[-1, -1])
    assert(total == sat_np[-1, -1])
    assert(np.array_equal(sat[1:, 1:], sat_np))


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
