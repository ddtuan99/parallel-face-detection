import sys
import numpy as np
from numba import jit
import cv2

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
        for c in range(len(in_pixels[r])):
            blue = in_pixels[r][c][0]
            green = in_pixels[r][c][1]
            red = in_pixels[r][c][2]
            out_pixels[r][c] = 0.114*blue + 0.587*green + 0.299*red


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
    height = len(in_pixels)
    width = len(in_pixels[0])
    for r in range(height):
        for c in range(width):
            x_left = c - 1
            y_above = r - 1
            sat[r][c] += in_pixels[r][c]
            if x_left > -1 and x_left < width:
                sat[r][c] += sat[r][x_left]
                if y_above > -1 and y_above < height:
                    sat[r][c] += sat[y_above][c] - sat[y_above][x_left]
            elif y_above > -1 and y_above < height:
                sat[r][c] += sat[y_above][c]


def main():
    # Read arguments
    if len(sys.argv) != 3:
        print('python sequential.py INPUT OUTPUT')
        sys.exit(1)
    ifname = sys.argv[1]
    ofname = sys.argv[2]

    # Read image
    im = cv2.imread(ifname)

    # Convert image to grayscale using jitted function
    gray_img = np.empty((im.shape[0], im.shape[1]), dtype=im.dtype)
    convert_rgb2gray(im, gray_img)

    # Convert image to grayscale using Numpy function/operator
    rgb_weights = [0.114, 0.587, 0.299]
    gray_img_np = np.dot(im[...,:3], rgb_weights)
    
    # Test convert_rgb2gray
    print('Jitted vs Numpy error:', 
          np.mean(np.abs(gray_img.astype(np.int16) - gray_img_np)))
    print('Jitted vs Opencv error:', 
          np.mean(np.abs(gray_img.astype(np.int16) - 
                         cv2.cvtColor(im, code=cv2.COLOR_BGR2GRAY))))

    # Calculate summed area table using jitted function
    sat = np.empty(gray_img.shape, dtype=np.int64)
    calculate_sat(gray_img, sat)

    # Calculate summed area table using Numpy function/operator
    sat_np = gray_img.cumsum(axis=0).cumsum(axis=1)

    # Test
    assert(np.sum(gray_img) == sat[-1, -1])
    assert(np.sum(gray_img) == sat_np[-1, -1])
    assert(np.array_equal(sat, sat_np))

    # Write image
    cv2.imwrite(ofname, gray_img)

# Execute
main()