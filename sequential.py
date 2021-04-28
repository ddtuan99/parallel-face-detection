import sys
import numpy as np
from numba import jit
# import ...

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

    raise NotImplementedError()


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

    raise NotImplementedError()


def main():
    # Read arguments
    if len(sys.argv) != 3:
        print('python sequential.py INPUT OUTPUT')
        sys.exit(1)
    ifname = sys.argv[1]
    ofname = sys.argv[2]

    # Read image
    raise NotImplementedError()
    # img = ...

    # Convert image to grayscale using jitted function
    gray_img = np.empty((img.shape[0], img.shape[1]), dtype=img.dtype)
    convert_rgb2gray(img, gray_img)

    # Convert image to grayscale using Numpy function/operator
    raise NotImplementedError()
    # gray_img_np = ...
    # ...

    # Test convert_rgb2gray
    print('Jitted vs Numpy error:', 
          np.mean(np.abs(gray_img.astype(np.int16) - gray_img_np)))
    print('Jitted vs Opencv error:', 
          np.mean(np.abs(gray_img.astype(np.int16) - 
                         cv.cvtColor(img, code=cv.COLOR_RGB2GRAY))))

    # Calculate summed area table using jitted function
    sat = np.empty(gray_img.shape, dtype=np.int64)
    calculate_sat(gray_img, sat)

    # Calculate summed area table using Numpy function/operator
    raise NotImplementedError()
    # sat_np = ...
    # ...

    # Test
    assert(np.sum(gray_img) == sat[-1, -1])
    assert(np.sum(gray_img) == sat_np[-1, -1])
    assert(np.array_equal(sat, sat_np))

    # Write image
    raise NotImplementedError()


# Execute
main()