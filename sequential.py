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
    height = in_pixels.shape[0]
    width = in_pixels.shape[1]

    for r in range(0,height):
        for c in range(0,width):
            i = r*width+c
            red = img[r][c][0]
            green = img[r][c][1]
            blue = img[r][c][2]
    #         print(red,green,blue)
            out_pixels[r][c] = 0.299*red + 0.587*green + 0.114*blue
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

    height = in_pixels.shape[0]
    width = in_pixels.shape[1]

    for r in range(0,height):
        for c in range(0,width):
            sat[r][c]=in_pixels[r][c] + sat[r][c-1] + sat[r-1,c] - sat[r-1][c-1];    

    raise NotImplementedError()


def main():
    # Read arguments
    if len(sys.argv) != 3:
        print('python sequential.py INPUT OUTPUT')
        sys.exit(1)
    ifname = sys.argv[1]
    ofname = sys.argv[2]

    # Read image
    file_in = open("in.pnm","r")
    file_type = file_in.readline()
    dimension = file_in.readline().split()
    width = int(size[0])
    height = int(size[1])
    max_val = int(file_in.readline())
    rgb = file_in.read().split()
    rgb = [int(i) for i in rgb]
    rgb = [width,height]+rgb
    raise NotImplementedError()
    # img = ...
    img = np.array(rgb)
    img = np.delete(img,[0,1])
    img = img.reshape(width,height,3)


    # Convert image to grayscale using jitted function
    gray_img = np.empty((img.shape[0], img.shape[1]), dtype=img.dtype)
    convert_rgb2gray(img, gray_img)

    # Convert image to grayscale using Numpy function/operator
    raise NotImplementedError()
    # gray_img_np = ...
    gray_img_np = np.dot(img[...,:3], [0.299, 0.587, 0.114])
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