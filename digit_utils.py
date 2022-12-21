import numpy as np
import cv2


def get_bounding_box(img):
    col_sum = np.where(np.sum(img, axis = 0)>0)
    row_sum = np.where(np.sum(img, axis = 1)>0)
    y1, y2 = row_sum[0][0], row_sum[0][-1]
    x1, x2 = col_sum[0][0], col_sum[0][-1]
    bbox = [x1, y1, x2, y2, x2 - x1, y2 - y1]
    
    return bbox

def add_padding(img, pad_l, pad_t, pad_r, pad_b):
    height, width = img.shape
    #Adding padding to the left side.
    pad_left = np.zeros((height, pad_l), dtype = np.int)
    img = np.concatenate((pad_left, img), axis = 1)

    #Adding padding to the top.
    pad_up = np.zeros((pad_t, pad_l + width))
    img = np.concatenate((pad_up, img), axis = 0)

    #Adding padding to the right.
    pad_right = np.zeros((height + pad_t, pad_r))
    img = np.concatenate((img, pad_right), axis = 1)

    #Adding padding to the bottom
    pad_bottom = np.zeros((pad_b, pad_l + width + pad_r))
    img = np.concatenate((img, pad_bottom), axis = 0)

    return img

def prep_digit(img):
    # Transform
    img = cv2.resize(img, (28, 28), interpolation = cv2.INTER_LINEAR)
    img = cv2.bitwise_not(img)
    (thresh, img) = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY)
    
    # Centering
    ### ToDo: Improve it before using
    # bbox = get_bounding_box(img)
    # padd_x = int((28 - bbox[4]) / 2)
    # padd_y = int((28 - bbox[5]) / 2)
    # cropped_image = img[bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1]
    # padded_image = add_padding(cropped_image,
    #                             padd_x, padd_y, 
    #                             27 - bbox[4] - padd_x, 27 - bbox[5] - padd_y)
    
    return img