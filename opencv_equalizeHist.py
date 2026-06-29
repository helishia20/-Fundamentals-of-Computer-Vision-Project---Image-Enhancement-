import cv2 as cv
"""
استفاده از کتابخانه ی open cv  
"""


def opencv_eq_Hist(img):
    return cv.equalizeHist(img)