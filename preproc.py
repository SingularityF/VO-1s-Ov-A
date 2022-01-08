import cv2
import tensorflow as tf


def tf_int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def tf_bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        # BytesList won't unpack a string from an EagerTensor.
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def debug_contour(img, contours):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(image=img, contours=contours, contourIdx=-1,
                     color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
    cv2.imshow("", img)
    cv2.waitKey(0)


def line_detector(img, debug_flag=False):
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2000, 2))
    dilation = cv2.dilate(img, rect_kernel, iterations=1)
    contours, hierarchy = cv2.findContours(
        dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    lines = []
    if debug_flag:
        debug_contour(img, contours)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        lines.append((x, y, w, h))
    return sorted(lines, key=lambda x: x[1])


def word_detector(img, debug_flag=False):
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 100))
    dilation = cv2.dilate(img, rect_kernel, iterations=1)
    contours, hierarchy = cv2.findContours(
        dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    words = []
    if debug_flag:
        debug_contour(img, contours)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        words.append((x, y, w, h))
    return sorted(words, key=lambda x: x[0])


def character_detector(img, debug_flag=False):
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 100))
    dilation = cv2.dilate(img, rect_kernel, iterations=1)
    contours, hierarchy = cv2.findContours(
        dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    chars = []
    if debug_flag:
        debug_contour(img, contours)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        chars.append((x, y, w, h))
    return sorted(chars, key=lambda x: x[0])


def preproc_img_color(img, resize_factor):
    height, width, _ = img.shape
    img = cv2.resize(img, (int(width*resize_factor),
                           int(height*resize_factor)), interpolation=cv2.INTER_AREA)
    img = cv2.rectangle(img, (780, 93), (1000, 200), (0, 0, 0), -1)
    return img


def thresholding(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw_img = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
    return bw_img


def char_img2arr(char):
    arr = cv2.resize(
        char, (28, 28), interpolation=cv2.INTER_AREA)
    arr = tf.expand_dims(arr, -1)
    return arr
