import os

import cv2.cv2 as cv2
import numpy as np

feature_params = dict(maxCorners=4,
                      qualityLevel=0.2,
                      minDistance=400,
                      blockSize=50)


def file_type_list(path, ending=""):
    # returns a list of files of the "." + ending type in path
    # if ending = "", return all types?
    files_type = []
    for file in os.listdir(path):
        if file.endswith("." + ending.upper()) or file.endswith("." + ending.lower()):
            files_type.append(file)
    return files_type


def convert_to_list(corners):
    poly = []
    for corner in corners:
        x, y = corner.ravel()
        poly.append([int(x), int(y)])
    # correct order of poly:right,up,left,down
    poly = points_to_quad(poly)
    return poly


def points_to_quad(point_list):
    n = len(point_list)
    x_c, y_c = (sum(x[0] for x in point_list) / n, sum(y[1] for y in point_list) / n)
    ordered_point_list = point_list[:]
    for x, y in point_list:
        if x >= x_c and y > y_c:
            quad = 0
        elif x > x_c and y <= y_c:
            quad = 1
        elif x <= x_c and y < y_c:
            quad = 2
        else:
            quad = 3
        ordered_point_list[quad] = [x, y]
    return ordered_point_list


def find_corners(file_path_name):
    img_corners = cv2.imread(file_path_name)
    img_gray = cv2.cvtColor(img_corners, cv2.COLOR_BGR2GRAY)
    w, h = img_gray.shape
    feature_params['minDistance'] = min(w, h) * 0.4
    feature_params['blockSize'] = int(feature_params['minDistance'] * 0.02)
    corners = cv2.goodFeaturesToTrack(img_gray, **feature_params)
    if corners is not None:
        for x, y in np.int0(corners).reshape(-1, 2):
            cv2.circle(img_corners, (x, y), feature_params['blockSize'], (0, 255, 0), 10)
    corners = convert_to_list(corners)
    _, img_bw = cv2.threshold(img_gray, 64, 255, cv2.THRESH_BINARY)
    return corners, img_bw, img_corners
