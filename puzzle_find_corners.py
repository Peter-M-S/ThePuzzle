import cv2.cv2
import numpy as np
import math
import os
import matplotlib.pyplot as plt
import cv2.cv2 as cv2
import time
from my_frames import Frame


# SHOW = True
SHOW = False

THRESHOLD = 25
PADDING = 0.1
FRAMEFACTOR = 1/6
STEPFACTOR = 1/5
CORNERFACTOR_MAX = 0.25 * 1.1
CORNERFACTOR_MIN = 0.25 * 0.9
BLACK = 0
WHITE = 255


def file_type_list(path, ending=""):
    # returns a list of files of the "." + ending type in path
    # if ending = "", return all types?
    # todo ensure lower upper case
    files_type = []
    for file in os.listdir(path):
        if file.endswith("." + ending):
            files_type.append(file)
    return files_type


def draw_corners_on_img(img, corners, rims):
    # draw rgb-lines on bw_image
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    w, h, d = img.shape
    radius = round(min(w, h)/35)

    for c in corners:
        cv2.circle(img, (c[1], c[0]), radius, (255, 0, 0), -1)

    for r in rims:
        cv2.rectangle(img, (r[0][1], r[0][0]), (r[1][1], r[1][0]), (0, 0, 255), int(radius/10))

    return img


def plot_to_pdf(images_list):
    plt.close("all")
    plt.figure(num=1, figsize=(9, 12))
    # plt.suptitle('Puzzle outlines {}'.format("not inverse" if INVERSE else "inverse"))

    plots = int(len(images_list) / 2)

    for i, img in enumerate(images_list):
        plt.subplot(plots, 2, i + 1)
        plt.imshow(img, cmap="Greys_r")

    plt.savefig("puzzle_image_analysis.pdf")
    # plt.show()


def find_my_corners(img_bw_clip):
    corners = []
    rims = []
    rows, cols = img_bw_clip.shape

    # initialize frame
    top, left = 0, 0
    frame_size = min(round(rows * FRAMEFACTOR), round(cols * FRAMEFACTOR))
    step = round(frame_size * STEPFACTOR)

    frame = Frame((top, left), (frame_size, frame_size))

    blacks_max = frame.total * CORNERFACTOR_MAX
    blacks_min = frame.total * CORNERFACTOR_MIN

    while frame:
        # evaluate img_bw in frame
        blacks = 0
        too_much_blacks = False
        for r, c in frame.points:
            if img_bw_clip[r][c] == BLACK:
                blacks += 1
            if blacks > blacks_max:
                too_much_blacks = True
                break
        if blacks_min < blacks and not too_much_blacks:  # possible corner
            # print(round(blacks / frame.total * 100, 1))
            corners.append(frame.center)
            rims.append([frame.start, frame.end])

        # move frame to next position
        frame = frame.move_frame_in_array(img_bw_clip, step)

    return corners, rims


def read_file_to_bw(filepath, threshold=THRESHOLD):
    img = cv2.imread(filepath)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_bw = img_gray
    img_bw[img_gray <= threshold] = BLACK
    img_bw[img_gray > threshold] = WHITE
    return img_bw


def clip_to_padding(img_bw):
    rows, cols = img_bw.shape
    # print(img_bw.shape)

    # clip img_bw near to black
    top = 0
    bottom = rows - 1
    for i, r in enumerate(img_bw):
        if any(c == BLACK for c in r):
            top = i
            break
    for i in range(1, rows):
        if any(c == BLACK for c in img_bw[-i]):
            bottom = rows - i
            break

    left = cols
    for r in img_bw[top:bottom + 1]:
        for i, c in enumerate(r):
            if c == BLACK:
                left = min(i, left)
                break
    right = 0
    for r in img_bw[top:bottom + 1]:
        for i in range(1, cols):
            if r[-i] == BLACK:
                right = max(cols - i, right)
                break

    padding_rows, padding_cols = round(rows * PADDING), round(cols * PADDING)
    top = max(0, top - padding_rows)
    bottom = min(rows - 1, bottom + padding_rows)
    left = max(0, left - padding_cols)
    right = min(cols - 1, right + padding_cols)

    red_img_bw = img_bw[top:bottom + 1, left:right + 1]
    
    return red_img_bw


if __name__ == '__main__':
    start_time = time.perf_counter()
    # PATH = "RV0781508/"
    PATH = "./"
    images = []

    # for f in file_type_list(PATH, "JPG"):
    for f in file_type_list(PATH, "jpg"):
        print(f"file: {f}")

        img_bw = read_file_to_bw(PATH + f)
        if SHOW:
            plt.imshow(img_bw, cmap="Greys_r")
            plt.show()
            
        img_bw_clip = clip_to_padding(img_bw)
        if SHOW:
            plt.imshow(img_bw_clip, cmap="Greys_r")
            plt.show()

        corners, rims = find_my_corners(img_bw_clip)

        img = draw_corners_on_img(img_bw_clip, corners, rims)
        if SHOW:
            plt.imshow(img, cmap="Greys_r")
            plt.show()

        images.append(img)

    print(time.perf_counter()-start_time)
    plot_to_pdf(images)
