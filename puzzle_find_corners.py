import cv2.cv2
import numpy as np
import math
import os
import matplotlib.pyplot as plt
import cv2.cv2 as cv2
import time
from my_frames import Frame


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

    for c in corners:
        cv2.circle(img, (c[1], c[0]), 40, (255, 0, 0), -1)

    for r in rims:
        cv2.rectangle(img, (r[0][1], r[0][0]), (r[1][1], r[1][0]), (0, 0, 255), 5)

    plt.imshow(img, cmap="Greys_r")
    plt.show()
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


def find_my_corners(img_bw):
    corners = []
    rims = []
    rows, cols = len(img_bw), len(img_bw[0])
    print(rows, cols)

    # clip img to no full white lines
    top = 0
    bottom = rows-1
    for i, r in enumerate(img_bw):
        if any(c == 0 for c in r):
            top = i
            break
    for i in range(1, len(img_bw)):
        if any(c == 0 for c in img_bw[-i]):
            bottom = rows - i
            break

    left = cols
    for r in img_bw[top:bottom+1]:
        for i, c in enumerate(r):
            if c == 0:
                left = min(i, left)
                break
    right = 0
    for r in img_bw[top:bottom+1]:
        for i in range(1, cols):
            if r[-i] == 0:
                right = max(cols - i, right)
                break

    # add half_frame to each side
    frame_cols, frame_rows = int(rows/5), int(cols/5)
    top = max(0, top - round(frame_rows/2))
    bottom = min(rows-1, bottom + round(frame_rows/2))
    left = max(0, left - round(frame_cols/2))
    right = min(cols-1, cols + round(frame_cols/2))
    print(top, bottom)
    print(left, right)
    red_img_bw = []
    for r in range(top, bottom+1):
        temp = []
        for c in range(left, right+1):
            temp.append(img_bw[r][c])
        red_img_bw.append(temp)

    print(len(red_img_bw), len(red_img_bw[0]))

    # plt.imshow(red_img_bw, cmap="Greys_r")
    # plt.show()

    # initialize frame
    top, left = 0, 0
    step = 100
    frame = Frame((top, left), (frame_rows, frame_cols))
    print(frame.total)
    while frame:
        # evaluate img_bw in frame
        # print(min(frame.points), max(frame.points))
        black = 0
        too_much_black = False
        for r, c in frame.points:
            # print(r, c)
            if not red_img_bw[r][c]:
                black += 1
            if black > (frame.total / 4) * 1.1:
                too_much_black = True
                break
        if (frame.total / 4) * 0.9 < black and not too_much_black:  # possible corner
            print(round(black/frame.total*100, 1))
            corners.append(frame.center)
            rims.append([(frame.top, frame.left), (frame.bottom, frame.right)])

        # move frame to next position
        frame = frame.move_frame_in_array(red_img_bw, step)

    return red_img_bw, corners, rims


if __name__ == '__main__':
    start_time = time.perf_counter()
    # PATH = "RV0781508/"
    PATH = "./"
    images = []

    # for f in file_type_list(PATH, "JPG"):
    for f in file_type_list(PATH, "jpg"):
        print(f"file: {f}")

        THRESHOLD = 25
        img_gray = cv2.imread(PATH + f, cv2.IMREAD_GRAYSCALE)
        img_bw = img_gray
        img_bw[img_gray <= THRESHOLD] = 0
        img_bw[img_gray > THRESHOLD] = 255

        # list_bw = img_bw.tolist()
        # plt.imshow(list_bw, cmap="Greys_r")
        # plt.show()
        red_img_bw, corners, rims = find_my_corners(img_bw)
        img_np = np.array(red_img_bw, dtype=np.uint8)
        # print(img_np.shape)
        img = draw_corners_on_img(img_np, corners, rims)
        # assert len(corners) == len(SIDES)
        # continue
        # img_bw = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        # keep img_bw clean for analysing
        images.append(img)

    plot_to_pdf(images)
    # assert len(corners) == len(SIDES)
