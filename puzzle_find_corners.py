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
    # return img


def plot_to_pdf(images_list):
    plt.close("all")
    plt.figure(num=1, figsize=(9, 12))
    # plt.suptitle('Puzzle outlines {}'.format("not inverse" if INVERSE else "inverse"))

    plots = int(len(images_list) / 2)

    for i, img in enumerate(images_list):
        plt.subplot(plots, 2, i + 1)
        plt.imshow(img, cmap="binary")

    plt.savefig("puzzle_image_analysis.pdf")
    # plt.show()


def find_my_corners(img_bw):
    corners = []
    rims = []
    rows, cols = len(img_bw), len(img_bw[0])
    print(rows, cols)

    # clip img to no full white lines
    top = 0
    bottom = rows
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
    top -= round(frame_rows/2)
    bottom += round(frame_rows/2)
    left -= int(frame_cols/2)
    right += int(frame_cols/2)
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

    print(corners)
    print(rims)
    img_np = np.array(red_img_bw, dtype=np.uint8)
    print(img_np.shape)
    draw_corners_on_img(img_np, corners, rims)
    exit()

    half_a = 50
    pixels_tot = (half_a * 2) ** 2
    pixels_min = pixels_tot / 4 * 0.8
    pixels_max = pixels_tot / 4 * 1.2
    for row in range(half_a, rows - 2 * half_a, half_a):
        for col in range(half_a, cols - 2 * half_a, half_a):
            if img[row][col]:
                continue
            print(row, col)
            black = set()
            is_black_quarter = False
            for x in range(row - half_a, row + 2 * half_a, 1):
                if len(black) > pixels_max:
                    break
                for y in range(col - half_a, col + 2 * half_a, 1):
                    if len(black) > pixels_max:
                        break
                    if not img[x][y]:
                        black.add((x, y))
            if pixels_min < len(black):
                is_black_quarter = True

            if is_black_quarter:
                cluster = 0
                for x, y in black:
                    white_neighbours = 0
                    for xx in range(x - 1, x + 2, 1):
                        if white_neighbours:
                            break
                        for yy in range(y - 1, y + 2, 1):
                            if white_neighbours:
                                break
                            if (xx, yy) not in black:
                                white_neighbours += 1
                    if white_neighbours == 9:
                        cluster += 1

                if cluster >= (len(black) - (2 * half_a)) * 0.9:
                    corners.append(np.array([row, col]))

    return np.array(corners)


if __name__ == '__main__':
    start_time = time.perf_counter()
    # PATH = "RV0781508/"
    PATH = "./"
    images = []

    # for f in file_type_list(PATH, "JPG"):
    for f in file_type_list(PATH, "jpg"):
        print(f"file: {f}")

        # img = cv2.imread(PATH + f)
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #
        # gray = np.float32(gray)
        # dst = cv2.cornerHarris(gray, 2, 3, 0.1)
        # # result is dilated for marking the corners, not important
        # dst = cv2.dilate(dst, None)
        # # Threshold for an optimal value, it may vary depending on the image.
        # img[dst > 0.01 * dst.max()] = [0, 255, 0]
        # cv2.imshow('dst', img)
        # if cv2.waitKey(0) & 0xff == 27:
        #     cv2.destroyAllWindows()

        THRESHOLD = 25
        img_gray = cv2.imread(PATH + f, cv2.IMREAD_GRAYSCALE)
        img_bw = img_gray
        img_bw[img_gray <= THRESHOLD] = 0
        img_bw[img_gray > THRESHOLD] = 255

        list_bw = img_bw.tolist()
        # plt.imshow(list_bw, cmap="Greys_r")
        # plt.show()
        corners = find_my_corners(list_bw)
        print(corners)
        exit()
        # corners = cv2.goodFeaturesToTrack(img, 16, 0.00001, w * 0.2)
        # corners = np.int0(corners)
        draw_corners_on_img(img, corners)
        # assert len(corners) == len(SIDES)
        # continue
        # img_bw = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        # keep img_bw clean for analysing
        images.append(img)

    # plot_to_pdf(images)
    # assert len(corners) == len(SIDES)
