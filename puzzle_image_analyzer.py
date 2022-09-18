import numpy as np
import math
import os
import matplotlib.pyplot as plt
import cv2.cv2 as cv2
import time

# import matplotlib
# matplotlib.use('Qt5Agg')

DRAW = True
# DRAW = False
SHOW = True
# SHOW = False
RESOLUTION = 1_000
SIDE_COLOR = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255), 3: (255, 255, 0), "V": (128, 128, 128)}
BOW_COLOR = {1: 255, -1: 0}
SIDES = [0, 1, 2, 3]  # index of edges top, right, down, left
EC_STD_VEC = {1: np.array([-RESOLUTION, 0]), -1: np.array([RESOLUTION, 0]), 0: np.array([RESOLUTION, 0])}


def file_type_list(path, ending=""):
    # returns a list of files of the "." + ending type in path
    # if ending = "", return all types?
    files_type = []
    for file in os.listdir(path):
        if file.endswith("." + ending):
            files_type.append(file)
    return files_type


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


def polygon_to_lines(polygon):
    polygon_1 = polygon[1:] + polygon[:1]
    lines = []
    for i, p in enumerate(polygon):
        lines.append([p, polygon_1[i]])
    return lines


def get_line_coordinates(line, img_bw):
    # line: points x,y
    line = np.int0(line)
    img_tmp = np.copy(img_bw)
    cv2.line(img_tmp, line[0], line[1], 128, 1)
    # plt.imshow(img_tmp, cmap="Greys_r")
    # plt.show()
    coords_yx = np.argwhere(img_tmp == 128)  # to use in np and img as row=y, column=x
    coords_xy = np.array([[x, y] for y, x in coords_yx])  # to use in points and cv2 as x, y
    return coords_xy, coords_yx


def normal_vector_ccw(p0, p1):
    # p0, p1: numpy arrays of start=0 and end=1 of line, positive x,y: right, down
    u_v = unit_vector(p0, p1)
    n_v_cw = np.array([u_v[1], -u_v[0]])
    return n_v_cw


def unit_vector(p0, p1):
    # p0, p1: numpy arrays of start=0 and end=1 of line
    vector = p1 - p0
    length = np.linalg.norm(vector)
    if not length:
        return False
    return vector / length


def find_max_dist_of_color(p0, p1, img_bw, vec, col):
    size_max = 0
    p2, p3 = p0, p1
    coordinates_xy, _ = get_line_coordinates([p0, p1], img_bw)
    for p in coordinates_xy:
        q1 = p
        q1_float = q1
        last_q1 = q1
        while img_bw[q1[1]][q1[0]] == col:  # coordinates of img need to be swapped x,y -> row=y, column=x
            q1_float = q1_float + vec
            q1 = np.int0(q1_float)
            last_q1 = q1
        if (size := np.linalg.norm(last_q1 - p)) > size_max:
            size_max = size
            p3 = last_q1
            p2 = p

    return p2, p3


def draw_edge_points_on_img_bw():
    # draw rgb-lines on bw_image
    img = cv2.cvtColor(img_bw, cv2.COLOR_GRAY2RGB)

    for e in ep:
        if edge_code:
            cv2.circle(img, e, 10, SIDE_COLOR[side], -1)
    for idx in range(0, len(ep), 2):
        thick = 1 if idx else 5
        cv2.line(img, ep[idx], ep[idx + 1], SIDE_COLOR[side], thick)
    cv2.line(img, inner_parallel[0], inner_parallel[1], SIDE_COLOR[side], 1)
    cv2.line(img, outer_parallel[0], outer_parallel[1], SIDE_COLOR[side], 1)

    for v in edge_vertices:
        cv2.line(img, ep[0], ep[0] + v, SIDE_COLOR["V"], 3)

    if SHOW:
        plt.imshow(img, cmap="Greys_r")
        plt.show()
    return img

def draw_corners_on_img(img, corners):
    # draw rgb-lines on bw_image
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    for c in corners:
            cv2.circle(img, *c, 10, SIDE_COLOR[side], -1)

    plt.imshow(img, cmap="Greys_r")
    plt.show()


def angle_to_side_vector(p0, p1, side_unit_vec):
    v1_u = unit_vector(p0, p1)
    return np.arccos(np.clip(np.dot(v1_u, side_unit_vec), -1.0, 1.0)), True


def rot_mat_2D(theta, clockwise):
    # theta in radiant
    # rotation anti-clockwise
    c, s = np.cos(theta), np.sin(theta)
    if clockwise:
        R = np.array(((c, -s), (s, c)))
    else:
        R = np.array(((c, s), (-s, c)))
    return R


if __name__ == '__main__':
    start_time = time.perf_counter()

    images = []
    pieces = []

    for f in file_type_list(".", "jpg"):
        print(f"file: {f}")
        pieces.append(f[:-4])

        img_raw = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        # images.append(img_raw)

        for side in SIDES:
            print(f"side: {side}")
            img = np.rot90(img_raw, k=side)  # CCW rotation
            corners = cv2.goodFeaturesToTrack(img, len(SIDES), 0.1, 600)
            corners = np.int0(corners)
            draw_corners_on_img(img, corners)
            # assert len(corners) == len(SIDES)
            continue
            img_bw = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            # keep img_bw clean for analysing

            poly = []
            for corner in corners:
                x, y = corner.ravel()
                poly.append([x, y])

            # correct order of poly:right,up,left,down
            poly = points_to_quad(poly)

            ep: list = [np.array(poly[2]), np.array(poly[1])]
            # print(f"ep: {ep}")

            edge_length = np.linalg.norm(ep[1] - ep[0])
            # print(edge_length)

            edge_normal_ccw = normal_vector_ccw(ep[0], ep[1])
            # print(f"edge normal ccw: {edge_normal_ccw}")

            # parallels (0.1*len), both times in positive x
            offset = edge_normal_ccw * edge_length * 0.1
            p0 = ep[0] + offset
            p1 = ep[1] + offset
            outer_parallel = np.int0([p0, p1])
            p0 = ep[0] - offset
            p1 = ep[1] - offset
            inner_parallel = np.int0([p0, p1])

            # check along inner/outer parallels if all (95%) same color
            _, coordinates_yx = get_line_coordinates(inner_parallel, img_bw)
            b = 0
            for p in coordinates_yx:
                if img_bw[p[0]][p[1]] == BOW_COLOR[-1]:
                    b += 1
            has_white = b / len(coordinates_yx) < 0.95
            inner_bow = has_white

            _, coordinates_yx = get_line_coordinates(outer_parallel, img_bw)
            w = 0
            for p in coordinates_yx:
                if img_bw[p[0]][p[1]] == BOW_COLOR[1]:
                    w += 1
            has_black = w / len(coordinates_yx) < 0.95
            outer_bow = has_black

            # print(f"inner, outer: {inner_bow}, {outer_bow}")
            edge_code = 0
            if inner_bow != outer_bow:
                edge_code = 1 if inner_bow else -1  # y-direction of bow
            print(f"edge code: {edge_code}")

            if edge_code == 1:
                ep[0], ep[1] = ep[1], ep[0]  # swap start and end of edge
            # print(f"ep: {ep}")

            # find tips of bows
            if edge_code:
                vec = edge_normal_ccw * edge_code * -1
                col = BOW_COLOR[edge_code]
                p2, p3 = find_max_dist_of_color(ep[0], ep[1], img_bw, vec, col)
                ep.extend([p2, p3])
            # print(f"edge_points: {len(ep)}")

            # find sides from tip of bows
            if edge_code:
                p3 = ep[3]  # point at tip of bow
                p2 = ep[2]
                # define stop point at 80% to avoid size_max outside neck of bow
                p_stop = p3 + (p2 - p3) * 0.8
                vec = normal_vector_ccw(p3, p2)
                col = BOW_COLOR[edge_code]
                p4, p5 = find_max_dist_of_color(p3, p_stop, img_bw, vec, col)
                p6, p7 = find_max_dist_of_color(p3, p_stop, img_bw, -vec, col)
                ep.extend([p4, p5, p6, p7])
            # print(f"edge_points: {len(ep)}")

            # create list of vertices
            if edge_code:
                # todo check to avoid hard-coded selection of points
                edge_vertices = [p - ep[0] for p in [ep[0], ep[1], ep[7], ep[3], ep[5]]]
            else:
                edge_vertices = [p - ep[0] for p in ep]
            print(f"vertices: {edge_vertices}")

            # get angle and rotation direction
            angle, clockwise = 0, True
            if edge_code:  # modify only when edge code != 0
                dx, dy = edge_vertices[1]
                cos_angle = dx / edge_length * -edge_code
                angle = np.arccos(cos_angle)
                # clockwise in x,y coordinates : right, down
                clockwise = dy < 0 if edge_code == -1 else not dy < 0
            # print(f"angel, clockwise: {angle}, {clockwise}")

            # rotate vertices by angle bool(clockwise)
            vertices_std = [edge_vertices[0], EC_STD_VEC[0]]
            if edge_code:  # modify only when edge code != 0
                rotation = rot_mat_2D(angle, clockwise)
                factor = RESOLUTION / edge_length
                vertices_std = [edge_vertices[0]]
                for v in edge_vertices[1:]:
                    v_std = rotation @ v
                    v_std *= factor
                    vertices_std.append(np.rint(v_std))
            print(f"vertices std: {vertices_std}")

            # convert vertices to edge code
            edge_code = str(edge_code)
            for p in vertices_std[2:]:
                for x in p:
                    edge_code = " ".join([edge_code, str(int(x))])

            print(f"edge_code: {edge_code}")
            pieces.append(edge_code)

            print()
            if DRAW:
                img = draw_edge_points_on_img_bw()
                images.append(img)

    with open("puzzle_pieces.txt", "w") as f:
        for i, line in enumerate(pieces):
            if i and i % 5 == 0:
                f.write("\n")
            f.write(line + "\n")

    end_time = time.perf_counter() - start_time
    print(end_time)

    # plot_to_pdf
    if DRAW:
        plot_to_pdf(images)
