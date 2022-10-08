#
# main program:
# get list of all pictures (pieces have order and orientation, but are shuffled)
# for each picture:
#   get corners(img) -> list of 4 corners, img_gray, img_corners
#   get edgecodes(img_corners, img_gray, corners) -> list of edgecodes for each piece
# save pieces to disc -> pieces have same order same orientation, standardized edgecodes
#
# use solver(pieces) -> list of pieces with same order giving coordinates and new orientation
#


import numpy as np
import math
import os
import matplotlib.pyplot as plt
import cv2.cv2 as cv2
import time
import puzzle_utils as pu

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


def plot_to_pdf(images_list):
    plt.close("all")
    plt.figure(num=1, figsize=(9, 12))
    # plt.suptitle('Puzzle outlines {}'.format("not inverse" if INVERSE else "inverse"))

    plots = int(len(images_list) / 4)

    for i, img in enumerate(images_list):
        plt.subplot(plots, 4, i + 1)
        plt.imshow(img, cmap="binary")

    plt.savefig("puzzle_image_analysis.pdf")
    # plt.show()


# def polygon_to_lines(polygon):
#     polygon_1 = polygon[1:] + polygon[:1]
#     lines = []
#     for i, p in enumerate(polygon):
#         lines.append([p, polygon_1[i]])
#     return lines


def get_line_coordinates(line, img):
    # line: points x,y
    line = np.int0(line)
    img_tmp = img.astype(np.uint8)
    # print(f"shape: {img_tmp.shape} imgid: {id(img)} tmpid {id(img_tmp)} dtype {img_tmp.dtype}")
    cv2.line(img_tmp, line[0], line[1], 128, 1)
    # plt.imshow(img_tmp, cmap="Greys_r")
    # plt.show()
    coords_yx = np.argwhere(img_tmp == 128)  # to use in np and img as row=y, column=x
    coords_xy = np.array([[x, y] for y, x in coords_yx])  # to use in points and cv2 as x, y
    return coords_xy, coords_yx


def has_bow(img, inner, line):
    img_mask = np.zeros_like(img)
    cv2.line(img_mask, *line, 1, 1)
    line_total = np.count_nonzero(img_mask)
    img_test = img.copy()
    if inner:
        img_test = cv2.bitwise_not(img_test)
    img_bow = cv2.bitwise_and(img_test, img_mask)
    bow_total = np.count_nonzero(img_bow)
    return bow_total / line_total < 0.95


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


def draw_edge_points(edge, img_marked):
    # draw ep and lines

    for e in edge["points"]:
        if edge["type"]:
            cv2.circle(img_marked, e, 10, SIDE_COLOR[side], -1)
    for idx in range(0, len(edge["points"]), 2):
        thick = 1 if idx else 5
        cv2.line(img_marked, edge["points"][idx], edge["points"][idx + 1], SIDE_COLOR[side], thick)
    cv2.line(img_marked, edge["inner"][0], edge["inner"][1], SIDE_COLOR[side], 1)
    cv2.line(img_marked, edge["outer"][0], edge["outer"][1], SIDE_COLOR[side], 1)

    for v in edge["vertices"]:
        cv2.line(img_marked, edge["points"][0], edge["points"][0] + v, SIDE_COLOR["V"], 3)

    return img_marked


# def draw_corners_on_img(img, corners):
#     # draw rgb-lines on bw_image
#     img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
#
#     for c in corners:
#         cv2.circle(img, *c, 10, SIDE_COLOR[side], -1)
#
#     plt.imshow(img, cmap="Greys_r")
#     plt.show()


# def angle_to_side_vector(p0, p1, side_unit_vec):
#     v1_u = unit_vector(p0, p1)
#     return np.arccos(np.clip(np.dot(v1_u, side_unit_vec), -1.0, 1.0)), True


def rot_mat_2D(theta, clockwise):
    # theta in radiant
    # rotation anti-clockwise
    c, s = np.cos(theta), np.sin(theta)
    if clockwise:
        R = np.array(((c, -s), (s, c)))
    else:
        R = np.array(((c, s), (-s, c)))
    return R


def rotate_corners(p, ymax):
    # rotate CCW
    # x -> ymax - x
    # y -> x
    p_new = []
    for x, y in p:
        x_new = y
        y_new = ymax - x
        p_new.append([x_new, y_new])
    return p_new


def rotate_img_to_side(img_bw, corners, side):
    img_rot = np.rot90(img_bw, k=side)  # CCW rotation

    img_rot = np.ascontiguousarray(img_rot, dtype=np.uint8)

    corners_new = corners
    if side:
        # todo use correct img_rot or which?
        corners_new = rotate_corners(corners, img_rot.shape[1])
        # re-order corners: corner[0] = bottom_right
        corners_new = corners_new[-1:] + corners_new[:-1]

    return img_rot, corners_new


def get_edge_type(edge, img):
    ep0 = edge["points"][0]
    ep1 = edge["points"][1]
    edge["length"] = np.linalg.norm(ep1 - ep0)
    # print(edge_length)

    edge["normal_ccw"] = normal_vector_ccw(ep0, ep1)
    # print(f"edge normal ccw: {edge_normal_ccw}")

    # parallels (0.1*len), both times in positive x
    offset = edge["normal_ccw"] * edge["length"] * 0.1
    edge["outer"] = np.int0(edge["points"] + offset)
    edge["inner"] = np.int0(edge["points"] - offset)

    # check along inner/outer parallels if all (95%) same color
    inner_bow = has_bow(img, True, edge["inner"])
    outer_bow = has_bow(img, False, edge["outer"])

    # print(f"inner, outer: {inner_bow}, {outer_bow}")
    edge["type"] = 0
    if inner_bow != outer_bow:
        edge["type"] = 1 if inner_bow else -1  # y-direction of bow

    return edge


def get_edge_points(edge, img):
    if edge["type"] == 1:
        # swap start and end of edge
        edge["points"][0], edge["points"][1] = edge["points"][1], edge["points"][0]

    # find tips of bows
    if edge["type"]:
        vec = edge["normal_ccw"] * edge["type"] * -1
        col = BOW_COLOR[edge["type"]]
        p2, p3 = find_max_dist_of_color(edge["points"][0], edge["points"][1], img, vec, col)
        edge["points"].extend([p2, p3])
    # print(f"edge_points: {len(ep)}")

    # find sides from tip of bows
    if edge["type"]:
        p3 = edge["points"][3]  # point at tip of bow
        p2 = edge["points"][2]
        # define stop point at 80% to avoid size_max outside neck of bow
        p_stop = p3 + (p2 - p3) * 0.8
        vec = normal_vector_ccw(p3, p2)
        col = BOW_COLOR[edge["type"]]
        p4, p5 = find_max_dist_of_color(p3, p_stop, img, vec, col)
        p6, p7 = find_max_dist_of_color(p3, p_stop, img, -vec, col)
        edge["points"].extend([p4, p5, p6, p7])
    # print(f"edge_points: {len(ep)}")
    return edge


def get_edge_vertices(edge):
    # create list of vertices
    if edge["type"]:
        # todo check to avoid hard-coded selection of points
        edge["vertices"] = [p - edge["points"][0] for p in [edge["points"][a] for a in [0, 1, 7, 3, 5]]]
    else:
        edge["vertices"] = [p - edge["points"][0] for p in edge["points"]]

    return edge


def get_vertices_std(edge):
    # get angle and rotation direction
    angle, clockwise = 0, True
    if edge["type"]:  # modify only when edge code != 0
        dx, dy = edge["vertices"][1]
        cos_angle = dx / edge["length"] * edge["type"] * -1
        angle = np.arccos(cos_angle)
        # clockwise in x,y coordinates : right, down
        clockwise = dy < 0 if edge["type"] == -1 else not dy < 0
    # print(f"angel, clockwise: {angle}, {clockwise}")

    # rotate vertices by angle bool(clockwise)
    edge["vertices_std"] = [edge["vertices"][0], EC_STD_VEC[0]]
    if edge["type"]:  # modify only when edge code != 0
        rotation = rot_mat_2D(angle, clockwise)
        factor = RESOLUTION / edge["length"]
        edge["vertices_std"] = [edge["vertices"][0]]
        for v in edge["vertices"][1:]:
            v_std = rotation @ v
            v_std *= factor
            edge["vertices_std"].append(np.rint(v_std))

    return edge


def show_corners(img, corners):
    img_corners = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for idx, c in enumerate(corners):
        cv2.circle(img_corners, c, 50, SIDE_COLOR[idx], -1)
    plt.imshow(img_corners, cmap="Greys_r")
    plt.show()


if __name__ == '__main__':
    start_time = time.perf_counter()

    PATH = "RV0781508/"
    # PATH = "./"
    images = []
    pieces = []

    for f_idx, f in enumerate(sorted(pu.file_type_list(PATH, "jpg"), reverse=True)):
        print(f"Piece #{f_idx} file: {f}")
        pieces.append(" ".join([str(f_idx), f[:-4]]))

        corners, img_bw, img_marked = pu.find_corners(PATH + f)
        if SHOW:
            show_corners(img_bw, corners)

        for side in SIDES:
            print(f"side: {side}")

            img, corners = rotate_img_to_side(img_bw, corners, side)
            if side:
                img_marked = np.rot90(img_marked)
                img_marked = np.ascontiguousarray(img_marked, dtype=np.uint8)

            # Edge points = top left  and top right corner
            edge = {"points": [np.array(corners[2]), np.array(corners[1])]}

            edge = get_edge_type(edge, img)

            edge = get_edge_points(edge, img)

            edge = get_edge_vertices(edge)

            edge = get_vertices_std(edge)

            # convert edge type and vertices_std to edge code
            edge["code"] = str(edge["type"])
            for p in edge["vertices_std"][2:]:
                for x in p:
                    edge["code"] = " ".join([edge["code"], str(int(x))])

            print(f"edge_code: {edge['code']}")
            pieces.append(edge["code"])

            print()
            if DRAW:
                img_marked = draw_edge_points(edge, img_marked)

        img_marked = np.rot90(img_marked)
        img_marked = np.ascontiguousarray(img_marked, dtype=np.uint8)
        if SHOW:
            plt.imshow(img_marked, cmap="Greys_r")
            plt.show()
        images.append(img_marked)

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
