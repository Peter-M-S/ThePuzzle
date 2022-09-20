#
# class to handle frame on a 2d array/list
# points: set of coordinates (row, col) inside frame
# values: sub-array of values at coordinates inside frame
# Frame will move 1. left to right, 2. top to bottom, depending of actual top/left-piont

import numpy as np


class Frame:

    def __init__(self, top_left, height_width, array):
        self.top, self.left = top_left  # index of first row, col
        self.height, self.width = height_width  # amount of rows cols
        self.array = array
        self.array_rows, self.array_cols = self.array.shape[0], self.array.shape[1]
        self.array_points_rows = [[(i, j) for j in range(self.array_cols)] for i in range(self.array_rows)]
        self.array_points_cols = [[(j, i) for j in range(self.array_rows)] for i in range(self.array_cols)]

        self.points = self.get_points()

    @property
    def bottom(self):
        return self.top + self.height - 1  # index of last row

    @property
    def right(self):
        return self.left + self.width - 1  # index of last col

    @property
    def total(self):
        return len(self.points)

    @property
    def center(self):
        r = round(self.top + self.height / 2)
        c = round(self.left + self.width / 2)
        return r, c

    @property
    def start(self):
        return self.top, self.left

    @property
    def end(self):
        return self.bottom, self.right

    def get_points(self):
        points = set()
        for row in range(self.top, self.top + self.height):
            for col in range(self.left, self.left + self.width):
                points.add((row, col))
        return points

    def add_points_col(self, to_right=True):
        # add the points right of frame if to_right = True otherwise points left of frame
        self.width += 1
        if to_right:
            self.points.update(self.array_points_cols[self.right][self.top:self.top + self.height])
        else:
            self.left -= 1
            self.points.update(self.array_points_cols[self.left][self.top:self.top + self.height])

    def remove_points_col(self, from_left=True):
        # remove the points left in frame if from_left = True otherwise points right in frame
        self.width -= 1
        if from_left:
            self.points.difference_update(self.array_points_cols[self.left][self.top:self.top + self.height])
            self.left += 1
        else:
            self.points.difference_update(self.array_points_cols[self.right][self.top:self.top + self.height])

    def add_points_row(self, to_bottom=True):
        # add the points bottom of frame if to_bottom = True otherwise points top of frame
        self.height += 1
        if to_bottom:
            self.points.update(self.array_points_rows[self.bottom][self.left:self.left + self.width])
        else:
            self.top -= 1
            self.points.update(self.array_points_rows[self.top][self.left:self.left + self.width])

    def remove_points_row(self, from_top=True):
        # remove the points left in frame if from_left = True otherwise points right in frame
        self.height -= 1
        if from_top:
            self.points.difference_update(self.array_points_rows[self.top][self.left:self.left + self.width])
            self.top += 1
        else:
            self.points.difference_update(self.array_points_rows[self.bottom][self.left:self.left + self.width])

    def move_frame_in_array(self, array, step):

        c_max = len(array[0]) - 1  # index of last column of array
        r_max = len(array) - 1  # index of last row of array

        if self.right == c_max:  # frame at right end?

            if self.bottom == r_max:  # frame at bottom end?
                return False  # no move

            if self.bottom + step > r_max:  # if next step gets over b_max, shorten step
                step = r_max - self.bottom

            self.top = self.top + step  # move frame starting row step down
            self.left = 0
            self.points = self.get_points()  # make new point set starting at (new_t_row, 0)
            return self

        else:

            if self.right + step > c_max:  # if next step gets over b_max, shorten step
                step = c_max - self.right

            for r in range(self.top, self.top + self.height):  # move each row of frame step*1 col to right
                for s in range(step):
                    self.points.add((r, self.right + s + 1))  # add new right col
                    self.points.remove((r, self.left + s))  # delete left col
            self.left = self.left + step
            return self


if __name__ == '__main__':
    # Test unit for my_frames
    ROWS = 5
    COLS = 8
    a = np.zeros([ROWS, COLS])
    for i, r in enumerate(a):
        for j, c in enumerate(r):
            a[i, j] += i + 0.1 * j

    f = Frame((0, 0), (3, 3), a)
    print(f.array_rows)
    print(f.array_cols)
    print(f.array_points_rows)
    print(f.array_points_cols)
    print(f.points)
    f.add_points_col()
    f.remove_points_col()
    print(f.points)
    f.add_points_row()
    f.remove_points_row()
    print(f.points)
