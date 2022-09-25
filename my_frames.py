#
# class to handle frame on a 2d array/list
# points: set of coordinates (row, col) inside frame
# values: sub-array of values at coordinates inside frame
# Frame will move 1. left to right, 2. top to bottom, depending of actual top/left-piont

import numpy as np


class Frame:

    def __init__(self, top_left, height_width, array, use_points=False, use_values=True):
        self.top, self.left = top_left  # index of first row, col
        self.height, self.width = height_width  # amount of rows cols
        self.array = array
        self.use_points = use_points
        self.use_values = use_values
        self.array_rows, self.array_cols = self.array.shape[0], self.array.shape[1]
        self.array_points_rows = [[(i, j) for j in range(self.array_cols)]
                                  for i in range(self.array_rows)] if self.use_points else False
        self.array_points_cols = [[(j, i) for j in range(self.array_rows)]
                                  for i in range(self.array_cols)] if self.use_points else False
        self.to_right = True
        self.to_bottom = True
        self.points = self.get_points() if self.use_points else False
        self.values = self.get_values() if self.use_values else False

    @property
    def bottom(self):
        return self.top + self.height - 1  # index of last row

    @property
    def right(self):
        return self.left + self.width - 1  # index of last col

    @property
    def total(self):
        return self.width * self.height

    @property
    def center(self):
        r = round(self.top + self.height / 2)
        c = round(self.left + self.width / 2)
        return r, c

    @property
    def quadrant(self):
        _top = 0 if self.center[0] <= self.array_rows/2 else 1
        _left = 0 if self.center[1] <= self.array_cols/2 else 1
        return (_top * 2) + _left

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

    def get_values(self):
        values = self.array[self.top:self.top + self.height, self.left:self.left + self.width]
        return values

    def add_col(self):
        # add the point and/or values right of frame if to_right = True otherwise point and/or values left of frame
        self.width += 1
        if self.to_right:
            if self.right > self.array_cols - 1:
                return False
            if self.use_points:
                self.points.update(self.array_points_cols[self.right][self.top:self.top + self.height])
            if self.use_values:
                new_col = self.array[self.top:self.top + self.height, self.right]
                self.values = np.c_[self.values, new_col]
        else:
            if self.left <= 0:
                return False
            self.left -= 1
            if self.use_points:
                self.points.update(self.array_points_cols[self.left][self.top:self.top + self.height])
            if self.use_values:
                new_col = self.array[self.top:self.top + self.height, self.left]
                self.values = np.c_[new_col, self.values]

    def remove_col(self):
        # remove the point and/or values left in frame if to_right = True otherwise point and/or values right in frame
        if self.to_right:
            if self.use_points:
                self.points.difference_update(self.array_points_cols[self.left][self.top:self.top + self.height])
            if self.use_values:
                self.values = np.delete(self.values, 0, 1)
            self.left += 1
        else:
            if self.use_points:
                self.points.difference_update(self.array_points_cols[self.right][self.top:self.top + self.height])
            if self.use_values:
                self.values = np.delete(self.values, -1, 1)
        self.width -= 1

    def add_row(self):
        # add the point and/or values bottom of frame if to_bottom = True otherwise point and/or values top of frame
        self.height += 1
        if self.to_bottom:
            if self.bottom > self.array_rows - 1:
                return False
            if self.use_points:
                self.points.update(self.array_points_rows[self.bottom][self.left:self.left + self.width])
            if self.use_values:
                new_row = self.array[self.bottom, self.left:self.left + self.width]
                self.values = np.r_[self.values, [new_row]]
        else:
            if self.top <= 0:
                return False
            self.top -= 1
            if self.use_points:
                self.points.update(self.array_points_rows[self.top][self.left:self.left + self.width])
            if self.use_values:
                new_row = self.array[self.top, self.left:self.left + self.width]
                self.values = np.r_[[new_row], self.values]

    def remove_row(self):
        # remove the point and/or values top in frame if to_bottom = True otherwise point and/or values bottom in frame
        self.height -= 1
        if self.to_bottom:
            if self.use_points:
                self.points.difference_update(self.array_points_rows[self.top][self.left:self.left + self.width])
            if self.use_values:
                self.values = np.delete(self.values, 0, 0)
            self.top += 1
        else:
            if self.use_points:
                self.points.difference_update(self.array_points_rows[self.bottom][self.left:self.left + self.width])
            if self.use_values:
                self.values = np.delete(self.values, -1, 0)

    # def add_points_col(self, to_right=True):
    #     # add the points right of frame if to_right = True otherwise points left of frame
    #     self.width += 1
    #     if to_right:
    #         self.points.update(self.array_points_cols[self.right][self.top:self.top + self.height])
    #     else:
    #         self.left -= 1
    #         self.points.update(self.array_points_cols[self.left][self.top:self.top + self.height])

    # def remove_points_col(self, to_right=True):
    #     # remove the points left in frame if to_right = True otherwise points right in frame
    #     if to_right:
    #         self.points.difference_update(self.array_points_cols[self.left][self.top:self.top + self.height])
    #         self.left += 1
    #     else:
    #         self.points.difference_update(self.array_points_cols[self.right][self.top:self.top + self.height])
    #     self.width -= 1

    # def add_points_row(self, to_bottom=True):
    #     # add the points bottom of frame if to_bottom = True otherwise points top of frame
    #     self.height += 1
    #     if to_bottom:
    #         self.points.update(self.array_points_rows[self.bottom][self.left:self.left + self.width])
    #     else:
    #         self.top -= 1
    #         self.points.update(self.array_points_rows[self.top][self.left:self.left + self.width])

    # def remove_points_row(self, from_top=True):
    #     # remove the points left in frame if from_left = True otherwise points right in frame
    #     self.height -= 1
    #     if from_top:
    #         self.points.difference_update(self.array_points_rows[self.top][self.left:self.left + self.width])
    #         self.top += 1
    #     else:
    #         self.points.difference_update(self.array_points_rows[self.bottom][self.left:self.left + self.width])

    # def move_frame_in_array(self, array, step):
    #
    #     c_max = len(array[0]) - 1  # index of last column of array
    #     r_max = len(array) - 1  # index of last row of array
    #
    #     if self.right == c_max:  # frame at right end?
    #
    #         if self.bottom == r_max:  # frame at bottom end?
    #             return False  # no move
    #
    #         if self.bottom + step > r_max:  # if next step gets over b_max, shorten step
    #             step = r_max - self.bottom
    #
    #         self.top = self.top + step  # move frame starting row step down
    #         self.left = 0
    #         self.points = self.get_points()  # make new point set starting at (new_t_row, 0)
    #         return self
    #
    #     else:
    #
    #         if self.right + step > c_max:  # if next step gets over b_max, shorten step
    #             step = c_max - self.right
    #
    #         for r in range(self.top, self.top + self.height):  # move each row of frame step*1 col to right
    #             for s in range(step):
    #                 self.points.add((r, self.right + s + 1))  # add new right col
    #                 self.points.remove((r, self.left + s))  # delete left col
    #         self.left = self.left + step
    #         return self

    def snake_frame(self, step):
        # "snake" through array to keep most of the points same and change only step rows or cols per step
        if (self.to_right and self.right < self.array_cols - 1) or (not self.to_right and self.left > 0):
            self.move_sideways(step)
            return self
        elif self.bottom < self.array_rows - 1:
            self.move_down(step)
            return self
        else:
            return False

    def move_sideways(self, step):
        this_step = min(step, self.array_cols - 1 - self.right) if self.to_right else min(step, self.left)
        # print(f"moving sideways {this_step}")
        for s in range(int(this_step)):
            self.add_col()
            self.remove_col()

    def move_down(self, step):
        this_step = min(step, self.array_rows - 1 - self.bottom)
        # print(f"moving down {this_step}")
        for s in range(int(this_step)):
            self.add_row()
            self.remove_row()
        self.to_right = not self.to_right  # invert the sideways direction

    def corners_dict(self, corner_fraction):
        # list with 4 sub arrays of values of width/height*corner_fraction,
        _width, _height = int(self.width * corner_fraction + .5), int(self.height * corner_fraction + .5)
        corners = {0b00: self.values[0:_height, 0:_width],
                   0b01: self.values[0:_height, self.width - _width:self.width],
                   0b11: self.values[self.height - _height:self.height, self.width - _width:self.width],
                   0b10: self.values[self.height - _height:self.height, 0:_width]}
        return corners

    def mid_frame(self, mid_fraction):
        _width, _height = int(self.width * mid_fraction / 2 + .5), int(self.height * mid_fraction / 2 + .5)
        mid = self.values[self.center[0]-_width:self.center[0]+_width, self.center[1]-_height:self.center[1]+_height]
        return mid


if __name__ == '__main__':
    # Test unit for my_frames
    ROWS = 20
    COLS = 25
    a = np.zeros([ROWS, COLS])
    for i, r in enumerate(a):
        for j, c in enumerate(r):
            a[i, j] = i + 0.01 * j

    f = Frame((0, 0), (8, 8), a, use_points=False, use_values=True)

    print(a)

    # print(f.points)
    # print(f.values)
    #
    # f.add_col()
    # print(f.points)
    # print(f.values)
    #
    # f.remove_col()
    # print(f.points)
    # print(f.values)
    #
    # f.add_row()
    # print(f.points)
    # print(f.values)
    #
    # f.remove_row()
    # print(f.points)
    # print(f.values)
    #
    # f.to_right = False
    # f.to_bottom = False
    #
    # f.add_col()
    # print(f.points)
    # print(f.values)
    #
    # f.remove_col()
    # print(f.points)
    # print(f.values)
    #
    # f.add_row()
    # print(f.points)
    # print(f.values)
    #
    # f.remove_row()
    # print(f.points)
    # print(f.values)
    #
    # f.to_right = True
    # f.to_bottom = True

    # print(len(f.points))
    print(f.start, f.end)
    print(f.values)

    print("snake_frame")

    while f.snake_frame(100):
        # print(len(f.points))
        # print(f.points)
        print(f.start, f.end)
        print(f.values)
        for i in range(4):
            print(f.get_corners(0.25)[i])
