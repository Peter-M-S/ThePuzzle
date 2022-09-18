#
# class to handle frame on a 2d array/list
# points: set of coordinates inside frame
#
# Frame will move 1. left to right, 2. top to bottom, depending of actual top/left-piont


class Frame:

    def __init__(self, top_left, height_width):
        self.top, self.left = top_left              # index of first row, col
        self.height, self.width = height_width      # amount of rows cols
        self.points = self.get_points()


    @property
    def bottom(self):
        return self.top + self.height - 1           # index of last row

    @property
    def right(self):
        return self.left + self.width - 1           # index of last col

    @property
    def total(self):
        return len(self.points)

    @property
    def center(self):
        r = round(self.top + self.height/2)
        c = round(self.left + self.width/2)
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

    def move_frame_in_array(self, array, step):

        r_max = len(array[0]) - 1             # index of last column of array
        b_max = len(array) - 1                # index of last row of array

        if self.right == r_max:               # frame at right end?

            if self.bottom == b_max:          # frame at bottom end?
                return False                  # no move

            if self.bottom + step > b_max:    # if next step gets over b_max, shorten step
                step = b_max - self.bottom

            self.top = self.top + step        # move frame starting row step down
            self.left = 0
            self.points = self.get_points()   # make new point set starting at (new_t_row, 0)
            return self

        else:

            if self.right + step > r_max:     # if next step gets over b_max, shorten step
                step = r_max - self.right

            for r in range(self.top, self.top + self. height):    # move each row of frame step*1 col to right
                for s in range(step):
                    self.points.add((r, self.right + s + 1))      # add new right col
                    self.points.remove((r, self.left + s))        # delete left col
            self.left = self.left + step
            return self
