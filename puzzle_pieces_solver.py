import numpy as np
import matplotlib.pyplot as plt
import cv2.cv2 as cv2

SIDES = [0, 1, 2, 3]


def unbox_pieces(text_file):
    pieces = {}
    with open(text_file) as f:
        lines = f.readlines()
        n = len(lines)
        for i in range(0, n, 6):
            key, name = lines[i].split()
            e_c = []
            for j in range(1, 5):
                e_c.append(np.array([int(x) for x in lines[i + j].split()]))

            pieces[int(i / 6)] = (Piece(int(key), name, e_c))

    return pieces


# def is_match(piece1, s, piece2, s_opp, tolerance=10):
#     e1 = piece1.edge_codes[s]
#     e2 = piece2.edge_codes[s_opp]
#     if e1[0] * e2[0] >= 0:
#         return False
#     # only -1, 1 or 1, -1
#     squares = np.square(e1 + e2)
#     d = sum(squares)
#     print(d)
#     result = d <= tolerance
#     return result


def is_valid_next_position(s, position, board):
    row, col = position
    if s == 0:
        row -= 1
    elif s == 1:
        col += 1
    elif s == 2:
        row += 1
    else:
        col -= 1

    if row < 0 or col < 0 or row > len(board) or row > len(board[0]):
        print("matching but out of board")
        return False
    if board[row][col] is not None:
        # print("matching, but board not free")
        return False
    return [row, col]


def get_neighbors(pos, board):
    neighbors = [False, False, False, False]
    l_board = len(board[0])
    h_board = len(board)
    # top of position s = 0, s_neighbor = 2
    n_row, n_col = pos[:]
    if n_row == 0:
        neighbors[0] = "rim"
    if n_row - 1 >= 0:
        n_row -= 1
        if board[n_row][n_col] is not None:
            neighbors[0] = board[n_row][n_col].edge_codes[2]

    # right of position s = 1, s_neighbor = 3
    n_row, n_col = pos[:]
    if n_col == l_board - 1:
        neighbors[1] = "rim"
    if n_col + 1 < l_board:
        n_col += 1
        if board[n_row][n_col] is not None:
            neighbors[1] = board[n_row][n_col].edge_codes[3]

    # bottom of position s = 2, s_neighbor = 0
    n_row, n_col = pos[:]
    if n_row == h_board - 1:
        neighbors[2] = "rim"
    if n_row + 1 < h_board:
        n_row += 1
        if board[n_row][n_col] is not None:
            neighbors[2] = board[n_row][n_col].edge_codes[0]

    # left of position s = 3, s_neighbor = 1
    n_row, n_col = pos[:]
    if n_col == 0:
        neighbors[3] = "rim"
    if n_col - 1 >= 0:
        n_col -= 1
        if board[n_row][n_col] is not None:
            neighbors[3] = board[n_row][n_col].edge_codes[1]

    return neighbors


def delta_edge_code(ec1, ec2):
    if ec1[0] * ec2[0] >= 0:
        return 100_000
    squares = np.square(ec1 + ec2)
    d = sum(squares)
    return d


def display_board(board):
    for row in board:
        for p in row:
            if p is not None:
                print("          {:2}           ".format(p.edge_codes[0][0]), end="")
            else:
                print("                       ", end="")
        print("|", end="\n")
        for p in row:
            if p is not None:
                print(" {:2} | {:8} {} | {:2} |".format(p.edge_codes[3][0], p.name, p.rotation, p.edge_codes[1][0]),
                      end="")
            else:
                print("    |            |    |", end="")
        print("|", end="\n")
        for p in row:
            if p is not None:
                print("          {:2}           ".format(p.edge_codes[2][0]), end="")
            else:
                print("                       ", end="")
        print("|", end="\n")

    print()


def get_candidates(neighbors, remaining_pieces):
    candidates = []
    for piece_id, p in remaining_pieces.items():
        total_delta = 1_000_000
        cand = [total_delta, p.id, p.rotation]
        for _ in [1, 2, 3, 0]:
            p.rotate()
            total_rot = 0
            for s_n, n in enumerate(neighbors):
                if n is not False:
                    if type(n) == str and n == "rim":
                        if p.edge_codes[s_n][0] != 0:
                            total_rot += 100_000
                    else:
                        ecp = p.edge_codes[s_n]
                        total_rot += delta_edge_code(ecp, n)

            if total_rot < total_delta:
                cand[0] = total_rot
                cand[2] = p.rotation
                total_delta = total_rot

        candidates.append(cand)

    return candidates


def delta_match(p1, p2):
    delta = 0
    for i in [1, 3, 5]:
        x1, y1 = p1[i], p1[i + 1]
        x2, y2 = p2[i], p2[i + 1]
        delta += ((x2 + x1) ** 2 + (y2 + y1) ** 2) ** 0.5
    return delta


def show_delta_match(p1, p2):
    img = np.ones([500, 1200, 3])
    delta = 0
    for i in [1, 3, 5]:
        x1, y1 = abs(p1[i]), abs(p1[i + 1])
        cv2.circle(img, [x1, y1], 20, (1, 0, 0), 5)
        x2, y2 = abs(p2[i]), abs(p2[i + 1])
        cv2.circle(img, [x2, y2], 20, (0, 1, 0), 5)
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), 5)
        delta += ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    plt.imshow(img, cmap="Greys_r")
    plt.show()
    return delta


def dfs_puzzle(solution, used, remaining_pieces, board, position):
    p1 = solution[-1]
    # solution found?
    if len(remaining_pieces) == 0:
        return solution

    # iterate through all sides of the last piece
    for s1 in SIDES:
        # skip outside edges
        if p1.rims[s1]:
            continue

        # next position
        next_position = is_valid_next_position(s1, position, board)
        if not next_position:
            continue

        # neighbors = get_neighbors(next_position, board)
        # print(neighbors)

        # get candidates as list of (total_tolerances, piece_id ,matching piece side)

        c = p1.candidates[s1][0]
        # for c in candidates:
        p2 = remaining_pieces[c[1]]
        matching_side = c[2]

        if s1 == 2:
            rotations = matching_side
        elif s1 == 1:
            rotations = next_side(matching_side)
        elif s1 == 0:
            rotations = next_side(next_side(matching_side))
        else:
            rotations = prev_side(matching_side)

        for i in range(rotations):
            p2.rotate()
        print(" {} > {} < {} ".format(p1.id, c[0], p2.id))
        # delta_match(p1, s1, p2)

        board[next_position[0]][next_position[1]] = p2

        remaining_pieces.pop(p2.id)

        # display_board(board)

        # add test-piece to solution and step into next level
        if intermediate_solution := dfs_puzzle(solution + [p2], used.union({p2.id}), remaining_pieces, board,
                                               next_position):
            return intermediate_solution


def next_side(s):
    return s + 1 if s < SIDES[-1] else 0


def prev_side(s):
    return s - 1 if s > 0 else SIDES[-1]


class Piece:
    max_tolerance = 1000

    def __init__(self, piece_id, name, edge_codes, rotation=0):
        self.id: int = piece_id
        self.name: str = name
        self.edge_codes: list = edge_codes
        self.rotation: int = rotation
        self.rims = [len(e) == 1 for e in self.edge_codes]
        self.is_edge = any(self.rims)
        self.is_corner = sum(self.rims) == 2
        # , self.is_corner = self.check_edges()
        # self.neighbors = [False, False, False, False]
        self.candidates = [not a for a in self.rims]

    def rotate(self):
        # 1 step ccw
        self.edge_codes = self.edge_codes[1:] + self.edge_codes[:1]
        self.rims = self.rims[1:] + self.rims[:1]
        self.candidates = self.candidates[1:] + self.candidates[:1]
        self.rotation = next_side(self.rotation)

    # def check_edges(self):
    #     c = len([e for e in self.edge_codes if len(e) == 1])
    #     if c == 2:
    #         return True, True
    #     if c > 0:
    #         return True, False
    #     return False, False

    def get_candidates(self, pieces):
        for s1 in SIDES:
            if self.rims[s1]:
                continue
            candidates = []
            next_pos_is_rim = self.rims[prev_side(s1)] or self.rims[next_side(s1)]
            for key, p in pieces.items():
                if p == self:
                    continue
                if self.is_corner and not p.is_edge:
                    continue
                if next_pos_is_rim and not p.is_edge:
                    continue
                candidate = (100_000, key, 0)
                delta_piece = self.max_tolerance
                for s2 in SIDES:
                    if p.rims[s2]:
                        continue
                    if self.edge_codes[s1][0] * p.edge_codes[s2][0] >= 0:
                        continue
                    # check for same side rims
                    if next_pos_is_rim:
                        if self.rims[prev_side(s1)] and not p.rims[next_side(s2)]:
                            continue
                        if self.rims[next_side(s1)] and not p.rims[prev_side(s2)]:
                            continue
                    ec1 = self.edge_codes[s1]
                    ec2 = p.edge_codes[s2]
                    delta_s = delta_match(ec1, ec2)
                    if delta_s < delta_piece:
                        delta_piece = delta_s
                        candidate = (round(delta_piece, 3), p.id, s2)
                candidates.append(candidate)
            self.candidates[s1] = sorted(candidates)


def edge_code_checker(pieces_in_box):
    delta = 1000
    best_match = [delta, "img0", 0, "img1", 0]
    for i in range(23):
        p1 = pieces_in_box[i]
        for s1 in [0, 1, 2, 3]:
            ec1 = p1.edge_codes[s1]
            if ec1[0] == 0:
                continue
            for j in range(i + 1, 24):
                p2 = pieces_in_box[j]
                for s2 in [0, 1, 2, 3]:
                    ec2 = p2.edge_codes[s2]
                    if ec2[0] == 0:
                        continue
                    delta_i = delta_match(ec1, ec2)
                    if delta_i < 42:
                        print(delta_i, p1.name, s1, p2.name, s2)
                    if delta_i < delta:
                        best_match = [delta_i, p1.name, s1, p2.name, s2]
                        delta = delta_i
    print(best_match)


def main(puzzle_file):
    pieces_in_box = unbox_pieces(puzzle_file)
    print("from here on program has to find solution...")
    print("number of pieces: ", len(pieces_in_box))

    for key, p in pieces_in_box.items():
        p.get_candidates(pieces_in_box)

    # edge_code_checker(pieces_in_box)

    # search any corner to start with
    p = pieces_in_box[0]
    for key, p in pieces_in_box.items():
        if p.is_corner:
            break

    # rotate to top_left corner
    while not (p.edge_codes[0][0] == 0 and p.edge_codes[3][0] == 0):
        p.rotate()

    # max_board_size = len(pieces_in_box)
    max_board_width = 6
    max_board_height = 4
    board = [[None for _ in range(max_board_width)] for _ in range(max_board_height)]
    board[0][0] = p

    pieces_in_box.pop(p.id)

    # start recursive search
    solution = dfs_puzzle([p], {p.id}, pieces_in_box, board, [0, 0])
    display_board(board)
    print(len(solution))


if __name__ == '__main__':
    main(puzzle_file="puzzle_pieces.txt")
