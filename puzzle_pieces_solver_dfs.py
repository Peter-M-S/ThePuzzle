import numpy as np
from copy import deepcopy

SIDES = [0, 1, 2, 3]


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
        self.candidates = [not a for a in self.rims]
        self.position = [None, None]

    def rotate(self):
        # 1 step ccw
        self.edge_codes = self.edge_codes[1:] + self.edge_codes[:1]
        self.rims = self.rims[1:] + self.rims[:1]
        self.candidates = self.candidates[1:] + self.candidates[:1]
        self.rotation = next_side(self.rotation)

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
                candidate = False
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
                if candidate:
                    candidates.append(candidate)
            self.candidates[s1] = sorted(candidates)


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

    if row < 0 or col < 0:
        print("out of board")
        return False
    if (row, col) in board:
        print("board not free")
        return False
    return row, col


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
                print(" {:2} | {:8} {} | {:2} |".format(p.edge_codes[3][0], p.id, p.rotation, p.edge_codes[1][0]),
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


def delta_match(ec1, ec2):
    delta = 0
    for i in [1, 3, 5]:
        x1, y1 = ec1[i], ec1[i + 1]
        x2, y2 = ec2[i], ec2[i + 1]
        delta += ((x2 + x1) ** 2 + (y2 + y1) ** 2) ** 0.5
    return delta


def get_rotations(s1, matching_side):
    if s1 == 2:
        return matching_side
    elif s1 == 1:
        return next_side(matching_side)
    elif s1 == 0:
        return next_side(next_side(matching_side))
    else:
        return prev_side(matching_side)


def is_matching_to_neighbors(test_solution, board, next_position, rotations, edge_codes):
    p2_edge_codes = edge_codes[rotations:] + edge_codes[:rotations]
    r, c = next_position

    top = (r - 1, c) if r - 1 >= 0 else False
    bottom = (r + 1, c)
    left = (r, c - 1) if c - 1 >= 0 else False
    right = (r, c + 1)

    if top in board:
        for p1 in test_solution:
            if p1.position == top:
                ec1 = p1.edge_codes[2]
                ec2 = p2_edge_codes[0]
                if ec1[0] * ec2[0] >= 0:
                    return False
                if delta_match(ec1, ec2) > p1.max_tolerance:
                    return False
    if bottom in board:
        for p1 in test_solution:
            if p1.position == bottom:
                ec1 = p1.edge_codes[0]
                ec2 = p2_edge_codes[2]
                if ec1[0] * ec2[0] >= 0:
                    return False
                if delta_match(ec1, ec2) > p1.max_tolerance:
                    return False
    if left in board:
        for p1 in test_solution:
            if p1.position == left:
                ec1 = p1.edge_codes[1]
                ec2 = p2_edge_codes[3]
                if ec1[0] * ec2[0] >= 0:
                    return False
                if delta_match(ec1, ec2) > p1.max_tolerance:
                    return False
    if right in board:
        for p1 in test_solution:
            if p1.position == right:
                ec1 = p1.edge_codes[3]
                ec2 = p2_edge_codes[1]
                if ec1[0] * ec2[0] >= 0:
                    return False
                if delta_match(ec1, ec2) > p1.max_tolerance:
                    return False
    return True


def dfs_puzzle(test_solution, remaining_pieces, board, position, debug_list):
    # solution found?
    if len(remaining_pieces) == 0:
        return test_solution

    p1 = test_solution[-1]
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
        candidates = p1.candidates[s1]
        for c_idx, c in enumerate(candidates):
            if c[1] not in remaining_pieces:
                continue
            next_remaining_pieces = deepcopy(remaining_pieces)
            p2 = next_remaining_pieces[c[1]]

            matching_side = c[2]

            rotations = get_rotations(s1, matching_side)

            # check if other edges match to already existing neighbors
            if not is_matching_to_neighbors(test_solution, board, next_position, rotations, p2.edge_codes):
                continue

            for i in range(rotations):
                p2.rotate()

            p2.position = next_position
            board.add(next_position)

            next_remaining_pieces.pop(p2.id)

            next_test_solution = deepcopy(test_solution)
            next_test_solution += [p2]

            next_debug = deepcopy(debug_list)

            next_debug.append([p1.id, p1.rotation, s1, c_idx, p2.id, p2.rotation])
            print(f"{len(next_debug)} ", end="")
            print("   ".join([f"{_i1}_{_r1}_{_s}_{_c}:{_i2}_{_r2}" for _i1, _r1, _s, _c, _i2, _r2 in next_debug]))

            # add test-piece to solution and step into next level
            solution = dfs_puzzle(next_test_solution, next_remaining_pieces, board, next_position, next_debug)
            if solution:
                return solution
            else:
                board.remove(next_position)


def next_side(s):
    return s + 1 if s < SIDES[-1] else 0


def prev_side(s):
    return s - 1 if s > 0 else SIDES[-1]


def get_board(solution, board):
    rows = max(p[0] for p in board) + 1
    cols = max(p[1] for p in board) + 1

    new_board = []
    for r in range(rows):
        temp = []
        for c in range(cols):
            temp.append(None)
        new_board.append(temp)

    for p in solution:
        r = p.position[0]
        c = p.position[1]
        new_board[r][c] = p

    return new_board


def main_recursive(puzzle_file):
    pieces_in_box = unbox_pieces(puzzle_file)
    print("number of pieces: ", len(pieces_in_box))

    for key, p in pieces_in_box.items():
        p.get_candidates(pieces_in_box)

    # search any corner to start with
    p = pieces_in_box[0]
    for key, p in pieces_in_box.items():
        if p.is_corner:
            break

    # rotate to top_left corner
    while not (p.edge_codes[0][0] == 0 and p.edge_codes[3][0] == 0):
        p.rotate()

    max_board_width = max_board_height = len(pieces_in_box)

    # board = [[None for _ in range(max_board_width)] for _ in range(max_board_height)]
    position = (0, 0)
    p.position = position
    board = {position}

    pieces_in_box.pop(p.id)

    test_solution = [p]

    debug_list = []

    # start recursive search
    solution = dfs_puzzle(test_solution, pieces_in_box, board, [0, 0], debug_list)

    board = get_board(solution, board)

    display_board(board)
    print(len(solution))
    print()


if __name__ == '__main__':
    main_recursive(puzzle_file="puzzle_pieces.txt")

