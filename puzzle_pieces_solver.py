import numpy as np

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

    def rotate(self):
        # 1 step ccw
        self.edge_codes = self.edge_codes[1:] + self.edge_codes[:1]
        self.rims = self.rims[1:] + self.rims[:1]
        self.rotation = next_side(self.rotation)


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


def display_board_simple(board):
    for row in board:
        for p in row:
            if p is not None:
                print("| {:4}   R{} |".format(p.id, p.rotation),
                      end="")
            else:
                print("|            |", end="")
        print("|", end="\n")
    print()


def delta_match(ec1, ec2):
    delta = 0
    for i in [1, 3, 5]:
        x1, y1 = ec1[i], ec1[i + 1]
        x2, y2 = ec2[i], ec2[i + 1]
        delta += ((x2 + x1) ** 2 + (y2 + y1) ** 2) ** 0.5
    return delta


def type_match(ec1, ec2):
    return ec1[0] * ec2[0] < 0


def next_side(s):
    return s + 1 if s < SIDES[-1] else 0


def get_neighbors(pos, board):
    neighbors = [False, False, False, False]
    l_board = len(board[0])
    h_board = len(board)
    # top of position s = 0, s_neighbor = 2
    n_row, n_col = pos[:]
    if n_row == 0:
        neighbors[0] = "rim"
    elif board[n_row - 1][n_col] is not None:
        neighbors[0] = board[n_row - 1][n_col].edge_codes[2]

    # right of position s = 1, s_neighbor = 3
    if n_col == l_board - 1:
        neighbors[1] = "rim"
    elif board[n_row][n_col + 1] is not None:
        neighbors[1] = board[n_row][n_col + 1].edge_codes[3]

    # bottom of position s = 2, s_neighbor = 0
    if n_row == h_board - 1:
        neighbors[2] = "rim"
    elif board[n_row + 1][n_col] is not None:
        neighbors[2] = board[n_row + 1][n_col].edge_codes[0]

    # left of position s = 3, s_neighbor = 1
    if n_col == 0:
        neighbors[3] = "rim"
    elif board[n_row][n_col - 1] is not None:
        neighbors[3] = board[n_row][n_col - 1].edge_codes[1]

    return neighbors


def is_type_matching_to_neighbors(p1, rot, neighbors):
    edge_codes1 = p1.edge_codes[rot:] + p1.edge_codes[:rot]
    for s in SIDES:
        ec2 = neighbors[s]
        if ec2 is not False:
            ec1 = edge_codes1[s]
            if not type_match(ec1, ec2):
                return False
    return True


def main_iterative(puzzle_file):
    pieces_in_box = unbox_pieces(puzzle_file)
    print("number of pieces: ", len(pieces_in_box))

    edge_pieces = [p for p in pieces_in_box.values() if p.is_edge]
    inner_pieces = [p for p in pieces_in_box.values() if not p.is_edge]
    corner_pieces = [p for p in edge_pieces if p.is_corner]

    first_corner = corner_pieces[0]
    edge_pieces.remove(first_corner)
    corner_pieces.remove(first_corner)
    top_frame = [first_corner]
    # rotate to top_left corner
    while not (first_corner.rims[0] and first_corner.rims[3]):
        first_corner.rotate()

    # rotate all edge_pieces with rim to top
    for ep in edge_pieces:
        while not ep.rims[0]:
            ep.rotate()
        if ep.is_corner:
            while not (ep.rims[0] and ep.rims[1]):
                ep.rotate()

    ec1 = first_corner.edge_codes[1]
    top_frame_end = False
    # get best edge piece to connect
    while not top_frame_end:
        best_piece = edge_pieces[0]
        delta = 100_000
        for ep in edge_pieces:
            if not type_match(ec1, ep.edge_codes[3]):
                continue
            delta_i = delta_match(ec1, ep.edge_codes[3])
            if delta_i < delta:
                delta = delta_i
                best_piece = ep
        edge_pieces.remove(best_piece)
        top_frame.append(best_piece)
        ec1 = best_piece.edge_codes[1]
        top_frame_end = best_piece.is_corner
        if top_frame_end:
            corner_pieces.remove(best_piece)

    width = len(top_frame)
    height = int(len(pieces_in_box) / width)

    board = [[None for i in range(width)] for j in range(height)]

    board[0] = top_frame
    # exclude last corners from edge_pieces
    for c in corner_pieces:
        if c in edge_pieces:
            edge_pieces.remove(c)

    # rotate all edge_pieces with rim to right
    for ep in edge_pieces:
        while not ep.rims[1]:
            ep.rotate()

    ec1 = top_frame[-1].edge_codes[2]
    right_frame = []
    right_frame_end = False
    # get best edge piece to connect
    while not right_frame_end:
        best_piece = edge_pieces[0]
        delta = 100_000
        for ep in edge_pieces:
            if not type_match(ec1, ep.edge_codes[0]):
                continue
            delta_i = delta_match(ec1, ep.edge_codes[0])
            if delta_i < delta:
                delta = delta_i
                best_piece = ep
        edge_pieces.remove(best_piece)
        right_frame.append(best_piece)
        ec1 = best_piece.edge_codes[2]
        right_frame_end = len(right_frame) == height - 2

    # add bottom right corner

    for c in corner_pieces:
        while not (c.rims[1] and c.rims[2]):
            c.rotate()
    best_piece = corner_pieces[0]
    delta = 100_000
    ec1 = right_frame[-1].edge_codes[2]
    for c in corner_pieces:
        if not type_match(ec1, c.edge_codes[0]):
            continue
        delta_i = delta_match(ec1, c.edge_codes[0])
        if delta_i < delta:
            delta = delta_i
            best_piece = c
    corner_pieces.remove(best_piece)
    right_frame.append(best_piece)

    for i, p in enumerate(right_frame):
        board[i + 1][-1] = p

    assert len(corner_pieces) == 1

    # rotate all edge_pieces with rim to bottom
    for ep in edge_pieces:
        while not ep.rims[2]:
            ep.rotate()

    ec1 = right_frame[-1].edge_codes[3]
    bottom_frame = []
    bottom_frame_end = False
    # get best edge piece to connect
    while not bottom_frame_end:
        best_piece = edge_pieces[0]
        delta = 100_000
        for ep in edge_pieces:
            if not type_match(ec1, ep.edge_codes[1]):
                continue
            delta_i = delta_match(ec1, ep.edge_codes[1])
            if delta_i < delta:
                delta = delta_i
                best_piece = ep
        edge_pieces.remove(best_piece)
        bottom_frame.append(best_piece)
        ec1 = best_piece.edge_codes[3]
        bottom_frame_end = len(bottom_frame) == width - 2

    # add bottom left corner (last remaining corner_piece)

    for c in corner_pieces:
        while not (c.rims[2] and c.rims[3]):
            c.rotate()
    best_piece = corner_pieces[0]
    assert type_match(bottom_frame[-1].edge_codes[3], best_piece.edge_codes[1])
    corner_pieces.remove(best_piece)
    bottom_frame.append(best_piece)

    for i, p in enumerate(bottom_frame):
        board[-1][width - (i + 2)] = p

    # rotate all edge_pieces with rim to left
    for ep in edge_pieces:
        while not ep.rims[3]:
            ep.rotate()

    ec1 = bottom_frame[-1].edge_codes[0]
    left_frame = []
    left_frame_end = False
    # get best edge piece to connect
    while not left_frame_end:
        best_piece = edge_pieces[0]
        delta = 100_000
        for ep in edge_pieces:
            if not type_match(ec1, ep.edge_codes[2]):
                continue
            delta_i = delta_match(ec1, ep.edge_codes[2])
            if delta_i < delta:
                delta = delta_i
                best_piece = ep
        edge_pieces.remove(best_piece)
        left_frame.append(best_piece)
        ec1 = best_piece.edge_codes[0]
        # left_frame_end = delta_match(ec1, top_frame[0].edge_codes[2]) < best_piece.max_tolerance
        left_frame_end = len(edge_pieces) == 0
    assert type_match(ec1, top_frame[0].edge_codes[2])
    # print(f"delta left frame to top frame: {delta_match(ec1, top_frame[0].edge_codes[2])}")
    for i, p in enumerate(left_frame):
        board[height - 2 - i][0] = p

    # print("frame completed")

    # display_board(board)

    # fit inner pieces
    # loop thru board 1,1 to -1,-1
    for row in range(1, len(board) - 1):
        for col in range(1, len(board[0]) - 1):
            # get neighbors
            neighbors = get_neighbors([row, col], board)

            best_piece = inner_pieces[0]
            best_rotation = 0
            delta_total = 100_000
            for p in inner_pieces:
                for rot in SIDES:
                    delta = 0
                    if is_type_matching_to_neighbors(p, rot, neighbors):
                        edge_codes1 = p.edge_codes[rot:] + p.edge_codes[:rot]
                        for s1 in SIDES:
                            ec1 = edge_codes1[s1]
                            ec2 = neighbors[s1]
                            if ec2 is not False:
                                delta += delta_match(ec1, ec2)
                        if delta < delta_total:
                            best_piece = p
                            best_rotation = rot
                            delta_total = delta
            for rot in range(best_rotation):
                best_piece.rotate()
            board[row][col] = best_piece
            inner_pieces.remove(best_piece)

    print()
    display_board_simple(board)
    print()


if __name__ == '__main__':
    main_iterative(puzzle_file="puzzle_pieces.txt")
