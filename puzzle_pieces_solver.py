import numpy as np


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
    if n_col == l_board-1:
        neighbors[1] = "rim"
    if n_col + 1 < l_board:
        n_col += 1
        if board[n_row][n_col] is not None:
            neighbors[1] = board[n_row][n_col].edge_codes[3]

    # bottom of position s = 2, s_neighbor = 0
    n_row, n_col = pos[:]
    if n_row == h_board-1:
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
                print(" {:2} | {:8} {} | {:2} |".format(p.edge_codes[3][0], p.name, p.rotation, p.edge_codes[1][0]), end="")
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


def dfs_puzzle(solution, used, remaining_pieces, board, position):
    # solution found?
    if len(remaining_pieces) == 0:
        return solution

    # iterate through all sides of the last piece
    for s in [0, 1, 2, 3]:
        # skip outside edges
        if solution[-1].edge_codes[s][0] == 0:
            continue

        # next position
        next_position = is_valid_next_position(s, position, board)
        if not next_position:
            continue

        neighbors = get_neighbors(next_position, board)
        # print(neighbors)

        # get candidates as list of (total_tolerances, piece_id ,piece_rotation)
        candidates = sorted(get_candidates(neighbors, remaining_pieces))
        piece_id = candidates[0][1]
        rotations = candidates[0][2]
        p = remaining_pieces[piece_id]
        for i in range(rotations):
            p.rotate()
        print(" {} > {} < {} ".format(solution[-1].id, candidates[0][0], p.id))

        board[next_position[0]][next_position[1]] = p

        remaining_pieces.pop(p.id)

        # display_board(board)

        # add test-piece to solution and step into next level
        if intermediate_solution := dfs_puzzle(solution + [p], used.union({p.id}), remaining_pieces, board,
                                               next_position):
            return intermediate_solution


class Piece:

    def __init__(self, piece_id, name, edge_codes, rotation=0):
        self.id: int = piece_id
        self.name: str = name
        self.edge_codes: list = edge_codes
        self.rotation: int = rotation
        self.is_edge, self.is_corner = self.check_edges()
        self.neighbors = [False, False, False, False]

    def rotate(self):
        self.edge_codes = self.edge_codes[1:] + self.edge_codes[:1]
        self.neighbors = self.neighbors[1:] + self.neighbors[:1]
        if self.rotation < 3:
            self.rotation += 1
        else:
            self.rotation = 0

    def check_edges(self):
        c = len([e for e in self.edge_codes if len(e) == 1])
        if c == 2:
            return True, True
        if c > 0:
            return True, False
        return False, False


def main(puzzle_file):
    pieces_in_box = unbox_pieces(puzzle_file)
    print("from here on program has to find solution...")
    print("number of pieces: ", len(pieces_in_box))

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
    print(len(solution))
    display_board(board)


if __name__ == '__main__':
    main(puzzle_file="puzzle_pieces.txt")
