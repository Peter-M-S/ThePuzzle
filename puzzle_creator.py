import random


def main(rows, cols, edge_code_length, fuzzy):
    edge_code_default = [0]
    edge_default = [edge_code_default[:] for e in range(4)]  # todo check why/if [:] is really needed

    # default pieces
    pieces = [[edge_default[:] for c in range(cols)][:] for r in range(rows)]

    # create all needed edge-codes
    inner_edges = (rows - 1) * cols + rows * (cols - 1)
    edge_codes = []
    while len(edge_codes) < inner_edges:
        edge_type = random.choice([-1, 1])
        vertex1 = [random.randint(150, 450), random.randint(100, 200)]
        vertex2 = [random.randint(300, 600), random.randint(200, 400)]
        vertex3 = [random.randint(500, 850), random.randint(100, 200)]
        edge_code = [edge_type] + vertex1 + vertex2 + vertex3
        # eg. edge_code[-1] > inner_edges/2
        if edge_code not in edge_codes:
            edge_codes.append(edge_code)

    # print(edge_codes)
    counter_edge_codes = [[-i for i in edge_code] for edge_code in edge_codes]
    if fuzzy:
        for edge_code in edge_codes:
            for i in range(len(edge_code[1:])):
                edge_code[i+1] += random.randint(-fuzzy, fuzzy)

    # print(counter_edge_codes)

    # transfer edge-codes to pieces
    idx = 0
    for r, row in enumerate(pieces):
        for p, piece in enumerate(row):
            if p < cols - 1:
                piece[0] = edge_codes[idx]
                pieces[r][p + 1][2] = counter_edge_codes[idx]
                idx += 1
            if r < rows - 1:
                piece[3] = edge_codes[idx]
                pieces[r + 1][p][1] = counter_edge_codes[idx]
                idx += 1

    # print(pieces)

    # create text file with puzzle information
    with open(f"puzzle_{rows}x{cols}_ecl{edge_code_length}.txt", "w") as f:
        f.write(f"rows: {rows}\ncols: {cols}\n")
        f.write(f"edge_code_length: {edge_code_length}\n")
        for row in pieces:
            for piece in row:
                f.write(" ".join([str(n) for n in piece]) + "\n")

    # create list of rotated and shuffled pieces
    puzzle_box = []
    idx = 0
    for row in pieces:
        for piece in row:
            i = random.choice([0, 1, 2, 3])
            rot_piece = piece[i:] + piece[:i]
            puzzle_box.append([rot_piece, (idx, i)])
            idx += 1
    random.shuffle(puzzle_box)

    with open("puzzle_pieces.txt", "w") as f:
        for idx, p in enumerate(puzzle_box):
            f.write(" ".join([str(idx), f"piece_{p[1][0]}_rot_{p[1][1]}"]) + "\n")
            for p_line in p[0]:
                f.write(" ".join([str(n) for n in p_line]) + "\n")
            f.write("\n")

    # create dict pairing box_idx: (piece_idx, rotation)
    # box_dict = {key: val[1] for key, val in enumerate(puzzle_box)}

    # reduce puzzle box to pieces only
    # puzzle_box = [val[0] for val in puzzle_box]
    # # print(puzzle_box)
    #
    # # create text file with shuffled and rotated pieces
    # with open(f"puzzle_box.txt", "w") as f:
    #     # f.write(f"edge_code_type: {edge_code_type}\n")
    #     for i, piece in enumerate(puzzle_box):
    #         f.write("\n".join([str(n) for n in piece]) + "\n\n")

    # create text file with shuffled and rotated pieces
    # with open(f"puzzle_box_dict_ecl{edge_code_length}.txt", "w") as f:
    #     f.write(f"{box_dict}")


if __name__ == '__main__':
    random.seed(1000)
    main(rows=4, cols=6, edge_code_length=7, fuzzy=100)
