# ThePuzzle

<p>
  Starting from a set of pictures of puzzle pieces in the subdirectory "RV0781508", a text file puzzle_pieces.txt is created containing edges information for each piece (puzzle_image_to_edge_code.py).<br>
  The text file is then processed by the puzzle_pieces_solver.py which prints the resulting board with the ID and the rotation of each piece to the console.<br>
  To simulate puzzle pieces for testing the solver without actual pictures, 
  puzzle_creator.py can generate a text-file for a given size with random edges information.
  Since pictures are not very precisely processed, puzzle_creator.py uses a fuzzy parameter to make simulated edges not perfectly matching.
</p>

<p>
  The solver uses edge type information (corner, edge or inner piece) first to find a starting corner, find the complete frame. <br>
  It then proceeds for each board position to find the matching piece with the matching rotation from the remaining pieces.
</p>

<p>
  The recursive approach puzzle_pieces_solver_dfs.py takes much more time and seems not always to be successful, for some reason (hard for me to debug).
</p>

<p>
  The edges information is a sequence of seven integers. The first being a 1 for a bow pointing into the piece, -1 for a bow pointing out of the piece, and 0 for a straight edge or rim.
  The following six integers give the x and y coordinates of the three characteristic points of the bow in relation to the corner of the edge:<br>
  most left point of the bow, top point of the bow, most right point of the bow<br>
</p>

<p>
  Matching of two edges is tested by check the sum of edges information for 0 or minimum value, e.g. the best candidate for a match is the one with the least value of the sum.
  <br>
  Rim edges and bows pointing in the same direction are skipped before this check.
</p>

<p>
  puzzle_pieces_solver.py test runs: <br>
  with RV0781508 pictures - OK and fast<br>
  with creator 10x15, fuzzy=40 - OK and fast<br>
  with creator 20x30, fuzzy=30 - OK and ca. 4 sec<br>
  with creator 40x60, fuzzy=20 - OK and ca. 71 sec<br>
  with creator 60x90, fuzzy=10 - OK and ca. 400 sec<br>
</p>

<p>
    Any comments on improvements wellcome, especially how to manage the recursive approach which seems to be 
    more computerstylish than just mocking human behaviour as in my iterative approach.
</p>
