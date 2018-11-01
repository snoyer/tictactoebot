import math
import argparse

import numpy as np
import cv2

import tictactoebot
import tictactoebot.player

def find_games(img_gray, dist_tolerance=8):
    drawings = []
    new_drawings = []

    grids = tictactoebot.detect_grids(img_gray, dist_tolerance)
    for grid in grids:
        nu,nv = grid.shape
        if (nu,nv) == (3,3) and min(*grid.size)>30 and grid.irregularity<.4:
            symbols = tictactoebot.read_grid_symbols(grid, img_gray)
            yield grid, symbols



def analyze_games(games, play=True):
    old_drawings = []
    new_drawings = []

    for grid, symbols in games:
        for pos,symbol in enumerate(symbols):
            corners = [complex(p.x,p.y) for p in grid.cell_corners(pos)]
            if symbol=='o':
                old_drawings += draw_o(corners)
            if symbol=='x':
                old_drawings += draw_x(corners)
            if symbol=='?':
                old_drawings += draw_unsure(corners)

        if play:
            board = symbols.replace(' ', '-')
            move = tictactoebot.player.play_board(board)
            if move:
                player,pos = move
                new_board = tictactoebot.player.next_board(board, move)

                corners = [complex(p.x,p.y) for p in grid.cell_corners(pos)]
                new_drawings += draw_x(corners) if player == 'x' else draw_o(corners)
                winning = tictactoebot.player.winning_line(new_board)
                if winning:
                    new_drawings += draw_win([grid.cell_corners(pos) for pos in winning[0]])

    return old_drawings, new_drawings




def draw_unsure(cell_corners):
    c = sum(cell_corners)/len(cell_corners)
    corners2 = [c+(p-c)*.4 for p in cell_corners]
    return [
        corners2+corners2[:1],
    ]

def draw_x(cell_corners):
    c = sum(cell_corners)/len(cell_corners)
    corners2 = [c+(p-c)*.4 for p in cell_corners]
    return [
        [corners2[0],corners2[2]],
        [corners2[1],corners2[3]],
    ]

def draw_o(cell_corners, res=20):
    c = sum(cell_corners)/len(cell_corners)
    r = max(abs(c-p) for p in cell_corners) * .25
    circle = [c+complex(r*math.cos(math.pi*2*(i/res)),r*math.sin(math.pi*2*(i/res))) for i in range(res)]
    return [
        circle+circle[:1],
    ]

def draw_win(cells_corners):
    def pts():
        for cell_corners in cells_corners:
            yield from cell_corners

    return [
        fit_segment(pts())
    ]


def draw_polylines(img, drawings, color, width=1, lineType=4, arrows=False):
    for pts in drawings:
        for a,b in pairwise(pts):
            (cv2.arrowedLine if arrows else cv2.line)(img, ij(a), ij(b), color, max(1, int(round(width))), lineType)



def fit_segment(points):
    pts = [make_tuple(p) for p in points]
    vx,vy, cx,cy = cv2.fitLine(np.float32(pts), cv2.DIST_L2, 0, 0.01, 0.01)
    mind = float('nan')
    maxd = float('nan')
    for x,y in pts:
        d = dot(x-cx, y-cy, vx,vy);
        if not (d>mind): mind = d;
        if not (d<maxd): maxd = d;
    return (complex(cx+vx*mind, cy+vy*mind),
            complex(cx+vx*maxd, cy+vy*maxd))

def dot(x1, y1, x2, y2):
    return (x1*x2) + (y1*y2)


def make_tuple(o):
    if isinstance(o, complex):
        return o.real, o.imag
    if isinstance(o, tictactoebot.Point):
        return o.x, o.y
    if isinstance(o, tuple):
        return o
    return tuple(o)


def ij(xy):
    x,y = make_tuple(xy)
    return (int(round(x)),
            int(round(y)))

def pairwise(xs):
   it = iter(xs)
   a = next(it)
   for b in it:
      yield a,b
      a = b


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image')
    parser.add_argument('-o', '--output')

    args = parser.parse_args()

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    old_drawings, new_drawings = analyze_games(find_games(img_gray))

    draw_polylines(img, old_drawings, (255,0,0))
    draw_polylines(img, new_drawings, (0,0,255))

    if args.output:
        cv2.imwrite(args.output, img)
    else:
        cv2.imshow('a', img)
        cv2.waitKey()
