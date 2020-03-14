import numpy as np
import math

from .utils import fix, length

def make_idw(correlation, point_moves):
    infos = []
    for move in point_moves:
        p2, p1 = move
        x1, y1 = p1
        x2, y2 = p2
        infos.append((p1, (x2 - x1, y2 - y1)))

    def idw(x, y, origin_width, origin_height, new_width, new_height):
        s = 0
        tx = 0
        ty = 0
        for info in infos:
            point, delta = info
            px, py = point
            l = length(x - px, y - py) ** correlation
            s += 1 / (l + 1e-20)
            dx, dy = delta
            tx += dx / (l + 1e-20)
            ty += dy / (l + 1e-20)

        tx = x + tx / (s + 1e-20)
        ty = y + ty / (s + 1e-20)

        return fix(tx, ty, origin_width, origin_height)

    return idw
