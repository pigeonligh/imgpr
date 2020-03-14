import numpy as np
import math

from .utils import fix, length

def make_projective(p1, p2, p3, p4):
    x0, y0 = p1
    x1, y1 = p2
    x2, y2 = p3
    x3, y3 = p4

    dx1, dx2, dx3 = x1 - x3, x2 - x3, x0 + x3 - x1 - x2
    dy1, dy2, dy3 = y1 - y3, y2 - y3, y0 + y3 - y1 - y2

    u = np.linalg.det([[dx3, dx2], [dy3, dy2]]) / np.linalg.det([[dx1, dx2], [dy1, dy2]])
    v = np.linalg.det([[dx1, dx3], [dy1, dy3]]) / np.linalg.det([[dx1, dx2], [dy1, dy2]])

    mat = np.array([[x1 - x0 + x1 * u, y1 - y0 + y1 * u, u], [x2 - x0 + x2 * v, y2 - y0 + y2 * v, v], [x0, y0, 1]])
    mat = np.linalg.inv(mat)

    def projective(x, y, origin_width, origin_height, new_width, new_height):
        e = np.matmul(np.array([x, y, 1]), mat)
        tx = e[0] / e[2] * origin_width
        ty = e[1] / e[2] * origin_height
        return fix(tx, ty, origin_width, origin_height)

    return projective
