import numpy as np
from .utils import fix

def make_affine(p1, p2, p3):
    x0, y0 = p1
    x1, y1 = p2
    x2, y2 = p3

    mat = np.array([[x1 - x0, y1 - y0, 0], [x2 - x0, y2 - y0, 0], [x0, y0, 1]])
    mat = np.linalg.inv(mat)

    def affine(x, y, origin_width, origin_height, new_width, new_height):
        e = np.matmul(np.array([x, y, 1]), mat)
        tx = e[0] / e[2] * origin_width
        ty = e[1] / e[2] * origin_height
        return fix(tx, ty, origin_width, origin_height)

    return affine
