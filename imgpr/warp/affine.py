import numpy as np
from .utils import fix

def make_affine(vec_x, vec_y, move_x, move_y):
    mat = np.array([vec_x, vec_y])
    mat = np.linalg.inv(mat)

    def affine(x, y, origin_width, origin_height, new_width, new_height):
        e = np.matmul(np.array([x - move_x, y - move_y]), mat)
        tx = e[0] * origin_width
        ty = e[1] * origin_height
        return fix(tx, ty, origin_width, origin_height)

    return affine
