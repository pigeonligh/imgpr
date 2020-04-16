import numpy as np
from .utils import fix
from .projective import make_projective_by_matrix

def affine_matrix(p1, p2, p3):
    x0, y0 = p1
    x1, y1 = p2
    x2, y2 = p3
    return np.array([[x1 - x0, y1 - y0, 0], [x2 - x0, y2 - y0, 0], [x0, y0, 1]])

def make_affine(p1, p2, p3):
    return make_projective_by_matrix(affine_matrix(p1, p2, p3))
