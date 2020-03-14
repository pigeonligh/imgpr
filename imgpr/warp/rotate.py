import numpy as np
import math

from .utils import fix, rotate_vec

"""
degree = alpha * (length / L)**beta
length = delta * length
"""
def make_spin(alpha, beta, delta=1.0):
    def spin(x, y, origin_width, origin_height, new_width, new_height):
        x = x - new_width / 2
        y = y - new_height / 2

        length = math.sqrt(x * x + y * y)
        rate = length / math.sqrt(new_width * new_width + new_height * new_height)
        tx, ty = rotate_vec(x, y, math.pi * (alpha * rate**beta))

        return fix(origin_width / 2 + tx * delta, origin_height / 2 + ty * delta, origin_width, origin_height)

    return spin
